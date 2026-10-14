#!/usr/bin/env python3
"""Run an auditable, sequential argv-only queue without restarting partial jobs.

Usage (use the explicit nwm Python interpreter):
  reproduce_paper_queue.py --jobs queue.json [--state queue.state.json] [--resume]
  reproduce_paper_queue.py --jobs queue.json --validate
  reproduce_paper_queue.py --self-test

JSON may be a job list or {"jobs": [...], "prerequisite_metadata": [...]}.
Each job has "id", "argv" (string list), "cwd", "log", and optionally:
  "gpu": "4", "min_free_mib": 43000,
  "env": {"OMP_NUM_THREADS": "1"},
  "prerequisite_metadata": ["/path/previous/metadata.json"],
  "completion_metadata": ["/path/this/metadata.json"],
  "checks": [{"path": "/path/comparison.json",
              "pointer": "/all_pixels_identical", "equals": true}].

Metadata prerequisites wait for state="completed", exit_code=0; explicit
failure is fatal. Completion metadata and JSON Pointer checks are checked after
exit 0 and again when resuming a completed job. GPU is a memory gate only; use
argv/env to select a device in the command. No GPU field means no GPU query.
Paths are resolved relative to the queue JSON, except argv (used literally).

Resume skips only proven completed jobs, with unchanged job signatures and
nonempty completion timestamps and existing logs. Failed, launching, running,
or interrupted jobs are never restarted automatically: Table 1 cannot safely
resume partially generated outputs. This runner never terminates other jobs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


class QueueError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def emit(event, **values):
    print(json.dumps({"time_utc": now(), "event": event, **values}, sort_keys=True), flush=True)


def resolved_path(value, base):
    if not isinstance(value, str) or not value:
        raise QueueError(f"Expected nonempty path string, got {value!r}")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def path_list(value, base):
    if not isinstance(value, list):
        raise QueueError("Metadata prerequisites/completions must be a list of paths")
    return [resolved_path(item, base) for item in value]


def load_jobs(path):
    document = json.loads(path.read_text())
    if isinstance(document, list):
        raw_jobs, global_prerequisites = document, []
    elif isinstance(document, dict) and set(document) <= {"jobs", "prerequisite_metadata"}:
        raw_jobs = document.get("jobs")
        global_prerequisites = path_list(document.get("prerequisite_metadata", []), path.parent)
    else:
        raise QueueError("Queue JSON must be a job list or an object with jobs/prerequisite_metadata")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise QueueError("Queue must contain at least one job")
    allowed = {"id", "argv", "cwd", "log", "gpu", "min_free_mib", "env",
               "prerequisite_metadata", "completion_metadata", "checks"}
    jobs = []
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise QueueError(f"Job {index} has unknown fields or is not an object")
        identifier = raw.get("id", f"job_{index:04d}")
        argv = raw.get("argv")
        if not isinstance(identifier, str) or not identifier:
            raise QueueError(f"Job {index}: id must be a nonempty string")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise QueueError(f"Job {identifier}: argv must be a nonempty list of strings")
        if not argv[0] or any("\x00" in item for item in argv):
            raise QueueError(f"Job {identifier}: invalid executable/argument")
        cwd = resolved_path(raw.get("cwd"), path.parent)
        if not Path(cwd).is_dir():
            raise QueueError(f"Job {identifier}: cwd does not exist: {cwd}")
        gpu = raw.get("gpu")
        if gpu is not None and (not isinstance(gpu, (str, int)) or isinstance(gpu, bool)
                                or not str(gpu) or "," in str(gpu)):
            raise QueueError(f"Job {identifier}: gpu must select exactly one GPU")
        minimum = raw.get("min_free_mib", 0)
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or minimum < 0:
            raise QueueError(f"Job {identifier}: min_free_mib must be nonnegative")
        if gpu is None and minimum:
            raise QueueError(f"Job {identifier}: min_free_mib requires gpu")
        environment = raw.get("env", {})
        if not isinstance(environment, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()):
            raise QueueError(f"Job {identifier}: env must map strings to strings")
        checks = raw.get("checks", [])
        if not isinstance(checks, list):
            raise QueueError(f"Job {identifier}: checks must be a list")
        normalized_checks = []
        for check in checks:
            if not isinstance(check, dict) or set(check) != {"path", "pointer", "equals"}:
                raise QueueError(f"Job {identifier}: each check needs path, pointer, equals")
            pointer = check["pointer"]
            if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
                raise QueueError(f"Job {identifier}: checks use JSON Pointer paths")
            normalized_checks.append({**check, "path": resolved_path(check["path"], path.parent)})
        jobs.append({
            "id": identifier, "argv": argv, "cwd": cwd,
            "log": resolved_path(raw.get("log"), path.parent),
            "gpu": None if gpu is None else str(gpu), "min_free_mib": minimum,
            "env": environment,
            "prerequisite_metadata": global_prerequisites + path_list(
                raw.get("prerequisite_metadata", []), path.parent),
            "completion_metadata": path_list(raw.get("completion_metadata", []), path.parent),
            "checks": normalized_checks,
        })
    if len({job["id"] for job in jobs}) != len(jobs):
        raise QueueError("Job ids must be unique")
    if len({job["log"] for job in jobs}) != len(jobs):
        raise QueueError("Each job must have a separate log file")
    return jobs


def metadata_state(path, allow_wait):
    try:
        metadata = json.loads(Path(path).read_text())
    except FileNotFoundError:
        if allow_wait:
            return "missing"
        raise QueueError(f"Required completion metadata is absent: {path}")
    if not isinstance(metadata, dict):
        raise QueueError(f"Metadata must be a JSON object: {path}")
    state, exit_code = metadata.get("state"), metadata.get("exit_code")
    if state == "completed" and type(exit_code) is int and exit_code == 0:
        return "completed"
    if allow_wait and state in {"prepared", "pending", "running", "waiting"} and exit_code is None:
        return str(state)
    raise QueueError(f"Metadata is not successful: {path}: state={state!r}, exit_code={exit_code!r}")


def json_pointer(document, pointer):
    value = document
    if not pointer:
        return value
    for escaped in pointer[1:].split("/"):
        key = escaped.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def verify_completion(job):
    for path in job["completion_metadata"]:
        metadata_state(path, allow_wait=False)
    for check in job["checks"]:
        try:
            actual = json_pointer(json.loads(Path(check["path"]).read_text()), check["pointer"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise QueueError(f"Cannot evaluate JSON check {check}: {error}") from error
        # JSON true must not silently match the integer 1.
        if type(actual) is not type(check["equals"]) or actual != check["equals"]:
            raise QueueError(f"JSON check failed: {check['path']} {check['pointer']}: "
                             f"expected {check['equals']!r}, found {actual!r}")


def log_tail(path, lines=25):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-lines:]
    except FileNotFoundError:
        return []


def inspect_gpu(gpu):
    command = ["nvidia-smi", "--query-gpu=memory.free,memory.total",
               "--format=csv,noheader,nounits", "--id", gpu]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise QueueError(f"GPU inspection failed ({result.returncode}): {result.stderr.strip()}")
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise QueueError(f"GPU selection must return exactly one GPU, received: {result.stdout!r}")
    try:
        free, total = (int(value.strip()) for value in rows[0].split(","))
    except ValueError as error:
        raise QueueError(f"Unrecognized nvidia-smi memory output: {result.stdout!r}") from error
    return {"gpu": gpu, "free_mib": free, "total_mib": total,
            "inspected_utc": now(), "command": command}


def save_state(path, state):
    state["updated_utc"] = now()
    write_json(path, state)


def wait_ready(job, record, state, state_path, poll_seconds):
    last_status = None
    while True:
        dependencies = {path: metadata_state(path, allow_wait=True)
                        for path in job["prerequisite_metadata"]}
        if any(value != "completed" for value in dependencies.values()):
            record.update(status="waiting_prerequisites", prerequisite_states=dependencies)
            status = (record["status"], canonical_hash(dependencies))
        elif job["gpu"] is not None:
            gpu = inspect_gpu(job["gpu"])
            record["gpu_preflight"] = gpu
            if job["min_free_mib"] > gpu["total_mib"]:
                raise QueueError(f"Requested {job['min_free_mib']} MiB exceeds total {gpu['total_mib']} MiB")
            if gpu["free_mib"] >= job["min_free_mib"]:
                record["status"] = "ready"
                save_state(state_path, state)
                return
            record["status"] = "waiting_gpu"
            status = (record["status"], gpu["free_mib"])
        else:
            record["status"] = "ready"
            save_state(state_path, state)
            return
        save_state(state_path, state)
        if status != last_status:
            emit("waiting", job=job["id"], status=record["status"],
                 prerequisites=dependencies, gpu=record.get("gpu_preflight"))
            last_status = status
        time.sleep(poll_seconds)


def initial_state(args, jobs, state_path):
    digest = canonical_hash(jobs)
    if state_path.exists():
        if not args.resume:
            raise QueueError(f"State exists; inspect it and use --resume: {state_path}")
        state = json.loads(state_path.read_text())
        if state.get("schema_version") != 1 or state.get("jobs_sha256") != digest:
            raise QueueError("Resume refused: queue specification differs from recorded state")
        if len(state.get("jobs", [])) != len(jobs):
            raise QueueError("Resume refused: job count mismatch")
        for job, record in zip(jobs, state["jobs"]):
            if record.get("job_sha256") != canonical_hash(job) or record.get("id") != job["id"]:
                raise QueueError(f"Resume refused: job signature mismatch: {job['id']}")
            if record.get("status") == "completed":
                if (type(record.get("exit_code")) is not int or record["exit_code"] != 0
                        or not record.get("finished_utc") or not Path(job["log"]).is_file()):
                    raise QueueError(f"Resume refused: invalid completed status: {job['id']}")
                verify_completion(job)
            elif record.get("status") not in {"pending", "ready", "waiting_prerequisites", "waiting_gpu"}:
                raise QueueError(f"Resume refused: job {job['id']} has status {record.get('status')!r}; "
                                 "inspect existing processes and outputs before making a new queue. "
                                 "Partial Table 1 output must not be restarted in place.")
        return state
    if args.resume:
        raise QueueError(f"Resume requested but no recorded state exists: {state_path}")
    return {"schema_version": 1, "created_utc": now(), "status": "prepared",
            "jobs_file": str(args.jobs), "jobs_sha256": digest,
            "jobs": [{"id": job["id"], "job_sha256": canonical_hash(job),
                      "specification": job, "status": "pending"} for job in jobs]}


def execute_queue(args, jobs, state_path, lock):
    state = initial_state(args, jobs, state_path)
    state.update(status="running", queue_pid=os.getpid(), started_or_resumed_utc=now())
    save_state(state_path, state)
    current_record = None
    try:
        for job, record in zip(jobs, state["jobs"]):
            current_record = record
            if record["status"] == "completed":
                emit("resume_skip_completed", job=job["id"], exit_code=record["exit_code"])
                continue
            wait_ready(job, record, state, state_path, args.poll_seconds)
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Opening exclusively catches output from an earlier unrecorded
            # attempt. Never silently append/restart an incomplete command.
            with log_path.open("x") as log:
                record.update(status="launching", started_utc=now(),
                              shell_command=shlex.join(job["argv"]))
                save_state(state_path, state)
                emit("launch", job=job["id"], cwd=job["cwd"], argv=job["argv"],
                     shell_command=record["shell_command"], log=job["log"])
                process = subprocess.Popen(job["argv"], cwd=job["cwd"],
                                           env=dict(os.environ, **job["env"]),
                                           stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),))
                record.update(status="running", pid=process.pid)
                save_state(state_path, state)
                emit("started", job=job["id"], pid=process.pid, log=job["log"])
                while process.poll() is None:
                    try:
                        process.wait(timeout=args.poll_seconds)
                    except subprocess.TimeoutExpired:
                        record.update(last_observed_running_utc=now(), log_tail=log_tail(log_path))
                        save_state(state_path, state)
                return_code = process.returncode
            record.update(exit_code=return_code, finished_utc=now(), log_tail=log_tail(log_path))
            if return_code != 0:
                record["status"] = "failed"
                state.update(status="failed", finished_utc=now(), failed_job=job["id"])
                save_state(state_path, state)
                emit("failed", job=job["id"], pid=record["pid"], exit_code=return_code,
                     log=job["log"], log_tail=record["log_tail"])
                return return_code if 1 <= return_code <= 125 else 1
            record["status"] = "checking"
            save_state(state_path, state)
            verify_completion(job)
            record["status"] = "completed"
            save_state(state_path, state)
            emit("completed", job=job["id"], pid=record["pid"], exit_code=0,
                 log=job["log"], log_tail=record["log_tail"])
        state.update(status="completed", exit_code=0, finished_utc=now())
        save_state(state_path, state)
        emit("queue_completed", state=str(state_path), jobs=len(jobs), exit_code=0)
        return 0
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}", finished_utc=now())
        if current_record is not None:
            if current_record["status"] in {"launching", "running", "checking"}:
                current_record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            current_record["error"] = state["error"]
        save_state(state_path, state)
        raise


def self_test():
    """Small real CPU subprocess fixture; it never invokes nvidia-smi."""
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="opennwm-queue-fixture-") as temporary:
        root = Path(temporary)
        prerequisite = root / "prior.json"
        write_json(prerequisite, {"state": "completed", "exit_code": 0})
        marker = root / "executions.txt"
        completed = root / "done.json"
        job = {"id": "success", "cwd": str(root), "log": str(root / "success.log"),
               "argv": [sys.executable, "-c",
                        "from pathlib import Path; import json; "
                        f"Path({str(marker)!r}).write_text('once'); "
                        f"Path({str(completed)!r}).write_text(json.dumps({{'complete': True}})); print('OK')"],
               "prerequisite_metadata": [str(prerequisite)],
               "checks": [{"path": str(completed), "pointer": "/complete", "equals": True}]}
        jobs_path = root / "success.json"
        write_json(jobs_path, [job])
        command = [sys.executable, script, "--jobs", str(jobs_path), "--poll-seconds", "0.05"]
        first = subprocess.run(command, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr + first.stdout
        original_mtime = marker.stat().st_mtime_ns
        resumed = subprocess.run([*command, "--resume"], capture_output=True, text=True)
        assert resumed.returncode == 0, resumed.stderr + resumed.stdout
        assert marker.stat().st_mtime_ns == original_mtime
        failure_path = root / "failure.json"
        fail_job = {"id": "failure", "cwd": str(root), "log": str(root / "failure.log"),
                    "argv": [sys.executable, "-c", "print('expected failure'); raise SystemExit(7)"]}
        later = {"id": "must_not_run", "cwd": str(root), "log": str(root / "later.log"),
                 "argv": [sys.executable, "-c", "raise RuntimeError('should never run')"]}
        write_json(failure_path, [fail_job, later])
        fail_command = [sys.executable, script, "--jobs", str(failure_path), "--poll-seconds", "0.05"]
        failed = subprocess.run(fail_command, capture_output=True, text=True)
        assert failed.returncode == 7, failed.stderr + failed.stdout
        assert not (root / "later.log").exists()
        refused = subprocess.run([*fail_command, "--resume"], capture_output=True, text=True)
        assert refused.returncode != 0 and "Resume refused" in refused.stderr
        # Tampering with a completed output invalidates resume as well.
        write_json(completed, {"complete": False})
        bad_check = subprocess.run([*command, "--resume"], capture_output=True, text=True)
        assert bad_check.returncode != 0 and "JSON check failed" in bad_check.stderr
        state = json.loads(failure_path.with_suffix(".state.json").read_text())
        assert state["jobs"][0]["exit_code"] == 7 and state["jobs"][1]["status"] == "pending"
        waiting_metadata = root / "running_dependency.json"
        write_json(waiting_metadata, {"state": "running", "pid": os.getpid()})
        waiting_path = root / "waiting.json"
        write_json(waiting_path, [{"id": "wait_then_run", "cwd": str(root),
                                  "log": str(root / "wait.log"),
                                  "argv": [sys.executable, "-c", "print('dependency complete')"],
                                  "prerequisite_metadata": [str(waiting_metadata)]}])
        waiting_command = [sys.executable, script, "--jobs", str(waiting_path), "--poll-seconds", "0.05"]
        waiting = subprocess.Popen(waiting_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 5
        waiting_state_path = waiting_path.with_suffix(".state.json")
        while time.monotonic() < deadline:
            if waiting_state_path.exists():
                waiting_state = json.loads(waiting_state_path.read_text())
                if waiting_state["jobs"][0]["status"] == "waiting_prerequisites":
                    break
            time.sleep(0.01)
        else:
            raise AssertionError("CPU prerequisite fixture did not enter waiting state")
        duplicate = subprocess.run([*waiting_command, "--resume"], capture_output=True, text=True)
        assert duplicate.returncode != 0 and "queue lock" in duplicate.stderr
        assert not (root / "wait.log").exists()
        write_json(waiting_metadata, {"state": "completed", "exit_code": 0})
        stdout, stderr = waiting.communicate(timeout=5)
        assert waiting.returncode == 0, stderr + stdout
        emit("self_test_passed", checks=["CPU execution", "successful resume without restart",
             "failure stops queue", "failed resume refused", "completed JSON check revalidated",
             "running prerequisite waits then releases", "duplicate queue lock refused"], gpu_used=False)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--jobs", type=Path)
    result.add_argument("--state", type=Path)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--validate", action="store_true", help="Validate the queue specification without starting work")
    result.add_argument("--poll-seconds", type=float, default=15.0, help="Resource/process polling interval, >0 and <=30")
    result.add_argument("--self-test", action="store_true", help="Run isolated real-subprocess CPU fixtures")
    return result


def main():
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    if args.jobs is None:
        raise QueueError("--jobs is required")
    if not 0 < args.poll_seconds <= 30:
        raise QueueError("--poll-seconds must be >0 and <=30")
    args.jobs = args.jobs.resolve()
    jobs = load_jobs(args.jobs)
    if args.validate:
        emit("validated", jobs=len(jobs), jobs_sha256=canonical_hash(jobs), gpu_started=False)
        return 0
    state_path = (args.state or args.jobs.with_suffix(".state.json")).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if str(state_path) in {job["log"] for job in jobs} or state_path == args.jobs:
        raise QueueError("State file must differ from queue JSON and all job logs")
    with state_path.with_suffix(state_path.suffix + ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueError(f"Another process holds the queue lock: {state_path}") from error
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        return execute_queue(args, jobs, state_path, lock)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QueueError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
