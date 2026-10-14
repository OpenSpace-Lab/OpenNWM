#!/usr/bin/env python3
"""Resume the verified public Kaggle SynMars-TW v1 archive using byte ranges."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests

API = "https://www.kaggle.com/api/v1/datasets/download/bharathudupa22/synmars-tw?datasetVersionNumber=1"
SIZE = 7_481_499_441
MD5 = "d6290970b6f6797adf918b2dd25a3457"
CHUNK = 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--request-mib", default=1, type=int,
                        help="Maximum HTTP range size in MiB")
    parser.add_argument("--fresh-connections", action="store_true",
                        help="Close each completed HTTP connection to avoid sustained-connection throttling")
    parser.add_argument("--redirect-file", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.request_mib <= 16:
        parser.error("workers must be positive; request-mib must be between 1 and 16")
    meta = args.output / "_download"
    meta.mkdir(parents=True, exist_ok=True)
    archive = meta / "SynMars-TW.kaggle-v1.zip"
    partial = archive.with_suffix(".zip.part")
    state_path = meta / "transfer_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "source": API, "size": SIZE, "md5": MD5, "chunk_size": CHUNK, "completed": {}
    }
    assert (state["source"], state["size"], state["md5"]) == (API, SIZE, MD5)
    assert state["chunk_size"] in (CHUNK, 16 * CHUNK)
    if args.redirect_file:
        url = json.loads(args.redirect_file.read_text())["redirect"]
    else:
        response = requests.get(API, allow_redirects=False, timeout=30)
        assert response.status_code in (301, 302, 303, 307, 308), response.status_code
        url = response.headers["Location"]
    started = time.monotonic()
    print(f"PID={os.getpid()} source={API} workers={args.workers} target={archive}", flush=True)
    lock = threading.Lock()
    local = threading.local()
    transferred = 0
    stop_reporter = threading.Event()

    def report_progress():
        while not stop_reporter.wait(30):
            elapsed = time.monotonic() - started
            with lock:
                print(f"Transfer received={transferred:,} bytes this run; "
                      f"{transferred/max(elapsed,1)/1048576:.2f} MiB/s; "
                      f"verified_chunks={len(state['completed'])}; elapsed={elapsed:.0f}s", flush=True)

    def save_state():
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)

    if not archive.exists():
        if state["completed"] and not partial.exists():
            raise RuntimeError("Partial archive is missing; refusing to trust the saved ranges")
        fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, SIZE)
            # Resume only chunks whose bytes still match the recorded SHA-256.
            old_chunk = state["chunk_size"]
            for key, digest in list(state["completed"].items()):
                start = int(key) * old_chunk
                if hashlib.sha256(os.pread(fd, min(old_chunk, SIZE - start), start)).hexdigest() != digest:
                    del state["completed"][key]
            if old_chunk != CHUNK:
                migrated = {}
                for key in state["completed"]:
                    start = int(key) * old_chunk
                    for offset in range(start, min(start + old_chunk, SIZE), CHUNK):
                        migrated[str(offset // CHUNK)] = hashlib.sha256(
                            os.pread(fd, min(CHUNK, SIZE - offset), offset)).hexdigest()
                state.update(chunk_size=CHUNK, completed=migrated)
            save_state()
            initial = sum(min(CHUNK, SIZE - int(k) * CHUNK) for k in state["completed"])
            print(f"Resume verified: {initial:,}/{SIZE:,} bytes", flush=True)

            def fetch(index):
                nonlocal transferred
                if not hasattr(local, "session"):
                    local.session = requests.Session()
                    local.session.trust_env = False
                start = index * CHUNK
                end = min(start + CHUNK, SIZE) - 1
                content = bytearray()
                for part_start in range(start, end + 1, args.request_mib * 1048576):
                    part_end = min(part_start + args.request_mib * 1048576 - 1, end)
                    for attempt in range(1, 7):
                        try:
                            with local.session.get(url, headers={"Range": f"bytes={part_start}-{part_end}"}, timeout=(15, 60)) as response:
                                response.raise_for_status()
                                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {part_start}-{part_end}/{SIZE}":
                                    raise RuntimeError("Unexpected byte-range response")
                                part = response.content
                            if args.fresh_connections:
                                local.session.close()
                            if len(part) != part_end - part_start + 1:
                                raise RuntimeError("Short range response")
                            content.extend(part)
                            with lock:
                                transferred += len(part)
                            break
                        except Exception as exc:
                            local.session.close()
                            print(f"Retry range={index} offset={part_start} attempt={attempt}: {type(exc).__name__}", flush=True)
                            if attempt == 6:
                                raise RuntimeError(f"Range {index} failed after six attempts") from None
                            time.sleep(min(2 ** attempt, 20))
                written = 0
                while written < len(content):
                    written += os.pwrite(fd, content[written:], start + written)
                with lock:
                    state["completed"][str(index)] = hashlib.sha256(content).hexdigest()
                    save_state()
                    done = sum(min(CHUNK, SIZE - int(k) * CHUNK) for k in state["completed"])
                    elapsed = time.monotonic() - started
                    print(f"Range {len(state['completed'])}/{(SIZE + CHUNK - 1) // CHUNK}: {done:,}/{SIZE:,} bytes; {(done-initial)/max(elapsed,1)/1048576:.2f} MiB/s; elapsed={elapsed:.0f}s", flush=True)

            pending = [i for i in range((SIZE + CHUNK - 1) // CHUNK) if str(i) not in state["completed"]]
            reporter = threading.Thread(target=report_progress, daemon=True)
            reporter.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for result in pool.map(fetch, pending):
                        pass
            finally:
                stop_reporter.set()
                reporter.join()
            os.fsync(fd)
        finally:
            os.close(fd)
    source = archive if archive.exists() else partial
    print("Verifying complete archive MD5 and SHA-256...", flush=True)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with source.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            md5.update(data)
            sha256.update(data)
    if source.stat().st_size != SIZE or md5.hexdigest() != MD5:
        raise RuntimeError("Archive size or upstream MD5 mismatch")
    if source == partial:
        partial.replace(archive)
    state.update(status="complete", sha256=sha256.hexdigest(), elapsed_seconds=round(time.monotonic()-started, 2))
    save_state()
    print(f"COMPLETE archive={archive} bytes={SIZE} md5={md5.hexdigest()} sha256={sha256.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
