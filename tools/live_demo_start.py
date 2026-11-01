"""Detach the bounded public demo services from the interactive terminal."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--gpu", required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
commands = {
    "api": [sys.executable, "-u", "website/server/serve.py", "--host", "127.0.0.1", "--port", "8765", "--api-only", "--steps", "50",
            "--allow-origin", "https://openspace-lab.github.io", "--allow-origin", "https://b0b8k1ng.github.io",
            "--allow-origin", "http://127.0.0.1:8766", "--output", str(args.root)],
    "tunnel": [str(args.root / "bin/cloudflared"), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8765", "--metrics", "127.0.0.1:8767"],
}
record = {"startedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cwd": str(repo), "cudaVisibleDevices": args.gpu, "services": {}}
env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
for name, command in commands.items():
    log = args.root / f"{name}.log"
    with log.open("ab") as handle:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (args.root / f"{name}.pid").write_text(f"{process.pid}\n")
    record["services"][name] = {"command": command, "pid": process.pid, "log": str(log)}
    print(f"{name}: PID {process.pid}; log {log}")
(args.root / "processes.json").write_text(json.dumps(record, indent=2) + "\n")
