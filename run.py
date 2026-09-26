#!/usr/bin/env python3
"""Dispatch to the two independent runtimes without importing ML packages."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("OpenNWM\n\nUsage: python run.py {lam,nwm} <command> [options]\n"
              "  python run.py lam --help\n"
              "  python run.py nwm --help\n\n"
              "Set LAM_PYTHON / NWM_PYTHON to choose the runtime interpreter.")
        return 0
    component, *args = sys.argv[1:]
    if component not in {"lam", "nwm"}:
        print(f"Unknown component: {component}", file=sys.stderr)
        return 2
    python = os.environ.get(f"{component.upper()}_PYTHON", sys.executable)
    command = [python, str(root / component / "run.py"), *args]
    env = os.environ.copy()
    env["PYTHON"] = python
    # Inputs remain relative to the caller. Component launchers locate their
    # source/configuration directory themselves.
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
