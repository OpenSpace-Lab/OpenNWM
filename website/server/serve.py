"""Local same-origin website and queued OpenNWM prediction HTTP service."""
import argparse
from collections import deque
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

from inference import Predictor, public_manifest
from scenes import SCENE_MAP, WEBSITE, DEFAULT_RUN, DEFAULT_OUTPUT
from trajectory import validate_points


class Application:
    def __init__(self, args):
        self.predictor = Predictor(args.run_dir, args.steps)
        self.output = Path(args.output).resolve() / "jobs"
        self.output.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.pending = queue.Queue(maxsize=3)
        self.lock = threading.Lock()
        self.recent = deque()
        self.allowed_origins = set(args.allow_origin)
        self.api_only = args.api_only
        self.rate_history = {}
        self.retention_seconds = 24 * 60 * 60
        self.restore_jobs()
        threading.Thread(target=self.work, daemon=True).start()

    def restore_jobs(self):
        """Restore completed outputs after a process restart, with bounded storage."""
        for directory in sorted(self.output.iterdir(), key=lambda item: item.stat().st_mtime):
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or not directory.is_dir():
                continue
            manifest = directory / "manifest.json"
            if not manifest.is_file() or not (directory / "prediction.mp4").is_file():
                shutil.rmtree(directory)
                continue
            try:
                data = json.loads(manifest.read_text())
            except (ValueError, OSError):
                continue
            job_id = directory.name
            self.jobs[job_id] = {"jobId": job_id, "sceneId": data["sceneId"], "status": "completed", "progress": 1,
                                "createdAt": manifest.stat().st_mtime, "protocol": data["protocol"], "error": None,
                                "video": f"/api/media/{job_id}/prediction.mp4", "poster": f"/api/media/{job_id}/poster.webp",
                                "manifest": f"/api/media/{job_id}/manifest.json"}
            self.recent.append(job_id)
        self.prune()

    def prune(self):
        now = time.time()
        for job_id in list(self.recent):
            job = self.jobs[job_id]
            if job["status"] in ("queued", "running"):
                continue
            if len(self.recent) <= 100 and now - job.get("createdAt", now) < self.retention_seconds:
                continue
            self.recent.remove(job_id)
            self.jobs.pop(job_id)
            if re.fullmatch(r"[0-9a-f]{32}", job_id):
                shutil.rmtree(self.output / job_id, ignore_errors=True)

    def submit(self, payload, client="local"):
        if not isinstance(payload, dict) or set(payload) - {"sceneId", "points", "seed"}:
            raise ValueError("Expected sceneId, points, and optional seed.")
        scene_id = payload.get("sceneId")
        if not isinstance(scene_id, str) or scene_id not in SCENE_MAP:
            raise ValueError("Unknown scene.")
        points = validate_points(payload.get("points"))
        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
            raise ValueError("Seed must be an integer from 0 to 2147483647.")
        job_id = uuid.uuid4().hex
        job = {"jobId": job_id, "sceneId": scene_id, "status": "queued", "progress": 0,
               "video": None, "poster": None, "manifest": None, "protocol": None, "error": None,
               "createdAt": time.time()}
        with self.lock:
            if self.pending.full():
                raise queue.Full()
            now = time.monotonic()
            for key, entries in list(self.rate_history.items()):
                while entries and now - entries[0] > 3600:
                    entries.popleft()
                if not entries:
                    self.rate_history.pop(key)
            history = self.rate_history.setdefault(client, deque())
            if len(history) >= 20 or (history and now - history[-1] < 15) or sum(map(len, self.rate_history.values())) >= 120:
                raise queue.Full()
            history.append(now)
            self.prune()
            self.jobs[job_id] = job
            self.recent.append(job_id)
            self.pending.put_nowait((job_id, scene_id, {"points": points, "seed": seed}))
        return job_id

    def work(self):
        while True:
            job_id, scene_id, request = self.pending.get()
            job = self.jobs[job_id]
            try:
                job["status"] = "running"
                manifests = self.predictor.predict(SCENE_MAP[scene_id], [request], [self.output / job_id], lambda value: job.update(progress=value))
                job.update(status="completed", progress=1, video=f"/api/media/{job_id}/prediction.mp4",
                           poster=f"/api/media/{job_id}/poster.webp", manifest=f"/api/media/{job_id}/manifest.json",
                           protocol=manifests[0]["protocol"])
                for frame in (self.output / job_id).glob("[0-9][0-9][0-9].png"):
                    frame.unlink()
            except Exception:
                traceback.print_exc()
                job.update(status="failed", error="Prediction failed. Please try again later.")
            finally:
                self.pending.task_done()
                with self.lock:
                    self.prune()


def handler_for(app):
    class Handler(SimpleHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def allowed_origin(self):
            origin = self.headers.get("Origin")
            return not origin or origin in app.allowed_origins or (not app.api_only and urlsplit(origin).netloc == self.headers.get("Host"))

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEBSITE), **kwargs)

        def json(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def end_headers(self):
            origin = self.headers.get("Origin")
            if origin and self.allowed_origin():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
            self.send_header("X-Content-Type-Options", "nosniff")
            if getattr(self, "_accept_ranges", False):
                self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def do_OPTIONS(self):
            if urlsplit(self.path).path != "/api/predict" or not self.allowed_origin():
                return self.json(403, {"error": "Origin or endpoint not allowed."})
            requested_headers = {value.strip().lower() for value in self.headers.get("Access-Control-Request-Headers", "").split(",") if value.strip()}
            if self.headers.get("Access-Control-Request-Method") != "POST" or requested_headers - {"content-type"}:
                return self.json(403, {"error": "Request method or headers not allowed."})
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def range_head(self, target):
            """Return (handled, stream) for a single GET byte range.

            Unsupported/malformed ranges fall back to the full representation.
            HEAD ignores Range, as required by HTTP. Static conditional requests
            remain with SimpleHTTPRequestHandler's existing cache handling.
            """
            value = self.headers.get("Range", "")
            if self.command != "GET" or not value or self.headers.get("If-Modified-Since"):
                return False, None
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip(), re.IGNORECASE)
            if not match or not any(match.groups()):
                return False, None
            try:
                first, last = (int(part) if part else None for part in match.groups())
            except ValueError:
                return False, None
            try:
                source = target.open("rb")
            except OSError:
                self.send_error(404, "File not found")
                return True, None
            stat = os.fstat(source.fileno())
            size = stat.st_size
            if_range = self.headers.get("If-Range")
            if if_range and if_range != self.date_time_string(stat.st_mtime):
                source.close()
                return False, None
            self._accept_ranges = True
            start = max(0, size - last) if first is None else first
            end = size - 1 if first is None or last is None else min(last, size - 1)
            if size == 0 or start >= size or end < start or (first is None and last == 0):
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True, None
            source.seek(start)
            self._range_length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(target)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(self._range_length))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return True, source

        def send_head(self):
            target = Path(self.translate_path(self.path)).resolve()
            # Check the same decoded, normalized filesystem path used to serve
            # static files; raw URL prefixes miss encoded names and dot segments.
            if target.is_relative_to((WEBSITE / "server").resolve()):
                self.json(404, {"error": "Not found."})
                return None
            if target.is_file():
                handled, source = self.range_head(target)
                if handled:
                    return source
                self._accept_ranges = True
            return super().send_head()

        def copyfile(self, source, output):
            remaining = getattr(self, "_range_length", None)
            try:
                if remaining is None:
                    return super().copyfile(source, output)
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # Seeking routinely cancels the browser's previous media read.
                pass

        def media(self, target):
            handled, source = self.range_head(target)
            if not handled:
                source = target.open("rb")
                stat = os.fstat(source.fileno())
                self._accept_ranges = True
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(stat.st_size))
                self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
                self.end_headers()
            if source:
                with source:
                    if self.command != "HEAD":
                        self.copyfile(source, self.wfile)

        def do_HEAD(self):
            return self.do_GET()

        def do_GET(self):
            self._accept_ranges = False
            self._range_length = None
            path = urlsplit(self.path).path
            if not self.allowed_origin():
                return self.json(403, {"error": "Origin not allowed."})
            if path == "/api/health":
                return self.json(200, {"status": "ok", "live": True, "modelLoaded": app.predictor.loaded, "queueLength": app.pending.qsize(), "maxQueue": 3, "model": "OpenNWM", "sampler": "DDPM", "diffusionSteps": app.predictor.steps})
            if path == "/api/scenes":
                return self.json(200, json.loads((WEBSITE / "content/playground.json").read_text()))
            if path.startswith("/api/jobs/"):
                job = app.jobs.get(path.removeprefix("/api/jobs/"))
                return self.json(200 if job else 404, job or {"error": "Unknown job."})
            if path.startswith("/api/media/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[3] not in app.jobs or parts[4] not in ("prediction.mp4", "poster.webp", "manifest.json"):
                    return self.json(404, {"error": "Unknown media."})
                target = app.output / parts[3] / parts[4]
                if not target.is_file():
                    return self.json(404, {"error": "Media is not ready."})
                if parts[4] == "manifest.json":
                    return self.json(200, public_manifest(json.loads(target.read_text())))
                return self.media(target)
            if app.api_only or path.startswith("/api/") or path.startswith("/server/"):
                return self.json(404, {"error": "Not found."})
            return super().do_HEAD() if self.command == "HEAD" else super().do_GET()

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                return self.json(404, {"error": "Not found."})
            if not self.allowed_origin():
                return self.json(403, {"error": "Origin not allowed."})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Use a fixed Content-Length.")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Request body must be between 1 and 32768 bytes.")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send application/json.")
                payload = json.loads(self.rfile.read(size))
                client = self.headers.get("CF-Connecting-IP", self.client_address[0]) if self.client_address[0] == "127.0.0.1" else self.client_address[0]
                job_id = app.submit(payload, client=client[:64])
                return self.json(202, {"jobId": job_id})
            except queue.Full:
                return self.json(429, {"error": "The GPU is busy or the request limit was reached. Please try again later."})
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                return self.json(400, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--api-only", action="store_true", help="Expose only allowlisted API endpoints, with no static files.")
    parser.add_argument("--allow-origin", action="append", default=[])
    args = parser.parse_args()
    app = Application(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(app))
    print(f"OpenNWM website + API: http://{args.host}:{args.port} (model loads on first request)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
