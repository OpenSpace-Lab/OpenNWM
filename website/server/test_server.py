"""CPU-only validation of the public action and HTTP contracts."""
from collections import deque
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from types import SimpleNamespace

from serve import Application, handler_for
from inference import public_manifest
from trajectory import FRAMES, PRESETS, path_to_actions, validate_points, actions_to_endpoints


class TrajectoryTests(unittest.TestCase):
    def test_forward_distance_and_body_axes(self):
        actions = path_to_actions(PRESETS["forward"])
        self.assertEqual(len(actions), FRAMES)
        self.assertAlmostEqual(sum(a[0] for a in actions), 12.)
        self.assertTrue(all(abs(a[1]) < 1e-8 and abs(a[2]) < 1e-8 for a in actions))

    def test_turns_are_mirrored_and_reconstruct_canvas_endpoint(self):
        left = path_to_actions(PRESETS["left"])
        right = path_to_actions(PRESETS["right"])
        for a, b in zip(left, right):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], -b[1])
            self.assertAlmostEqual(a[2], -b[2])
        self.assertGreater(sum(a[2] for a in left), 0)
        x = y = yaw = 0.
        for dx, dy, angle in left:
            x += math.cos(yaw) * dx - math.sin(yaw) * dy
            y += math.sin(yaw) * dx + math.cos(yaw) * dy
            yaw += angle
        self.assertAlmostEqual(x, 12.)
        self.assertAlmostEqual(y, 6.)
        endpoints = actions_to_endpoints(left)
        for endpoint in endpoints:
            self.assertTrue(all(math.isfinite(value) for value in endpoint))
        self.assertAlmostEqual(endpoints[-1][0], 12.)
        self.assertAlmostEqual(endpoints[-1][1], 6.)
        self.assertAlmostEqual(endpoints[-1][2], yaw)

    def test_invalid_requests_rejected(self):
        for points in ([], [[0, 0]], [[0, 0], [math.nan, 1]], [[0, 0], [2, 1]], [[.5, .5], [0, 1]], [[0, 0], [False, 1]], [[0, 0], [0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_points(points)

    def test_public_provenance_hides_paths_and_preserves_private_record(self):
        private = {"source": {"path": "/private/input.jpg", "sha256": "source-hash"},
                   "checkpoint": {"path": "/private/weights.pt", "sha256": "weight-hash"},
                   "points": PRESETS["forward"], "mode": "autoregressive"}
        public = public_manifest(private)
        self.assertNotIn("path", public["source"])
        self.assertNotIn("path", public["checkpoint"])
        self.assertEqual(public["source"]["sha256"], "source-hash")
        self.assertEqual(public["points"], PRESETS["forward"])
        self.assertIn("path", private["source"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.website = cls.root / "website"
        cls.website.mkdir()
        cls.media_bytes = bytes(range(256))
        (cls.website / "clip.mp4").write_bytes(cls.media_bytes)
        (cls.website / "empty.mp4").write_bytes(b"")
        (cls.website / "server").mkdir()
        (cls.website / "server" / "private.py").write_text("private server fixture")
        cls.website_patch = patch("serve.WEBSITE", cls.website)
        cls.website_patch.start()
        cls.app = Application.__new__(Application)
        cls.app.predictor = SimpleNamespace(loaded=False, steps=50)
        cls.app.allowed_origins = {"https://openspace-lab.github.io", "https://b0b8k1ng.github.io", "http://127.0.0.1:8766"}
        cls.app.api_only = False
        cls.app.rate_history = {}
        cls.app.retention_seconds = 86400
        cls.app.output = cls.root / "jobs"
        output = cls.app.output / "test-media"
        output.mkdir(parents=True)
        (output / "prediction.mp4").write_bytes(cls.media_bytes)
        (output / "manifest.json").write_text(json.dumps({"source": {"path": "/private/input.jpg"}, "checkpoint": {"path": "/private/model.pt"}}))
        cls.app.jobs = {"test-media": {"status": "completed"}}
        cls.app.lock = threading.Lock()
        cls.app.recent = deque()
        cls.app.pending = queue.Queue(maxsize=3)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(cls.app))
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.website_patch.stop()
        cls.temp.cleanup()

    def raw_request(self, path, method="GET", headers=None):
        request = Request(self.base + path, method=method, headers=headers or {})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def request(self, path, payload=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        request = Request(self.base + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        try:
            response = urlopen(request)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_health(self):
        status, result = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertFalse(result["modelLoaded"])

    def test_static_and_generated_media_full_and_head(self):
        for path in ("/clip.mp4", "/api/media/test-media/prediction.mp4"):
            with self.subTest(path=path):
                status, headers, body = self.raw_request(path)
                self.assertEqual(status, 200)
                self.assertEqual(body, self.media_bytes)
                self.assertEqual(headers["Content-Type"], "video/mp4")
                self.assertEqual(headers["Content-Length"], "256")
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertIsNone(headers["Content-Range"])
                for range_value in (None, "bytes=20-30", "bytes=999-"):
                    status, headers, body = self.raw_request(path, "HEAD", {"Range": range_value} if range_value else {})
                    self.assertEqual(status, 200)
                    self.assertEqual(body, b"")
                    self.assertEqual(headers["Content-Length"], "256")
                    self.assertEqual(headers["Accept-Ranges"], "bytes")
                    self.assertIsNone(headers["Content-Range"])

    def test_static_and_generated_media_byte_ranges(self):
        ranges = [("bytes=0-0", 0, 0), ("bytes=7-24", 7, 24),
                  ("bytes=250-", 250, 255), ("bytes=-10", 246, 255),
                  ("bytes=250-999", 250, 255), ("bytes=-999", 0, 255)]
        for path in ("/clip.mp4", "/api/media/test-media/prediction.mp4"):
            for value, start, end in ranges:
                with self.subTest(path=path, range=value):
                    status, headers, body = self.raw_request(path, headers={"Range": value})
                    self.assertEqual(status, 206)
                    self.assertEqual(body, self.media_bytes[start:end + 1])
                    self.assertEqual(headers["Content-Length"], str(end - start + 1))
                    self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/256")
                    self.assertEqual(headers["Accept-Ranges"], "bytes")

    def test_unsatisfiable_and_empty_ranges(self):
        for path in ("/clip.mp4", "/api/media/test-media/prediction.mp4"):
            for value in ("bytes=256-", "bytes=999-1000", "bytes=-0", "bytes=9-8"):
                with self.subTest(path=path, range=value):
                    status, headers, body = self.raw_request(path, headers={"Range": value})
                    self.assertEqual(status, 416)
                    self.assertEqual(headers["Content-Range"], "bytes */256")
                    self.assertEqual(headers["Content-Length"], "0")
                    self.assertEqual(body, b"")
        status, headers, body = self.raw_request("/empty.mp4", headers={"Range": "bytes=0-"})
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], "bytes */0")
        self.assertEqual(headers["Content-Length"], "0")
        self.assertEqual(body, b"")

    def test_unsupported_ranges_and_if_range_fall_back(self):
        for path in ("/clip.mp4", "/api/media/test-media/prediction.mp4"):
            for value in ("bytes=0-2,4-6", "items=1-2", "bytes=-", "bytes=invalid"):
                with self.subTest(path=path, range=value):
                    status, headers, body = self.raw_request(path, headers={"Range": value})
                    self.assertEqual(status, 200)
                    self.assertEqual(body, self.media_bytes)
                    self.assertEqual(headers["Content-Length"], "256")
                    self.assertIsNone(headers["Content-Range"])
            status, headers, body = self.raw_request(path, headers={"Range": "bytes=1-2", "If-Range": '"unknown-etag"'})
            self.assertEqual(status, 200)
            self.assertEqual(body, self.media_bytes)
            status, headers, body = self.raw_request(path, headers={"Range": "bytes=1-2", "If-Range": headers["Last-Modified"]})
            self.assertEqual(status, 206)
            self.assertEqual(body, self.media_bytes[1:3])

    def test_static_cache_and_json_routes_keep_their_contracts(self):
        _, headers, _ = self.raw_request("/clip.mp4")
        status, _, body = self.raw_request("/clip.mp4", headers={"Range": "bytes=1-2", "If-Modified-Since": headers["Last-Modified"]})
        self.assertEqual(status, 304)
        self.assertEqual(body, b"")
        status, _, body = self.raw_request("/api/media/test-media/manifest.json", headers={"Range": "bytes=0-1"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"source": {}, "checkpoint": {}})
        status, headers, body = self.raw_request("/api/health", "HEAD")
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")
        for path in ("/api/media/test-media/secret.txt", "/api/media/unknown/prediction.mp4", "/server/serve.py"):
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    status, _, body = self.raw_request(path, method, {"Range": "bytes=0-1"})
                    self.assertEqual(status, 404)
                    if method == "HEAD":
                        self.assertEqual(body, b"")

    def test_unknown_scene_and_cross_origin(self):
        status, _ = self.request("/api/predict", {"sceneId": "../../secret", "points": PRESETS["forward"]})
        self.assertEqual(status, 400)
        status, _ = self.request("/api/predict", {"sceneId": "indoor-huron", "points": PRESETS["forward"]}, "https://unrelated.example")
        self.assertEqual(status, 403)

    def test_public_preflight_and_api_only_exposure(self):
        headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
        origins = ("https://openspace-lab.github.io", "https://b0b8k1ng.github.io", "http://127.0.0.1:8766")
        for origin in origins:
            with self.subTest(origin=origin):
                headers["Origin"] = origin
                status, response_headers, body = self.raw_request("/api/predict", "OPTIONS", headers)
                self.assertEqual(status, 204)
                self.assertEqual(response_headers["Access-Control-Allow-Origin"], origin)
                self.assertEqual(body, b"")
                status, response_headers, _ = self.raw_request("/api/health", headers={"Origin": origin})
                self.assertEqual(status, 200)
                self.assertEqual(response_headers["Access-Control-Allow-Origin"], origin)
        headers["Origin"] = "https://unrelated.example"
        self.assertEqual(self.raw_request("/api/predict", "OPTIONS", headers)[0], 403)
        self.assertEqual(self.raw_request("/api/health", headers={"Origin": headers["Origin"]})[0], 403)
        with patch.object(self.app, "api_only", True):
            for path in ("/", "/clip.mp4", "/../.env", "/%2e%2e/.env", "/server/serve.py"):
                self.assertEqual(self.raw_request(path)[0], 404)
            for origin in origins:
                with self.subTest(origin=origin, api_only=True):
                    status, response_headers, body = self.raw_request("/api/media/test-media/prediction.mp4", headers={"Origin": origin, "Range": "bytes=0-15"})
                    self.assertEqual(status, 206)
                    self.assertEqual(response_headers["Access-Control-Allow-Origin"], origin)
                    self.assertEqual(body, self.media_bytes[:16])

    def test_rate_limit_and_retention(self):
        isolated = Application.__new__(Application)
        isolated.pending = queue.Queue(maxsize=3)
        isolated.lock = threading.Lock()
        isolated.recent = deque()
        isolated.jobs = {}
        isolated.rate_history = {}
        isolated.output = self.root / "retention"
        isolated.retention_seconds = 86400
        payload = {"sceneId": "indoor-huron", "points": PRESETS["forward"]}
        first = isolated.submit(payload, "client-a")
        with self.assertRaises(queue.Full):
            isolated.submit(payload, "client-a")
        isolated.jobs[first].update(status="completed", createdAt=0)
        directory = isolated.output / first
        directory.mkdir(parents=True)
        (directory / "prediction.mp4").write_bytes(b"test")
        isolated.prune()
        self.assertNotIn(first, isolated.jobs)
        self.assertFalse(directory.exists())

    def test_private_server_files_blocked_after_url_normalization(self):
        paths = ("/server", "/%73erver/", "/server/private.py",
                 "/%73erver/private.py", "/server%2fprivate.py",
                 "/x/../server/private.py", "/x/%2e%2e/server/private.py")
        for path in paths:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    status, _, body = self.raw_request(path, method, {"Range": "bytes=0-5"})
                    self.assertEqual(status, 404)
                    self.assertNotIn(b"private server fixture", body)
                    if method == "HEAD":
                        self.assertEqual(body, b"")

    def test_queue_capacity_and_polling(self):
        for _ in range(3):
            self.app.rate_history.clear()
            status, result = self.request("/api/predict", {"sceneId": "indoor-huron", "points": PRESETS["forward"], "seed": 0})
            self.assertEqual(status, 202)
            status, job = self.request(f"/api/jobs/{result['jobId']}")
            self.assertEqual(status, 200)
            self.assertEqual(job["status"], "queued")
        status, _ = self.request("/api/predict", {"sceneId": "indoor-huron", "points": PRESETS["forward"]})
        self.assertEqual(status, 429)


if __name__ == "__main__":
    unittest.main()
