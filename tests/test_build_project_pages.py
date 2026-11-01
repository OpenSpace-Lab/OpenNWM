"""Offline release-boundary checks using one tiny reviewed Go2 clip."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_project_pages as pages


class BuildProjectPagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="project-pages-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repository"
        self.site = self.repository / "website"
        self.site.mkdir(parents=True)
        self.output = self.base / "public"
        self.required = (pages.ROOT_FILES | pages.CONTENT_FILES | pages.ROOT_ASSETS
                         | {pages.REVIEW, "assets/go2/report.json"})
        for name in self.required:
            self.write(name, b"{}" if name.endswith(".json") else b"fixture\n")
        self.clip = {
            "id": "clip_one",
            "video": "assets/go2/clip_one/comparison-reviewed.mp4",
            "poster": "assets/go2/clip_one/poster-reviewed.webp",
            "downloads": {
                "metrics": "assets/go2/clip_one/metrics.json",
                "trajectories": "assets/go2/clip_one/trajectories.csv",
            },
            "privacy": {"reviewed": True},
        }
        payloads = {
            self.clip["video"]: b"reviewed video fixture\n",
            self.clip["poster"]: b"reviewed poster fixture\n",
            self.clip["downloads"]["metrics"]: b'{"error":0.1}\n',
            self.clip["downloads"]["trajectories"]: b"x,y\n0,1\n",
        }
        for name, data in payloads.items():
            self.write(name, data)
        self.selected_files = set(payloads)
        self.review_row = {"id": self.clip["id"], "downloads": {}}
        for key in ("video", "poster"):
            name = self.clip[key]
            self.review_row[key] = name
            self.review_row[key + "_sha256"] = hashlib.sha256(payloads[name]).hexdigest()
        for key, name in self.clip["downloads"].items():
            self.review_row["downloads"][key] = {
                "path": name, "bytes": len(payloads[name]),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
        self.manifest = {"version": 3, "status": "ready", "clips": [self.clip]}
        self.review = {
            "status": "passed", "selection_preserved": True,
            "clip_count": 1, "clips": [self.review_row],
        }
        self.save_manifests()
        delivery = json.dumps({"schemaVersion": 1, "enabled": False})
        self.write("content/media-delivery.json", delivery.encode())
        self.write("index.html", (
            '<!doctype html><script id="media-delivery-config" type="application/json">'
            + delivery + "</script>"
        ).encode())
        self.write("assets/hero/fixture.webp", b"public hero fixture\n")
        self.expected_files = self.required | self.selected_files | {"assets/hero/fixture.webp"}
        self.git("init", "-q")
        self.git("add", "--all")

    def write(self, name, data):
        path = self.site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repository), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def save_manifests(self):
        self.write("content/go2.json", json.dumps(self.manifest).encode())
        self.write(pages.REVIEW, json.dumps(self.review).encode())

    def assert_no_output(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob(".public-*")), [])

    def add_fallback(self):
        url = ("https://raw.githubusercontent.com/B0B8K1ng/b0b8k1ng.github.io/"
               + "a" * 40 + "/" + self.clip["video"])
        self.clip["fallback_video"] = url
        self.review_row["fallback_video"] = url
        self.review_row["fallback_video_sha256"] = self.review_row["video_sha256"]
        self.save_manifests()
        return url

    def test_build_copies_only_the_reviewed_public_selection(self):
        excluded = {
            "server/serve.py", "content/REDESIGN_VALIDATION.md",
            "content/unknown.json", "assets/internal/results.json",
            "assets/hero/unsafe.py", "assets/hero/unsafe.html", "assets/hero/unsafe.sh",
            "assets/hero/.secret.json", "private.py",
            "assets/go2/clip_one/comparison.mp4", "assets/go2/clip_one/poster.webp",
        }
        excluded.update("assets/go2/unselected/" + name for name in
                        ("comparison.mp4", "poster.webp", "metrics.json", "trajectories.csv"))
        for name in excluded:
            self.write(name, b"must stay private\n")
        self.git("add", "--all")
        untracked = "assets/hero/untracked.webp"
        self.write(untracked, b"not approved for release\n")

        report = pages.build(self.site, self.output)

        actual = {str(path.relative_to(self.output)) for path in self.output.rglob("*")
                  if path.is_file()}
        self.assertEqual(actual, self.expected_files | {".nojekyll"})
        for name in self.expected_files:
            self.assertEqual((self.output / name).read_bytes(), (self.site / name).read_bytes())
        for name in excluded | {untracked}:
            self.assertTrue((self.site / name).is_file())
            self.assertFalse((self.output / name).exists())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["clips"], 1)
        self.assertEqual(report["selected_media_sha256_verified"], 2)
        self.assertEqual(report["download_sha256_verified"], 2)
        self.assertEqual(report["excluded_tracked_files"], len(excluded))
        self.assertEqual(report["files"], len(actual))

    def test_changed_media_fails_review_without_creating_output(self):
        for key in ("video", "poster"):
            with self.subTest(media=key):
                path = self.site / self.clip[key]
                original = path.read_bytes()
                path.write_bytes(original + b"changed after review")
                with self.assertRaisesRegex(RuntimeError, "Privacy review does not match"):
                    pages.build(self.site, self.output)
                self.assert_no_output()
                path.write_bytes(original)

    def test_changed_download_fails_review_without_creating_output(self):
        self.write(self.clip["downloads"]["metrics"], b'{"error":99}')
        with self.assertRaisesRegex(ValueError, "Privacy review does not match download"):
            pages.build(self.site, self.output)
        self.assert_no_output()

    def test_symlink_file_and_ancestor_are_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "private.webp").write_bytes(b"private bytes")
        link = self.site / "assets/hero/link.webp"
        link.symlink_to(outside / "private.webp")
        self.git("add", "--all")
        with self.assertRaisesRegex(ValueError, "Symlink is not allowed"):
            pages.build(self.site, self.output)
        self.assert_no_output()

        link.unlink()
        self.git("add", "--all")
        directory = self.site / "assets/hero/nested"
        self.write("assets/hero/nested/private.webp", b"initial tracked bytes")
        self.git("add", "--all")
        shutil.rmtree(directory)
        directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink is not allowed"):
            pages.build(self.site, self.output)
        self.assert_no_output()

    def test_symlink_website_root_is_rejected(self):
        alias = self.base / "website-alias"
        alias.symlink_to(self.site, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Website root cannot be a symlink"):
            pages.build(alias, self.output)
        self.assert_no_output()

    def test_manifest_path_escape_is_rejected(self):
        outside = self.base / "private.mp4"
        outside.write_bytes(b"private bytes")
        for name in (str(outside), "../../private.mp4",
                     "assets/go2/clip_one/../private.mp4",
                     "assets/go2/clip_one/./comparison-reviewed.mp4"):
            with self.subTest(path=name):
                self.clip["video"] = name
                self.save_manifests()
                with self.assertRaisesRegex(ValueError, "Invalid selected Go2 video|Unsafe public path"):
                    pages.build(self.site, self.output)
                self.assert_no_output()
                self.assertEqual(outside.read_bytes(), b"private bytes")

    def test_reviewed_pinned_fallback_omits_only_the_output_master(self):
        url = self.add_fallback()

        report = pages.build(self.site, self.output)

        self.assertFalse((self.output / self.clip["video"]).exists())
        self.assertTrue((self.site / self.clip["video"]).is_file())
        for name in self.selected_files - {self.clip["video"]}:
            self.assertEqual((self.output / name).read_bytes(), (self.site / name).read_bytes())
        copied_manifest = json.loads((self.output / "content/go2.json").read_text())
        self.assertEqual(copied_manifest["clips"][0]["fallback_video"], url)
        self.assertEqual(report["pinned_fallback_masters_omitted"], 1)
        self.assertEqual(report["selected_media_sha256_verified"], 2)

    def test_fallback_must_match_the_reviewed_url_and_hash(self):
        for field in ("fallback_video", "fallback_video_sha256"):
            with self.subTest(field=field):
                self.add_fallback()
                self.review_row[field] = "mismatched review value"
                self.save_manifests()
                with self.assertRaisesRegex(ValueError, "Fallback is not the reviewed pinned public video"):
                    pages.build(self.site, self.output)
                self.assert_no_output()

    def test_copy_failure_removes_partial_output(self):
        original_copy = pages.shutil.copyfile
        copied = []

        def failing_copy(source, destination):
            if copied:
                raise OSError("simulated copy failure")
            result = original_copy(source, destination)
            copied.append(Path(destination))
            self.assertTrue(Path(destination).is_file())
            return result

        with patch.object(pages.shutil, "copyfile", side_effect=failing_copy):
            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                pages.build(self.site, self.output)
        self.assertEqual(len(copied), 1)
        self.assert_no_output()


if __name__ == "__main__":
    unittest.main()
