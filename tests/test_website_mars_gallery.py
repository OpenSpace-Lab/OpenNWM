"""Publication boundaries for estimated-action Mars examples (CPU only)."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_website_mars_gallery as gallery
from prepare_synmars_pose_actions import select_samples, project_actions


def write_json(path, value):
    path.write_text(json.dumps(value))


class MarsPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        dataset = cls.root / "SynMars-TW"
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), (160, 80, 40)).save(buffer, format="PNG")
        pixels = buffer.getvalue()
        specs = [(route, start) for route in range(1, 6) for start in (1, 81, 161, 241, 321, 401)]
        specs += [(8, start) for start in (1, 201, 401)]
        specs += [(9, start) for start in (1, 181, 361, 541, 721, 901)]
        for route, start in specs:
            directory = dataset / "images" / ("testing" if route == 9 else "training")
            directory.mkdir(parents=True, exist_ok=True)
            frames = range(start, start + 77) if (route, start) in {(1, 1), (9, 1)} else range(start, start + 80, 4)
            for frame in frames:
                (directory / f"forward_{route}_{frame:04d}_l.png").write_bytes(pixels)
        samples = select_samples(dataset)
        pose_model = {"model": "test pose estimator", "scale_policy": "relative units"}
        transforms = np.tile(np.eye(4)[:3], (20, 1, 1))
        transforms[:, 2, 3] = -np.arange(20) * .25
        actions, diagnostics = project_actions(transforms)
        for sample in samples:
            hashes = {path: gallery.sha256(path) for path in sample["context_source_paths"] + sample["gt_source_paths"]}
            pose_path = cls.root / f"{sample['id']}.json"
            input_frames, input_hashes = sample["frame_indices"], hashes
            if sample["id"] in {"synmars-forward-1-0001", gallery.SELECTED_IDS[0]}:
                input_frames = list(range(1, 78))
                input_paths = [str(Path(sample["context_source_paths"][0]).with_name(
                    f"forward_{sample['route_id']}_{frame:04d}_l.png")) for frame in input_frames]
                input_hashes = {path: gallery.sha256(path) for path in input_paths}
                dense_transforms = np.tile(np.eye(4)[:3], (77, 1, 1))
                dense_transforms[:, 2, 3] = -np.arange(77) * .0625
                write_json(pose_path.with_suffix(".dense_retry.json"), {
                    "sample_id": sample["id"], "provenance": pose_model,
                    "pose_input_frame_indices": input_frames, "pose_input_source_sha256": input_hashes,
                    "extrinsics_w2c": dense_transforms.tolist(),
                    "intrinsics": np.tile(np.eye(3), (77, 1, 1)).tolist()})
            write_json(pose_path, {"sample_id": sample["id"], "provenance": pose_model,
                                  "frame_indices": sample["frame_indices"], "source_sha256": hashes,
                                  "pose_input_frame_indices": input_frames, "pose_input_source_sha256": input_hashes,
                                  "extrinsics_w2c": transforms.tolist(),
                                  "intrinsics": np.tile(np.eye(3), (20, 1, 1)).tolist()})
            sample.update(actions=actions, pose_provenance=str(pose_path), pose_sha256=gallery.sha256(pose_path),
                          pose_input_frame_indices=input_frames, pose_input_source_sha256=input_hashes,
                          source_sha256=hashes, pose_diagnostics=diagnostics)
        selection_path = cls.root / "selection.json"
        write_json(selection_path, {"gallery_sample_ids": gallery.SELECTED_IDS,
                                   "samples": select_samples(dataset)})
        cls.data = {"completed_utc": "2026-10-29T00:00:00Z", "samples": samples,
                    "gallery_sample_ids": gallery.SELECTED_IDS, "selection_sha256": gallery.sha256(selection_path),
                    "pose_model": pose_model,
                    "split_policy": {"train_routes": [1, 2, 3, 4, 5], "validation_routes": [8], "test_routes": [9]}}
        cls.data_path = cls.root / "data_manifest.json"
        write_json(cls.data_path, cls.data)
        run = cls.root / "run"
        (run / "checkpoints").mkdir(parents=True)
        checkpoint = run / "checkpoints/latest.pth.tar"
        checkpoint.write_bytes(b"test checkpoint")
        cls.metrics_path = run / "metrics.json"
        training = {"status": "complete", "data_manifest_sha256": gallery.sha256(cls.data_path),
                    "checkpoint_sha256": gallery.sha256(checkpoint), "source_checkpoint_sha256": "base-hash",
                    "settings": {"steps": 1000}, "training_pairs": 480, "validation_pairs": 48,
                    "split_windows": gallery.EXPECTED_SPLITS,
                    "train_sample_ids": [s["id"] for s in samples if s["split"] == "train"],
                    "val_sample_ids": [s["id"] for s in samples if s["split"] == "val"]}
        write_json(cls.metrics_path, training)
        eval_samples = [s for s in samples if s["split"] == "cross_scene_test"]
        eval_path = cls.root / "eval_manifest.json"
        write_json(eval_path, {"samples": eval_samples})
        inference_samples = copy.deepcopy(eval_samples)
        for sample in inference_samples:
            sample.update(context_paths=sample["context_source_paths"], initial_path=sample["context_source_paths"][-1],
                          gt_paths=sample["gt_source_paths"], prediction_paths=sample["gt_source_paths"])
        cls.inference_path = cls.root / "inference.json"
        write_json(cls.inference_path, {"completed_utc": "2026-10-29T00:01:00Z", "generated_frames": 96,
                   "signature": {"checkpoint": str(checkpoint), "checkpoint_sha256": gallery.sha256(checkpoint),
                                 "seed": 0, "diffusion_steps": 250, "vae_encoding": "posterior_mean",
                                 "context_size": 4, "image_size": 224,
                                 "samples_manifest": {"path": str(eval_path), "sha256": gallery.sha256(eval_path)}},
                   "samples": inference_samples})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_export_retains_existing_cards_and_discloses_estimation(self):
        site = self.root / "site"
        (site / "content").mkdir(parents=True)
        originals = [{"id": "earth-example"}, {"id": "moon-example", "environment": "lunar"}]
        write_json(site / "content/demos.json", {"demos": originals})
        calls = []

        def fake_encode(paths, target, fps):
            calls.append((paths, fps))
            target.write_bytes(b"encoded test video")
            Image.open(paths[4]).save(target.with_suffix(".webp"))
            return {"sha256": gallery.sha256(target), "frameCount": len(paths)}

        with patch.object(gallery, "encode", side_effect=fake_encode):
            gallery.export_gallery(self.inference_path, self.metrics_path, self.data_path, site)
        catalog = gallery.read_json(site / "content/demos.json")
        self.assertEqual(catalog["demos"][:2], originals)
        self.assertEqual([d["sampleId"] for d in catalog["demos"][2:]], gallery.SELECTED_IDS)
        self.assertEqual(len(calls), 6)
        for paths, fps in calls:
            self.assertEqual((len(paths), fps), (17, 4))
        for demo in catalog["demos"][2:]:
            self.assertNotIn("horizon", demo)
            self.assertTrue(demo["conditioningTrajectory"]["estimated"])
            self.assertEqual(demo["conditioningTrajectory"]["positions"][0], [0, 0])
            public_path = site / demo["sourceManifest"]
            public = gallery.read_json(public_path)
            self.assertNotIn(str(self.root), public_path.read_text())
            self.assertEqual(len(public["estimated_poses"]["extrinsics_w2c"]), 20)
            pose = public["estimated_poses"]
            self.assertTrue(all(path.startswith("SynMars-TW/images/") for path in pose["pose_input_source_sha256"]))
            if demo["sampleId"] == gallery.SELECTED_IDS[0]:
                self.assertEqual(len(pose["pose_input_frame_indices"]), 77)
                self.assertEqual(pose["dense_retry"]["selected_indices"], list(range(0, 77, 4)))
            else:
                self.assertEqual(pose["pose_input_frame_indices"], public["frame_indices"])
                self.assertNotIn("dense_retry", pose)
            self.assertEqual(public["source_anchor_frame"], public["frame_indices"][3])
            self.assertIn("offline", public["protocol"])
            self.assertEqual(gallery.sha256(public_path), demo["sourceManifestSha256"])

    def test_changed_gallery_selection_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["gallery_sample_ids"][0:1] = [gallery.TEST_IDS[1]]
        with self.assertRaisesRegex(ValueError, "three predeclared"):
            gallery.validate_data(data)

    def test_wrong_source_stride_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["samples"][0]["frame_indices"][4] += 1
        with self.assertRaisesRegex(ValueError, "stride four"):
            gallery.validate_data(data)

    def test_source_change_is_rejected_even_outside_displayed_samples(self):
        path = Path(self.data["samples"][0]["gt_source_paths"][0])
        original = path.read_bytes()
        try:
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "Source image hash mismatch"):
                gallery.validate_inputs(self.inference_path, self.metrics_path, self.data_path)
        finally:
            path.write_bytes(original)

    def test_intermediate_pose_input_change_is_rejected_outside_displayed_samples(self):
        sample = self.data["samples"][0]
        path = Path(next(path for path in sample["pose_input_source_sha256"] if path not in sample["source_sha256"]))
        original = path.read_bytes()
        try:
            path.write_bytes(b"changed intermediate pose input")
            with self.assertRaisesRegex(ValueError, "Source image hash mismatch"):
                gallery.validate_inputs(self.inference_path, self.metrics_path, self.data_path)
        finally:
            path.write_bytes(original)

    def test_pose_input_outside_window_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["samples"][0]["pose_input_frame_indices"].append(78)
        with self.assertRaisesRegex(ValueError, "same window"):
            gallery.validate_data(data)

    def test_pose_input_from_other_route_is_rejected(self):
        data = copy.deepcopy(self.data)
        hashes = data["samples"][0]["pose_input_source_sha256"]
        path = next(path for path in hashes if "0002_l.png" in path)
        hashes[path.replace("forward_1_", "forward_8_")] = hashes.pop(path)
        with self.assertRaisesRegex(ValueError, "declared route and frame interval"):
            gallery.validate_data(data)

    def test_selected_sources_must_match_pose_inputs(self):
        data = copy.deepcopy(self.data)
        sample = data["samples"][0]
        sample["pose_input_source_sha256"][sample["context_source_paths"][0]] = "wrong-digest"
        with self.assertRaisesRegex(ValueError, "unchanged subset"):
            gallery.validate_data(data)

    def test_resolved_source_overlap_between_splits_is_rejected(self):
        train = Path(self.data["samples"][0]["context_source_paths"][0])
        validation = next(s for s in self.data["samples"] if s["split"] == "val")
        original = train.read_bytes()
        try:
            train.unlink()
            train.symlink_to(validation["context_source_paths"][0])
            with self.assertRaisesRegex(ValueError, "overlaps different evaluation splits"):
                gallery.validate_data(self.data)
        finally:
            train.unlink()
            train.write_bytes(original)

    def test_dense_retry_selected_camera_disagreement_is_rejected(self):
        sample = self.data["samples"][0]
        path = Path(sample["pose_provenance"]).with_suffix(".dense_retry.json")
        original = path.read_bytes()
        try:
            dense = gallery.read_json(path)
            dense["extrinsics_w2c"][0][0][3] += 1
            write_json(path, dense)
            with self.assertRaisesRegex(ValueError, "do not match the dense retry"):
                gallery.validate_inputs(self.inference_path, self.metrics_path, self.data_path)
        finally:
            path.write_bytes(original)

    def test_pose_artifact_input_provenance_must_match_manifest(self):
        sample = copy.deepcopy(self.data["samples"][0])
        sample["pose_input_frame_indices"] = sample["frame_indices"]
        with self.assertRaisesRegex(ValueError, "source does not match the sample"):
            gallery.public_pose(sample, self.data["pose_model"])

    def test_wrong_checkpoint_is_rejected(self):
        path = self.metrics_path.parent / "checkpoints/latest.pth.tar"
        original = path.read_bytes()
        try:
            path.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "verified trained checkpoint"):
                gallery.validate_inputs(self.inference_path, self.metrics_path, self.data_path)
        finally:
            path.write_bytes(original)

    def test_actions_must_be_derived_from_exported_poses(self):
        sample = copy.deepcopy(self.data["samples"][-1])
        sample["actions"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "estimated camera poses"):
            gallery.public_pose(sample, self.data["pose_model"])


if __name__ == "__main__":
    unittest.main()
