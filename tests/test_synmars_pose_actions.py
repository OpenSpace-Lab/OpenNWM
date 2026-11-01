"""Check the physical conventions and route isolation of Mars pseudo-actions."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_synmars_pose_actions import project_actions, select_samples, sha256, validate_cached_pose_inputs


def camera_extrinsics(forward, left=None, yaw=None):
    """Build known camera-to-world poses, then return OpenCV world-to-camera."""
    forward = np.asarray(forward, dtype=float)
    left = np.zeros_like(forward) if left is None else np.asarray(left, dtype=float)
    yaw = np.zeros_like(forward) if yaw is None else np.asarray(yaw, dtype=float)
    cameras = np.tile(np.eye(4), (len(forward), 1, 1))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    # Positive navigation yaw turns camera z towards negative camera x (left).
    cameras[:, 0, 0] = cosine
    cameras[:, 0, 2] = -sine
    cameras[:, 2, 0] = sine
    cameras[:, 2, 2] = cosine
    cameras[:, 0, 3] = -left
    cameras[:, 2, 3] = forward
    return np.linalg.inv(cameras)[:, :3]


class SynMarsProjectionTests(unittest.TestCase):
    def test_forward_motion_is_cumulative_from_fourth_context_frame(self):
        actions, diagnostics = project_actions(camera_extrinsics(np.arange(20)))
        expected = np.column_stack([np.arange(1, 17) * .25, np.zeros((16, 2))])
        np.testing.assert_allclose(actions, expected, atol=1e-12)
        np.testing.assert_allclose(diagnostics["normalized_adjacent_steps"], .25)
        self.assertAlmostEqual(diagnostics["translation_scale"], .25)

    def test_negative_camera_x_is_positive_left(self):
        actions, _ = project_actions(camera_extrinsics(np.zeros(20), np.arange(20)))
        np.testing.assert_allclose(np.asarray(actions)[:, 0], 0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(actions)[:, 1], np.arange(1, 17) * .25)
        np.testing.assert_allclose(np.asarray(actions)[:, 2], 0, atol=1e-12)

    def test_anchor_heading_rotates_translation_and_yaw_has_correct_sign(self):
        for turn_sign in (-1, 1):
            with self.subTest(turn_sign=turn_sign):
                yaw = turn_sign * np.arange(20) * .1
                actions, _ = project_actions(camera_extrinsics(np.arange(20), yaw=yaw))
                distance = np.arange(1, 17) * .25
                expected = np.column_stack([
                    np.cos(yaw[3]) * distance,
                    -np.sin(yaw[3]) * distance,
                    yaw[4:] - yaw[3],
                ])
                np.testing.assert_allclose(actions, expected, atol=1e-12)

    def test_heading_crossing_pi_uses_short_signed_rotation(self):
        yaw = np.r_[0, 1, 2, 3, np.linspace(3.2, 3.8, 16)]
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                actions, _ = project_actions(camera_extrinsics(np.arange(20), yaw=sign * yaw))
                np.testing.assert_allclose(np.asarray(actions)[:, 2], sign * (yaw[4:] - yaw[3]), atol=1e-12)

    def test_unknown_monocular_translation_scale_does_not_change_actions(self):
        forward = np.arange(20, dtype=float)
        left = .03 * forward ** 2
        yaw = .02 * forward
        baseline, _ = project_actions(camera_extrinsics(forward, left, yaw))
        for factor in (.01, 100):
            with self.subTest(factor=factor):
                scaled, _ = project_actions(camera_extrinsics(forward * factor, left * factor, yaw))
                np.testing.assert_allclose(scaled, baseline, atol=1e-12)

    def test_global_world_gauge_does_not_change_actions(self):
        original = camera_extrinsics(np.arange(20), np.arange(20) * .1, np.arange(20) * .02)
        transform = np.eye(4)
        angle = .7
        transform[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        transform[:3, 3] = [20, -5, 3]
        transformed = np.matmul(original, np.linalg.inv(transform))
        expected, _ = project_actions(original)
        actual, _ = project_actions(transformed)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_stationary_and_pure_rotation_clips_cannot_define_translation_scale(self):
        for yaw in (np.zeros(20), np.arange(20) * .1):
            with self.subTest(yaw=yaw.tolist()):
                with self.assertRaisesRegex(ValueError, "Degenerate"):
                    project_actions(camera_extrinsics(np.zeros(20), yaw=yaw))

    def test_large_single_pose_jump_is_rejected(self):
        forward = np.arange(20, dtype=float)
        forward[10:] += 100
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            project_actions(camera_extrinsics(forward))

    def test_nonfinite_and_wrong_shape_poses_are_rejected(self):
        extrinsics = camera_extrinsics(np.arange(20))
        with self.assertRaisesRegex(ValueError, "twenty finite"):
            project_actions(extrinsics[:19])
        extrinsics[5, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "twenty finite"):
            project_actions(extrinsics)

    def test_reflection_is_not_a_valid_camera_rotation(self):
        extrinsics = camera_extrinsics(np.arange(20))
        extrinsics[:, 1, 1] = -1
        with self.assertRaisesRegex(ValueError, "rotation"):
            project_actions(extrinsics)

    def test_normalized_step_must_be_positive_and_finite(self):
        extrinsics = camera_extrinsics(np.arange(20))
        for step in (0, -.25, np.nan, np.inf):
            with self.subTest(nominal_step=step):
                with self.assertRaises(ValueError):
                    project_actions(extrinsics, nominal_step=step)


class SynMarsPoseCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "images" / "training"
        self.directory.mkdir(parents=True)
        for frame in range(161, 238):
            self.path(frame).write_bytes(f"source-frame-{frame}".encode())
        indices = list(range(161, 238, 4))
        paths = [str(self.path(frame)) for frame in indices]
        self.sample = {"id": "synmars-forward-3-0161", "route_id": 3, "frame_indices": indices,
                       "context_source_paths": paths[:4], "gt_source_paths": paths[4:]}

    def path(self, frame):
        return self.directory / f"forward_3_{frame:04d}_l.png"

    def cached_inputs(self, dense):
        indices = list(range(161, 238)) if dense else self.sample["frame_indices"].copy()
        return {"sample_id": self.sample["id"], "frame_indices": self.sample["frame_indices"].copy(),
                "pose_input_frame_indices": indices,
                "pose_input_source_sha256": {str(self.path(frame)): sha256(self.path(frame)) for frame in indices}}

    def test_unchanged_sparse_and_dense_inputs_can_be_reused(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                saved = self.cached_inputs(dense)
                hashes, indices = validate_cached_pose_inputs(saved, self.sample)
                self.assertEqual(hashes, saved["pose_input_source_sha256"])
                self.assertEqual(indices, saved["pose_input_frame_indices"])
                self.assertEqual(len(hashes), 77 if dense else 20)

    def test_tampered_dense_only_rgb_invalidates_cache(self):
        saved = self.cached_inputs(dense=True)
        selected_hashes = {str(self.path(frame)): sha256(self.path(frame)) for frame in self.sample["frame_indices"]}
        self.path(194).write_bytes(b"changed sky image")
        self.assertNotIn(194, self.sample["frame_indices"])
        self.assertEqual(selected_hashes, {path: sha256(path) for path in selected_hashes})
        with self.assertRaisesRegex(ValueError, "source content changed"):
            validate_cached_pose_inputs(saved, self.sample)

    def test_tampered_selected_rgb_invalidates_sparse_cache(self):
        saved = self.cached_inputs(dense=False)
        self.path(193).write_bytes(b"changed selected frame")
        with self.assertRaisesRegex(ValueError, "source content changed"):
            validate_cached_pose_inputs(saved, self.sample)

    def test_invalid_order_missing_or_outside_input_indices_are_rejected(self):
        baseline = self.cached_inputs(dense=True)
        invalid_indices = [baseline["pose_input_frame_indices"][::-1],
                           baseline["pose_input_frame_indices"][:-1],
                           list(range(162, 239))]
        for indices in invalid_indices:
            with self.subTest(indices=indices):
                saved = copy.deepcopy(baseline)
                saved["pose_input_frame_indices"] = indices
                with self.assertRaisesRegex(ValueError, "input frame indices"):
                    validate_cached_pose_inputs(saved, self.sample)

    def test_missing_extra_or_wrong_route_input_paths_are_rejected(self):
        baseline = self.cached_inputs(dense=True)
        for change in ("missing", "extra", "wrong_route"):
            with self.subTest(change=change):
                saved = copy.deepcopy(baseline)
                hashes = saved["pose_input_source_sha256"]
                if change == "missing":
                    del hashes[str(self.path(194))]
                elif change == "extra":
                    hashes[str(self.path(238))] = "unused"
                else:
                    hashes[str(self.directory / "forward_9_0194_l.png")] = hashes.pop(str(self.path(194)))
                with self.assertRaisesRegex(ValueError, "input paths"):
                    validate_cached_pose_inputs(saved, self.sample)

    def test_cache_from_different_sample_or_selected_frames_is_rejected(self):
        for field, value in (("sample_id", "synmars-forward-9-0161"), ("frame_indices", list(range(162, 239, 4)))):
            with self.subTest(field=field):
                saved = self.cached_inputs(dense=True)
                saved[field] = value
                with self.assertRaisesRegex(ValueError, "sample or selected frame indices"):
                    validate_cached_pose_inputs(saved, self.sample)


class SynMarsSplitTests(unittest.TestCase):
    def test_preselected_routes_and_frames_are_disjoint_between_splits(self):
        specifications = {
            "train": (range(1, 6), (1, 81, 161, 241, 321, 401)),
            "val": ((8,), (1, 201, 401)),
            "cross_scene_test": ((9,), (1, 181, 361, 541, 721, 901)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for routes, starts in specifications.values():
                for route in routes:
                    directory = root / "images" / ("testing" if route == 9 else "training")
                    directory.mkdir(parents=True, exist_ok=True)
                    for start in starts:
                        for index in range(start, start + 80, 4):
                            (directory / f"forward_{route}_{index:04d}_l.png").touch()
            samples = select_samples(root)
            self.assertEqual(len({sample["id"] for sample in samples}), 39)
            selected_routes = {}
            selected_files = {}
            for split, (routes, starts) in specifications.items():
                group = [sample for sample in samples if sample["split"] == split]
                self.assertEqual(len(group), len(routes) * len(starts))
                selected_routes[split] = {sample["route_id"] for sample in group}
                self.assertEqual(selected_routes[split], set(routes))
                selected_files[split] = set()
                for sample in group:
                    context = sample["context_source_paths"]
                    future = sample["gt_source_paths"]
                    self.assertEqual(len(context), 4)
                    self.assertEqual(len(future), 16)
                    self.assertTrue(set(context).isdisjoint(future))
                    self.assertEqual(sample["prediction_steps"], list(range(1, 17)))
                    np.testing.assert_array_equal(np.diff(sample["frame_indices"]), 4)
                    self.assertIsNone(sample["source_fps"])
                    self.assertNotIn("reverse", sample["source_scene"])
                    self.assertTrue(selected_files[split].isdisjoint(context + future))
                    selected_files[split].update(context + future)
            for left, right in (("train", "val"), ("train", "cross_scene_test"), ("val", "cross_scene_test")):
                self.assertTrue(selected_routes[left].isdisjoint(selected_routes[right]))
                self.assertTrue(selected_files[left].isdisjoint(selected_files[right]))
            gallery = [sample for sample in samples if sample["gallery_selected"]]
            self.assertEqual([sample["frame_indices"][0] for sample in gallery], [1, 361, 721])
            self.assertTrue(all(sample["split"] == "cross_scene_test" for sample in gallery))
            Path(samples[-1]["gt_source_paths"][-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                select_samples(root)


if __name__ == "__main__":
    unittest.main()
