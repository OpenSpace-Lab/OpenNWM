"""Coordinate and clock regression tests for recorded planetary evaluation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.planetary_eval_data import (
    _nearest_indices,
    _read_lusnar_scene,
    local_action,
    wrap_angle,
)


class PlanarActionTests(unittest.TestCase):
    def test_quarter_turn_heading_maps_world_motion_to_body_forward(self):
        # A rover heading north moves north: this must remain forward motion.
        action = local_action([10, 20], np.pi / 2, [10, 24], np.pi / 2, spacing=2)
        np.testing.assert_allclose(action, [2, 0, 0], atol=1e-12)

    def test_left_and_right_displacements_and_turns_keep_their_signs(self):
        left = local_action([0, 0], 0, [0, 2], np.pi / 2, spacing=2)
        right = local_action([0, 0], 0, [0, -2], -np.pi / 2, spacing=2)
        np.testing.assert_allclose(left, [0, 1, np.pi / 2], atol=1e-12)
        np.testing.assert_allclose(right, [0, -1, -np.pi / 2], atol=1e-12)

    def test_yaw_crossing_pi_uses_short_rotation_in_both_directions(self):
        left = local_action([0, 0], np.pi - .1, [0, 0], -np.pi + .1)
        right = local_action([0, 0], -np.pi + .1, [0, 0], np.pi - .1)
        np.testing.assert_allclose(left, [0, 0, .2], atol=1e-12)
        np.testing.assert_allclose(right, [0, 0, -.2], atol=1e-12)

    def test_body_increment_rollout_reaches_direct_endpoint_after_turns(self):
        # Three forward legs with quarter turns require rotating each local
        # increment before accumulating it; summing local XY would be wrong.
        positions = np.asarray([[0, 0], [2, 0], [2, 3], [-2, 3]], dtype=float)
        headings = np.asarray([0, np.pi / 2, np.pi, -np.pi / 2])
        increments = [local_action(positions[i], headings[i], positions[i + 1],
                                   headings[i + 1], spacing=2) for i in range(3)]
        np.testing.assert_allclose(increments,
                                   [[1, 0, np.pi / 2], [1.5, 0, np.pi / 2],
                                    [2, 0, np.pi / 2]], atol=1e-12)
        accumulated = 0j
        heading = 0.0
        for forward, left, turn in increments:
            accumulated += complex(forward, left) * np.exp(1j * heading)
            heading += turn
        endpoint = [accumulated.real, accumulated.imag, float(wrap_angle(heading))]
        direct = local_action(positions[0], headings[0], positions[-1], headings[-1], spacing=2)
        np.testing.assert_allclose(endpoint, [-1, 1.5, -np.pi / 2], atol=1e-12)
        np.testing.assert_allclose(endpoint, direct, atol=1e-12)


class TimestampAndPoseTests(unittest.TestCase):
    def test_nearest_matching_preserves_nanoseconds_above_float_precision(self):
        base = 1_693_274_724_086_398_208
        # These queries lie on opposite sides of a nearest-neighbor boundary,
        # but become the same value if converted to float64 at epoch scale.
        self.assertEqual(float(base + 149), float(base + 151))
        actual = _nearest_indices([base, base + 300, base + 600],
                                  [base - 1, base + 149, base + 151, base + 601])
        np.testing.assert_array_equal(actual, [0, 0, 1, 2])

    def test_equal_distance_match_has_stable_earlier_frame_tie_break(self):
        base = 1_693_274_724_086_398_208
        np.testing.assert_array_equal(_nearest_indices([base, base + 300],
                                                       [base + 150]), [0])

    def _write_scene(self, root, *, conflict=False):
        scene = root / "Moon_1"
        (scene / "image0").mkdir(parents=True)
        base = 1_693_274_724_086_398_208
        timestamps = [base + i * 250_000_000 for i in range(3)]
        (scene / "image0/color_timestamp.txt").write_text(
            "#timestamp[ns], filename\n" + "".join(f"{t},{t}.png\n" for t in timestamps))

        def row(timestamp, x, y):
            # GT has position XYZ, quaternion WXYZ, velocity and two biases.
            return ",".join(map(str, [timestamp, x, y, 0, 1, 0, 0, 0] + [0] * 9)) + "\n"

        first = row(timestamps[0] + 100, 0, 0)
        duplicate = row(timestamps[0] + 100, 99 if conflict else 0, 0)
        (scene / "gt.txt").write_text("#timestamp, pose\n" + first + duplicate
                                     + row(timestamps[1] + 100, 1, .1)
                                     + row(timestamps[2] + 100, 2, .2))
        return timestamps

    def test_scene_deduplicates_identical_gt_and_retains_exact_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_timestamps = self._write_scene(root)
            _, timestamps, position, yaw, _, _, _, spacing, audit = _read_lusnar_scene(root, "Moon_1", fps=4)
        np.testing.assert_array_equal(timestamps, expected_timestamps)
        np.testing.assert_allclose(position, [[0, 0], [1, -.1], [2, -.2]])
        np.testing.assert_allclose(yaw, 0)
        self.assertEqual(audit["duplicate_gt_rows_removed"], 1)
        self.assertEqual(audit["maximum_image_gt_alignment_ns"], 100)
        self.assertAlmostEqual(spacing, np.sqrt(1.01))

    def test_conflicting_same_timestamp_poses_fail_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_scene(root, conflict=True)
            with self.assertRaisesRegex(ValueError, "conflicting duplicate pose"):
                _read_lusnar_scene(root, "Moon_1", fps=4)


if __name__ == "__main__":
    unittest.main()
