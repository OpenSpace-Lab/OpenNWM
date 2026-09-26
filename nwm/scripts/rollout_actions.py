"""Convert origin-frame pose differences to current-body-frame rollout actions."""
import torch


def body_frame_rollout_actions(delta, stride=1, legacy_stats=None):
    poses = delta.cumsum(dim=1)
    if legacy_stats is not None:
        lo = torch.as_tensor(legacy_stats['min'], device=delta.device, dtype=delta.dtype)
        hi = torch.as_tensor(legacy_stats['max'], device=delta.device, dtype=delta.dtype)
        poses = poses.clone()
        poses[..., :2] = (poses[..., :2] + 1) * (hi - lo) / 2 + lo
    poses = torch.cat((torch.zeros_like(poses[:, :1]), poses), dim=1)
    start, end = poses[:, :-1:stride], poses[:, stride::stride]
    movement = end - start
    c, s = start[..., 2].cos(), start[..., 2].sin()
    x, y = movement[..., 0].clone(), movement[..., 1].clone()
    movement[..., 0] = c * x + s * y
    movement[..., 1] = -s * x + c * y
    movement[..., 2] = torch.atan2(movement[..., 2].sin(), movement[..., 2].cos())
    if legacy_stats is not None:
        movement[..., :2] = 2 * (movement[..., :2] - lo) / (hi - lo) - 1
    return movement
