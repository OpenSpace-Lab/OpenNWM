import torch
from scripts.rollout_actions import body_frame_rollout_actions


def test_turning_motion_is_relative_to_current_heading():
    delta = torch.tensor([[[1., 0., torch.pi/2], [0., 1., 0.]]])
    actual = body_frame_rollout_actions(delta)
    torch.testing.assert_close(actual[0, :, :2], torch.tensor([[1., 0.], [1., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(body_frame_rollout_actions(delta, 2)[0, 0], torch.tensor([1., 1., torch.pi/2]))


def test_legacy_asymmetric_normalization_has_no_accumulated_offset():
    stats = {'min': [-2., -3.], 'max': [6., 3.]}
    # Two successive forward steps; absolute normalized x is -0.25, 0.
    delta = torch.tensor([[[-.25, 0., 0.], [.25, 0., 0.]]])
    result = body_frame_rollout_actions(delta, legacy_stats=stats)
    torch.testing.assert_close(result[..., 0], torch.tensor([[-.25, -.25]]))


def test_camera_optical_axis_defines_forward():
    import numpy as np
    from scripts.prepare_tum_camera_heading import camera_forward_yaw
    # Rotate optical +Z onto world +X. Image-right +X then points down.
    q=np.array([[0.,np.sin(np.pi/4),0.,np.cos(np.pi/4)]])
    np.testing.assert_allclose(camera_forward_yaw(q),[0.],atol=1e-12)
