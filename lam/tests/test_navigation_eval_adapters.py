from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("lightning")
pytest.importorskip("piq")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lam.model import LAM
from lam.navigation_evaluation.adapters import (
    inspect_evaluation_checkpoint,
    load_evaluation_adapter,
    registered_checkpoint_classes,
)
from lam.navigation_evaluation.metrics import (
    action_reconstruction_metrics,
    feature_reconstruction_metrics,
    positive_degradation,
    reconstruction_controls,
    rgb_reconstruction_metrics,
    wrapped_angle_difference,
)
from lam.navigation_variants import (
    ActionOnlyLAM,
    DINOFeatureLAM,
    PixelActionLAM,
    PixelLAM,
)


def _small_kwargs() -> dict:
    return {
        "image_channels": 3,
        "lam_model_dim": 8,
        "lam_latent_dim": 4,
        "lam_patch_size": 2,
        "lam_enc_blocks": 1,
        "lam_dec_blocks": 1,
        "lam_num_heads": 2,
        "lam_dropout": 0.0,
        "beta": 1e-6,
        "log_interval": 100,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    class_path: str,
    *,
    mutate_state=None,
) -> None:
    hparams = {
        key: value
        for key, value in dict(model.hparams).items()
        if key not in {"optimizer", "_instantiator", "_class_path"}
    }
    hparams["_class_path"] = class_path
    state = model.state_dict()
    if mutate_state is not None:
        state = mutate_state(state)
    checkpoint = {
        "epoch": 3,
        "global_step": 17,
        "pytorch-lightning_version": "test",
        "state_dict": state,
        "hyper_parameters": hparams,
        "datamodule_hyper_parameters": {
            "action_min": [-2.5, -4.0],
            "action_max": [5.0, 4.0],
        },
        # A tensor here verifies weights-only+mmap loading tolerates optimizer
        # payloads without the adapter ever constructing or restoring one.
        "optimizer_states": [{"state": {0: {"momentum": torch.ones(2)}}}],
    }
    hook = getattr(model, "on_save_checkpoint", None)
    if hook is not None and isinstance(model, DINOFeatureLAM):
        hook(checkpoint)
    torch.save(checkpoint, path)


@pytest.mark.parametrize(
    ("model", "class_path", "capabilities"),
    [
        (LAM(**_small_kwargs()), "lam.model.LAM", {"rgb"}),
        (
            PixelLAM(**_small_kwargs()),
            "lam.navigation_variants.PixelLAM",
            {"rgb"},
        ),
        (
            ActionOnlyLAM(**_small_kwargs()),
            "lam.navigation_variants.ActionOnlyLAM",
            {"action"},
        ),
        (
            PixelActionLAM(**_small_kwargs(), action_loss_weight=2.0),
            "lam.navigation_variants.PixelActionLAM",
            {"rgb", "action"},
        ),
    ],
)
def test_adapter_native_decode_equivalence_and_capabilities(
    tmp_path, model, class_path, capabilities
):
    path = tmp_path / "model.ckpt"
    _save_checkpoint(path, model, class_path)
    adapter = load_evaluation_adapter(path, "cpu", fused_attention=False)
    assert adapter.capabilities == capabilities
    assert adapter.checkpoint_metadata["class_path"] == class_path
    assert adapter.checkpoint_metadata["global_step"] == 17
    assert adapter.checkpoint_metadata["datamodule_hparams"]["action_min"] == [
        -2.5,
        -4.0,
    ]
    assert not adapter.model.training
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())

    videos = torch.rand(2, 2, 4, 4, 3)
    latent = adapter.encode(videos)
    assert latent.shape == (2, 1, 1, 4)
    assert adapter.model.lam.mu_record is None
    native = adapter.native(videos)
    assert adapter.model.lam.mu_record is None

    if "rgb" in capabilities:
        decoded = adapter.decode_rgb(videos[:, 0], latent)
        assert decoded.shape == (2, 4, 4, 3)
        torch.testing.assert_close(decoded, native["recon"][:, 0])
    else:
        with pytest.raises(NotImplementedError, match="RGB|rgb"):
            adapter.decode_rgb(videos[:, 0], latent)

    if "action" in capabilities:
        decoded_action = adapter.decode_action(latent)
        assert decoded_action.shape == (2, 1, 3)
        torch.testing.assert_close(decoded_action, native["action_recon"])
    else:
        with pytest.raises(NotImplementedError, match="action"):
            adapter.decode_action(latent)


def test_adapter_cross_latent_really_uses_donor(tmp_path):
    torch.manual_seed(4)
    model = LAM(**_small_kwargs())
    path = tmp_path / "pixel.ckpt"
    _save_checkpoint(path, model, "lam.model.LAM")
    adapter = load_evaluation_adapter(path, fused_attention=False)
    videos = torch.rand(2, 2, 4, 4, 3)
    latent = adapter.encode(videos)
    swapped = adapter.decode_rgb(videos[:, 0], latent.flip(0))
    expected = torch.cat(
        [
            adapter.decode_rgb(videos[0:1, 0], latent[1:2]),
            adapter.decode_rgb(videos[1:2, 0], latent[0:1]),
        ]
    )
    torch.testing.assert_close(swapped, expected)


def test_checkpoint_whitelist_metadata_and_strict_state(tmp_path):
    assert set(registered_checkpoint_classes()) == {
        "lam.model.LAM",
        "lam.navigation_variants.ActionOnlyLAM",
        "lam.navigation_variants.DINOFeatureLAM",
        "lam.navigation_variants.PixelActionLAM",
        "lam.navigation_variants.PixelLAM",
    }
    model = LAM(**_small_kwargs())
    valid = tmp_path / "valid.ckpt"
    _save_checkpoint(valid, model, "lam.model.LAM")
    metadata = inspect_evaluation_checkpoint(valid)
    assert metadata["class_path"] == "lam.model.LAM"

    unknown = tmp_path / "unknown.ckpt"
    _save_checkpoint(unknown, model, "arbitrary.module.UnsafeClass")
    with pytest.raises(ValueError, match="unsupported checkpoint class"):
        inspect_evaluation_checkpoint(unknown)

    def remove_first(state):
        state = state.copy()
        state.pop(next(iter(state)))
        return state

    broken = tmp_path / "broken.ckpt"
    _save_checkpoint(broken, model, "lam.model.LAM", mutate_state=remove_first)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_evaluation_adapter(broken, fused_attention=False)


def test_proc_fd_checkpoint_survives_atomic_source_replacement(tmp_path):
    torch.manual_seed(101)
    original_model = LAM(**_small_kwargs())
    source = tmp_path / "last.ckpt"
    _save_checkpoint(source, original_model, "lam.model.LAM")
    original_state = {
        key: value.detach().clone()
        for key, value in original_model.state_dict().items()
    }

    file_descriptor = os.open(source, os.O_RDONLY)
    access_path = Path(f"/proc/self/fd/{file_descriptor}")
    try:
        torch.manual_seed(202)
        replacement_model = LAM(**_small_kwargs())
        replacement = tmp_path / "replacement.ckpt"
        _save_checkpoint(replacement, replacement_model, "lam.model.LAM")
        os.replace(replacement, source)

        metadata = inspect_evaluation_checkpoint(access_path)
        adapter = load_evaluation_adapter(access_path, fused_attention=False)
        assert metadata["class_path"] == "lam.model.LAM"
        for key, expected in original_state.items():
            torch.testing.assert_close(adapter.model.state_dict()[key], expected)

        # CPU restoration must be detached from the mmap.  Modifying the held
        # old inode after load cannot mutate model parameters.
        first_name, first_parameter = next(iter(adapter.model.state_dict().items()))
        before = first_parameter.detach().clone()
        with access_path.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"BROKEN!!")
            handle.flush()
            os.fsync(handle.fileno())
        torch.testing.assert_close(adapter.model.state_dict()[first_name], before)
    finally:
        os.close(file_descriptor)


class _FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_scale = nn.Parameter(torch.tensor(1.0))

    def forward_features(self, images):
        summary = images.mean(dim=(1, 2, 3))
        tokens = summary[:, None, None].expand(-1, 300, 1024)
        return {"x_norm_patchtokens": tokens * self.teacher_scale}


@pytest.fixture
def fake_dino_files(tmp_path, monkeypatch):
    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# local test repository\n")
    weights = tmp_path / "teacher.pth"
    weights.write_bytes(b"frozen teacher")
    monkeypatch.setattr(
        DINOFeatureLAM,
        "_load_dino",
        staticmethod(lambda *_: _FakeDINO()),
    )
    return repo, weights


def _small_dino(repo: Path, weights: Path) -> DINOFeatureLAM:
    return DINOFeatureLAM(
        image_channels=3,
        lam_model_dim=8,
        lam_latent_dim=4,
        lam_patch_size=16,
        lam_enc_blocks=1,
        lam_dec_blocks=1,
        lam_num_heads=2,
        lam_dropout=0.0,
        dino_repo_path=str(repo),
        dino_weights_path=str(weights),
    )


def test_dino_adapter_feature_decode_and_metadata_validation(tmp_path, fake_dino_files):
    repo, weights = fake_dino_files
    model = _small_dino(repo, weights)
    path = tmp_path / "dino.ckpt"
    _save_checkpoint(path, model, "lam.navigation_variants.DINOFeatureLAM")
    adapter = load_evaluation_adapter(path, fused_attention=False)
    assert adapter.capabilities == {"feature"}
    assert (
        adapter.checkpoint_metadata["dino"]["weights_sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )

    videos = torch.rand(1, 2, 240, 320, 3)
    latent = adapter.encode(videos)
    native = adapter.native(videos)
    decoded = adapter.decode_feature(videos[:, 0], latent)
    target = adapter.extract_features(videos[:, 1])
    assert decoded.shape == target.shape == (1, 300, 1024)
    torch.testing.assert_close(decoded, native["feature_recon"][:, 0])
    torch.testing.assert_close(target, native["target_features"][:, 0])

    weights.write_bytes(b"changed teacher")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_evaluation_adapter(path, fused_attention=False)


class _FakeLPIPS(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_rgb_metrics_identity_boundaries_temporal_and_offline_failure():
    target = torch.rand(2, 2, 8, 8, 3)
    metrics = rgb_reconstruction_metrics(
        target.clone(), target, lpips_model=_FakeLPIPS()
    )
    torch.testing.assert_close(metrics["mse"], torch.zeros(2))
    torch.testing.assert_close(metrics["mae"], torch.zeros(2))
    assert torch.isinf(metrics["psnr"]).all()
    torch.testing.assert_close(metrics["ssim"], torch.ones(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(metrics["lpips"], torch.zeros(2))

    zeros = torch.zeros(1, 8, 8, 3)
    ones = torch.ones_like(zeros)
    opposite = rgb_reconstruction_metrics(zeros, ones, lpips_model=_FakeLPIPS())
    torch.testing.assert_close(opposite["mse"], torch.ones(1))
    torch.testing.assert_close(opposite["mae"], torch.ones(1))
    torch.testing.assert_close(opposite["psnr"], torch.zeros(1))
    torch.testing.assert_close(opposite["lpips"], torch.ones(1))

    with pytest.raises(RuntimeError, match="prepare-assets"):
        rgb_reconstruction_metrics(zeros, ones, lpips_model=None)
    diagnostic = rgb_reconstruction_metrics(
        zeros, ones, lpips_model=None, require_lpips=False
    )
    assert torch.isnan(diagnostic["lpips"]).all()

    minimal = rgb_reconstruction_metrics(
        zeros,
        ones,
        lpips_model=_FakeLPIPS(),
        metric_names=("psnr", "ssim", "lpips"),
    )
    assert set(minimal) == {"psnr", "ssim", "lpips"}


def test_feature_and_action_metrics_use_raw_tokens_physical_units_and_yaw_wrap():
    target_features = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    feature = feature_reconstruction_metrics(target_features, target_features)
    torch.testing.assert_close(feature["feature_mse"], torch.zeros(1))
    torch.testing.assert_close(feature["feature_mae"], torch.zeros(1))
    torch.testing.assert_close(feature["feature_cosine_similarity"], torch.ones(1))

    prediction = torch.tensor([[[1.0, 0.0, math.radians(179.0)]]], dtype=torch.float32)
    target = torch.tensor([[[-1.0, 0.0, math.radians(-179.0)]]], dtype=torch.float32)
    action = action_reconstruction_metrics(prediction, target, waypoint_spacing=0.25)
    torch.testing.assert_close(action["translation_epe_waypoints"], torch.tensor([7.5]))
    torch.testing.assert_close(action["translation_epe_meters"], torch.tensor([1.875]))
    torch.testing.assert_close(
        action["yaw_mae_degrees"], torch.tensor([2.0]), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        action["yaw_rmse_degrees"], torch.tensor([2.0]), atol=1e-4, rtol=0
    )
    circular = wrapped_angle_difference(prediction[..., 2], target[..., 2])
    torch.testing.assert_close(
        circular.abs(), torch.tensor([[math.radians(2.0)]]), atol=1e-6, rtol=0
    )


def test_controls_and_positive_degradation_direction():
    target = torch.zeros(2, 1)
    anchor = torch.full_like(target, 3.0)
    oracle = torch.full_like(target, 1.0)
    swapped = torch.full_like(target, 2.0)

    def metrics(prediction, ground_truth):
        error = (prediction - ground_truth).abs().squeeze(1)
        return {"mae": error, "ssim": 1.0 - error / 4.0}

    controls = reconstruction_controls(
        metrics,
        target=target,
        anchor=anchor,
        oracle=oracle,
        swapped=swapped,
    )
    torch.testing.assert_close(controls["anchor_copy"]["mae"], torch.full((2,), 3.0))
    torch.testing.assert_close(controls["oracle"]["mae"], torch.ones(2))
    torch.testing.assert_close(controls["swapped"]["mae"], torch.full((2,), 2.0))
    torch.testing.assert_close(controls["degradation"]["mae"], torch.ones(2))
    torch.testing.assert_close(controls["degradation"]["ssim"], torch.full((2,), 0.25))

    explicit = positive_degradation(
        {"score": torch.tensor([0.6])},
        {"score": torch.tensor([0.9])},
        higher_is_better={"score"},
    )
    torch.testing.assert_close(explicit["score"], torch.tensor([0.3]))
