from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning import __version__ as lightning_version
from torch import nn

pytest.importorskip("lightning")
pytest.importorskip("piq")

LAM_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAM_PROJECT))

from lam.model import LAM
from lam.modules.blocks import PositionalEncoding
from lam.navigation_variants import (
    ActionOnlyLAM,
    DINOFeatureLAM,
    PixelActionLAM,
    PixelLAM,
    _log_sampling_statistics,
    _masked_action_mse,
)


def test_position_encoding_is_device_dtype_safe_and_nonpersistent():
    encoding = PositionalEncoding(8)
    value = torch.zeros(1, 2, 4, 8, dtype=torch.float64)
    output = encoding(value)
    assert output.device == value.device
    assert output.dtype == value.dtype
    assert "pos_enc" not in encoding.state_dict()


def _small_kwargs():
    return {
        "image_channels": 3,
        "lam_model_dim": 8,
        "lam_latent_dim": 32,
        "lam_patch_size": 2,
        "lam_enc_blocks": 1,
        "lam_dec_blocks": 1,
        "lam_num_heads": 2,
        "lam_dropout": 0.0,
        "beta": 0.25,
        "log_interval": 100,
    }


def _pixel_batch(batch_size=1):
    return {
        "videos": torch.rand(batch_size, 2, 4, 4, 3),
        "actions": torch.rand(batch_size, 1, 3),
    }


@pytest.mark.parametrize("batch_size", [1, 2])
def test_action_only_shapes_loss_and_gradients(batch_size):
    torch.manual_seed(9)
    model = ActionOnlyLAM(**_small_kwargs())
    batch = _pixel_batch(batch_size)

    outputs, loss, aux = model.shared_step(batch)
    aux = dict(aux)
    assert outputs["z_rep"].shape == (batch_size, 1, 1, 32)
    assert outputs["action_recon"].shape == (batch_size, 1, 3)
    assert "recon" not in outputs
    assert not hasattr(model.lam, "patch_up")
    assert not hasattr(model.lam, "action_up")
    assert not hasattr(model.lam, "decoder")
    torch.testing.assert_close(loss, aux["action_mse"] + model.beta * aux["kl_loss"])

    loss.backward()
    assert model.action_decoder[-1].weight.grad is not None
    assert model.lam.encoder.out.weight.grad is not None


def test_action_only_requires_exact_action_sequence_shape():
    model = ActionOnlyLAM(**_small_kwargs())
    batch = _pixel_batch()
    batch["actions"] = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="actions must have shape"):
        model.shared_step(batch)


def test_masked_action_loss_ignores_unlabeled_values_and_gradients():
    prediction = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[100.0, 200.0, 300.0]]], requires_grad=True
    )
    target = torch.zeros_like(prediction)
    loss = _masked_action_mse(
        {"action_mask": torch.tensor([True, False])}, prediction, target
    )
    torch.testing.assert_close(loss, prediction[0].square().mean())
    loss.backward()
    assert torch.count_nonzero(prediction.grad[0]) == 3
    assert torch.count_nonzero(prediction.grad[1]) == 0


def test_action_loss_without_mask_preserves_legacy_full_batch_mean():
    prediction = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    target = torch.zeros_like(prediction)
    torch.testing.assert_close(
        _masked_action_mse({}, prediction, target), prediction.square().mean()
    )


def test_all_unlabeled_action_loss_is_connected_zero():
    prediction = torch.randn(2, 1, 3, requires_grad=True)
    loss = _masked_action_mse(
        {"action_mask": torch.zeros(2, dtype=torch.bool)},
        prediction,
        torch.zeros_like(prediction),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 0


def test_sampling_logging_expands_for_navanywhere15_without_changing_old_profiles():
    class LoggingStub:
        global_step = 0

        def __init__(self, source_count=None):
            self.logged = None
            self._trainer = (
                None
                if source_count is None
                else SimpleNamespace(
                    datamodule=SimpleNamespace(source_specs=({},) * source_count)
                )
            )

        def log_dict(self, metrics, **kwargs):
            del kwargs
            self.logged = metrics

    batch = {
        "dataset_id": torch.tensor([0, 17]),
        "frame_offset": torch.tensor([-8, 8]),
    }
    expanded = LoggingStub(source_count=18)
    _log_sampling_statistics(expanded, batch)
    assert "sampling/source_17_fraction" in expanded.logged

    original = LoggingStub()
    _log_sampling_statistics(original, batch)
    assert "sampling/source_15_fraction" in original.logged
    assert "sampling/source_16_fraction" not in original.logged


def test_pixel_lam_short_forward_backward_smoke():
    model = PixelLAM(**_small_kwargs())
    batch = _pixel_batch(batch_size=2)
    outputs = model.lam(batch)
    mse = (batch["videos"][:, 1:] - outputs["recon"]).square().mean()
    kl = -0.5 * torch.sum(
        1 + outputs["z_var"] - outputs["z_mu"].square() - outputs["z_var"].exp(),
        dim=1,
    ).mean()
    (mse + model.beta * kl).backward()
    assert model.lam.decoder.out.weight.grad is not None
    assert model.lam.encoder.out.weight.grad is not None


def test_pixel_action_preserves_original_pixel_path_and_loss():
    kwargs = _small_kwargs()
    torch.manual_seed(123)
    original = LAM(**kwargs)
    torch.manual_seed(123)
    variant = PixelActionLAM(**kwargs, action_loss_weight=2.5)

    assert original.lam.state_dict().keys() == variant.lam.state_dict().keys()
    for key, value in original.lam.state_dict().items():
        torch.testing.assert_close(value, variant.lam.state_dict()[key], rtol=0, atol=0)

    original.eval()
    variant.eval()
    batch = _pixel_batch(batch_size=1)
    original_outputs = original.lam(batch)
    variant_outputs, components = variant.loss_components(batch)
    torch.testing.assert_close(original_outputs["recon"], variant_outputs["recon"])
    torch.testing.assert_close(original_outputs["z_rep"], variant_outputs["z_rep"])
    expected = (
        components["mse_loss"]
        + 2.5 * components["action_mse"]
        + kwargs["beta"] * components["kl_loss"]
    )

    # eval() makes z deterministic, so loss_components and shared_step agree.
    _, loss, aux = variant.shared_step(batch)
    torch.testing.assert_close(loss, expected)
    assert set(dict(aux)) == {"mse_loss", "kl_loss", "action_mse"}


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(PixelLAM(**_small_kwargs()), id="pixel"),
        pytest.param(
            PixelActionLAM(**_small_kwargs(), action_loss_weight=1.0),
            id="pixel-action",
        ),
    ],
)
def test_pixel_quality_metrics_are_eval_only(model, monkeypatch):
    import lam.model as model_module

    metric_calls = []
    monkeypatch.setattr(
        model_module.piq,
        "psnr",
        lambda gt, recon: metric_calls.append("psnr") or torch.zeros(gt.shape[0]),
    )
    monkeypatch.setattr(
        model_module.piq,
        "ssim",
        lambda gt, recon: metric_calls.append("ssim") or torch.zeros(gt.shape[0]),
    )
    monkeypatch.setattr(model, "log_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "log_images", lambda *args, **kwargs: None)

    batch = _pixel_batch(batch_size=2)
    model.train()
    model.training_step(batch, 1)
    assert metric_calls == []

    model.eval()
    model.test_step(batch, 1)
    assert metric_calls == ["psnr", "ssim"]


def test_pixel_action_backward_reaches_both_paths():
    model = PixelActionLAM(**_small_kwargs(), action_loss_weight=1.0)
    _, losses = model.loss_components(_pixel_batch(batch_size=2))
    total = losses["mse_loss"] + losses["action_mse"] + model.beta * losses["kl_loss"]
    total.backward()
    assert model.lam.decoder.out.weight.grad is not None
    assert model.action_decoder[-1].weight.grad is not None
    assert model.lam.encoder.out.weight.grad is not None


def test_pixel_action_fails_before_training_without_calibration():
    with pytest.raises(ValueError, match="calibrated action_loss_weight"):
        PixelActionLAM(**_small_kwargs())


class _FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_scale = nn.Parameter(torch.tensor(1.0))
        self.last_input = None

    def forward_features(self, images):
        self.last_input = images.detach().clone()
        summary = images.mean(dim=(1, 2, 3), keepdim=False)
        tokens = summary[:, None, None].expand(-1, 300, 1024) * self.teacher_scale
        return {"x_norm_patchtokens": tokens}


@pytest.fixture
def fake_dino_files(tmp_path, monkeypatch):
    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# fake local torch hub repository\n")
    weights = tmp_path / "dinov3_vitl16.pth"
    weights.write_bytes(b"external frozen DINO weights")
    monkeypatch.setattr(
        DINOFeatureLAM,
        "_load_dino",
        staticmethod(lambda repo_path, weights_path, model_name: _FakeDINO()),
    )
    return repo, weights


def _small_dino(fake_dino_files):
    repo, weights = fake_dino_files
    return DINOFeatureLAM(
        image_channels=3,
        lam_model_dim=8,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=1,
        lam_dec_blocks=1,
        lam_num_heads=2,
        lam_dropout=0.0,
        beta=0.25,
        dino_repo_path=str(repo),
        dino_weights_path=str(weights),
    )


def _dino_batch(batch_size=1):
    return {"videos": torch.rand(batch_size, 2, 240, 320, 3)}


@pytest.mark.parametrize("batch_size", [1, 2])
def test_dino_feature_loss_shape_normalization_freezing_and_gradients(
    fake_dino_files, batch_size
):
    model = _small_dino(fake_dino_files)
    model.train()
    assert not model.dino_encoder.training
    assert all(
        not parameter.requires_grad for parameter in model.dino_encoder.parameters()
    )

    batch = _dino_batch(batch_size)
    outputs, loss, aux = model.shared_step(batch)
    aux = dict(aux)
    assert outputs["z_rep"].shape == (batch_size, 1, 1, 32)
    assert outputs["feature_recon"].shape == (batch_size, 1, 300, 1024)
    assert outputs["target_features"].shape == (batch_size, 1, 300, 1024)
    assert "recon" not in outputs
    torch.testing.assert_close(loss, aux["feature_mse"] + model.beta * aux["kl_loss"])
    assert aux["patch_cosine_similarity"].ndim == 0

    expected_first_pixel = (
        batch["videos"][0, 0, 0, 0].permute(0) - torch.tensor([0.485, 0.456, 0.406])
    ) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(
        model.dino_encoder.last_input[0, :, 0, 0], expected_first_pixel
    )

    loss.backward()
    assert model.dino_encoder.teacher_scale.grad is None
    assert model.lam.decoder.out.weight.grad is not None
    assert model.lam.encoder.out.weight.grad is not None
    optimizer = model.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.dino_encoder.teacher_scale) not in optimized


def test_dino_precomputed_features_bypass_lazy_teacher(fake_dino_files):
    repo, weights = fake_dino_files
    model = DINOFeatureLAM(
        image_channels=3,
        lam_model_dim=8,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=1,
        lam_dec_blocks=1,
        lam_num_heads=2,
        dino_repo_path=str(repo),
        dino_weights_path=str(weights),
        dino_weights_sha256="2" * 64,
        use_precomputed_dino_features=True,
    )
    assert model.dino_encoder is None
    features = torch.randn(2, 2, 300, 1024, dtype=torch.float16)
    outputs = model({"dino_features": features})
    assert model.dino_encoder is None
    assert outputs["target_features"].shape == (2, 1, 300, 1024)
    torch.testing.assert_close(outputs["target_features"], features[:, 1:].float())


def test_dino_precomputed_features_do_not_require_unused_external_source(tmp_path):
    model = DINOFeatureLAM(
        image_channels=3,
        lam_model_dim=8,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=1,
        lam_dec_blocks=1,
        lam_num_heads=2,
        dino_repo_path=str(tmp_path / "unmounted-dinov3"),
        dino_weights_path=str(tmp_path / "unmounted-weights.pth"),
        dino_repo_revision="1" * 40,
        dino_weights_sha256="2" * 64,
        use_precomputed_dino_features=True,
    )

    features = torch.randn(1, 2, 300, 1024, dtype=torch.float16)
    outputs = model({"dino_features": features})
    assert outputs["target_features"].shape == (1, 1, 300, 1024)

    with pytest.raises(FileNotFoundError, match="local repository"):
        model(_dino_batch())


def test_dino_checkpoint_excludes_teacher_and_strictly_restores_trainable_state(
    fake_dino_files, tmp_path
):
    torch.manual_seed(1)
    source = _small_dino(fake_dino_files)
    state = source.state_dict()
    assert state
    assert not any(key.startswith("dino_encoder.") for key in state)

    checkpoint = {}
    source.on_save_checkpoint(checkpoint)
    _, weights = fake_dino_files
    assert checkpoint["dino"] == {
        "model_name": "dinov3_vitl16",
        "repo_path": str(fake_dino_files[0].resolve()),
        "repo_revision": "",
        "weights_path": str(weights.resolve()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }

    torch.manual_seed(7)
    restored = _small_dino(fake_dino_files)
    teacher_before = restored.dino_encoder.teacher_scale.detach().clone()
    result = restored.load_state_dict(state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    torch.testing.assert_close(restored.dino_encoder.teacher_scale, teacher_before)
    for key, value in source.lam.state_dict().items():
        torch.testing.assert_close(
            value, restored.lam.state_dict()[key], rtol=0, atol=0
        )

    broken_state = state.copy()
    broken_state.pop(next(key for key in broken_state if key.startswith("lam.")))
    with pytest.raises(RuntimeError, match="Missing key"):
        _small_dino(fake_dino_files).load_state_dict(broken_state, strict=True)
    restored.on_load_checkpoint(checkpoint)

    lightning_checkpoint = {
        **checkpoint,
        "state_dict": state,
        "hyper_parameters": dict(source.hparams),
        "pytorch-lightning_version": lightning_version,
    }
    checkpoint_path = tmp_path / "dino.ckpt"
    torch.save(lightning_checkpoint, checkpoint_path)
    loaded = DINOFeatureLAM.load_from_checkpoint(
        checkpoint_path, strict=True, weights_only=False
    )
    assert not any(key.startswith("dino_encoder.") for key in loaded.state_dict())
    for key, value in source.lam.state_dict().items():
        torch.testing.assert_close(value, loaded.lam.state_dict()[key], rtol=0, atol=0)


def test_dino_checkpoint_restore_does_not_rehash_external_weights(fake_dino_files):
    model = _small_dino(fake_dino_files)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    _, weights = fake_dino_files
    weights.write_bytes(b"different external frozen DINO weights")
    model.on_load_checkpoint(checkpoint)


def test_dino_online_teacher_skips_source_and_weight_audits(fake_dino_files):
    repo, weights = fake_dino_files
    model = DINOFeatureLAM(
        dino_repo_path=str(repo),
        dino_weights_path=str(weights),
        dino_repo_revision="2" * 40,
        dino_weights_sha256="0" * 64,
    )
    assert model.dino_encoder is not None


def test_dino_fails_before_torch_hub_when_local_inputs_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="local repository"):
        DINOFeatureLAM(
            dino_repo_path=str(tmp_path / "missing-repo"),
            dino_weights_path=str(tmp_path / "missing.pth"),
        )
