from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import calibrate_action_weight as calibration


class _CalibrationDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {
            "videos": torch.full((2, 2, 2, 3), 2.0),
            "actions": torch.full((1, 3), 4.0),
        }


class _CalibrationData:
    def setup(self, stage: str) -> None:
        assert stage == "fit"

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_CalibrationDataset(), batch_size=2, shuffle=False)


class _CalibrationModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "recon": torch.zeros_like(batch["videos"][:, 1:]),
            "action_recon": torch.zeros_like(batch["actions"]),
        }


def _arguments(base: Path, variant: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        base_config=base,
        variant_config=variant,
        output_config=output,
        num_batches=2,
        batch_size=None,
        num_workers=None,
        device="cpu",
        data_root=[],
        split_manifest=[],
    )


def test_calibration_is_repeatable_and_generated_overlay_loads(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.yaml"
    variant = tmp_path / "variant.yaml"
    base.write_text("trainer:\n  precision: 32-true\nmodel: {}\ndata: {}\n")
    variant.write_text("model:\n  class_path: lam.navigation_variants.PixelActionLAM\n")
    outputs = [tmp_path / "first.yaml", tmp_path / "second.yaml"]
    arguments = iter(_arguments(base, variant, output) for output in outputs)
    monkeypatch.setattr(calibration, "parse_args", lambda: next(arguments))
    monkeypatch.setattr(
        calibration,
        "_build_components",
        lambda *args, **kwargs: (_CalibrationModel(), _CalibrationData()),
    )

    calibration.main()
    calibration.main()

    first = yaml.safe_load(outputs[0].read_text())
    second = yaml.safe_load(outputs[1].read_text())
    assert first == second == {"model": {"init_args": {"action_loss_weight": 0.25}}}
    text = outputs[0].read_text()
    assert "# pixel_mse: 4" in text
    assert "# action_mse: 16" in text

    command = [
        sys.executable,
        str(PROJECT_DIR / "navigation_main.py"),
        "fit",
        "--config",
        str(PROJECT_DIR / "config/navigation_base.yaml"),
        "--config",
        str(PROJECT_DIR / "config/navigation_4_pixel_action.yaml"),
        "--config",
        str(outputs[0]),
        "--print_config",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    resolved = yaml.safe_load(completed.stdout)
    assert resolved["model"]["init_args"]["action_loss_weight"] == 0.25
