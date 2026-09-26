#!/usr/bin/env python3
"""Write tiny independent RGB trajectories and LAM CPU smoke configurations.

This is an optional developer fixture, not a training prerequisite. All generated
data and configs go under --output; no original dataset is read or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import yaml


def make_fixture(output: Path, frames: int = 20) -> dict:
    if frames < 18:
        raise ValueError("At least 18 frames are needed for the +/-8 evaluation grid")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; reuse an existing fixture as-is")
    output.mkdir(parents=True, exist_ok=True)
    width, height = 16, 12
    for dataset_index, dataset in enumerate(("recon", "tartan_drive")):
        for split_index, split in enumerate(("train", "val", "test")):
            names = [f"{split}_{index}" for index in range(2)]
            manifest = output / "splits" / dataset / split / "traj_names.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(names) + "\n")
            for trajectory_index, name in enumerate(names):
                directory = output / "data" / dataset / name
                directory.mkdir(parents=True, exist_ok=True)
                positions = np.stack((np.arange(frames) * 0.25, np.zeros(frames)), axis=1)
                with (directory / "traj_data.pkl").open("wb") as handle:
                    pickle.dump({"position": positions, "yaw": np.zeros(frames)}, handle)
                for frame in range(frames):
                    yy, xx = np.mgrid[:height, :width]
                    base = 17 * dataset_index + 31 * split_index + 47 * trajectory_index + frame * 3
                    rgb = np.stack(((xx * 7 + base) % 256, (yy * 9 + base) % 256,
                                    ((xx + yy) * 5 + base) % 256), axis=-1).astype(np.uint8)
                    Image.fromarray(rgb).save(directory / f"{frame}.jpg", quality=95)
    source = {"data_root": str(output / "data/recon"),
              "split_manifest": str(output / "splits/recon/train/traj_names.txt"),
              "waypoint_spacing": 0.25, "display_name": "RECON"}
    train = {
        "model": {"init_args": {"lam_model_dim": 8, "lam_latent_dim": 4,
                                "lam_patch_size": 2, "lam_enc_blocks": 1,
                                "lam_dec_blocks": 1, "lam_num_heads": 2,
                                "lam_dropout": 0.0, "log_interval": 0}},
        "data": {"source_specs": [source], "batch_size": 2, "num_workers": 0,
                 "samples_per_epoch": 8, "max_frame_offset": 2, "color_aug": False,
                 "image_height": height, "image_width": width, "print_audit": False,
                 "pin_memory": False, "persistent_workers": False,
                 "index_cache_path": str(output / "train-index.pkl")},
        "trainer": {"precision": "32-true", "log_every_n_steps": 1,
                    "enable_progress_bar": False, "enable_model_summary": False},
    }
    (output / "lam_train.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
    template = Path(__file__).resolve().parents[1] / "lam/config/navigation_eval.yaml"
    evaluation = yaml.safe_load(template.read_text())
    evaluation["datasets"] = {
        dataset: {"display_name": dataset, "path_alias": dataset,
                  "data_root": str(output / "data" / dataset), "waypoint_spacing": 0.25,
                  "splits": {split: str(output / "splits" / dataset / split / "traj_names.txt")
                             for split in ("train", "test")}}
        for dataset in ("recon", "tartan_drive")
    }
    evaluation.update(domains={"train": ["recon"], "ood": ["tartan_drive"]}, protocols=["id", "ood"])
    evaluation["sampling"].update(metric_count=2, visual_count=0, image_height=height, image_width=width)
    evaluation["metrics"]["bootstrap_samples"] = 20
    evaluation["metrics"]["lpips"].update(enabled=False, asset_root=str(output / "assets"))
    evaluation["runtime"].update(device="cpu", precision="32", batch_size=2, num_workers=0)
    evaluation["output"].update(root=str(output / "evaluation"), save_parquet=False)
    (output / "lam_eval.yaml").write_text(yaml.safe_dump(evaluation, sort_keys=False))
    summary = {"output": str(output), "trajectories": 12, "images": 12 * frames,
               "image_size": [height, width], "train_config": str(output / "lam_train.yaml"),
               "eval_config": str(output / "lam_eval.yaml"),
               "note": "Synthetic developer fixture; not a benchmark or convergence result."}
    (output / "fixture.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(make_fixture(args.output, args.frames), indent=2))


if __name__ == "__main__":
    main()
