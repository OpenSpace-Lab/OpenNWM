"""Download the pinned DINOv3 ViT-L/16 teacher (resumes a partial download)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

MODELSCOPE_MODEL = "LittleBoss/LAM-weights"
MODELSCOPE_URL = "https://modelscope.cn/models/LittleBoss/LAM-weights/resolve/master/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DEFAULT_OUTPUT = Path(os.environ.get("DINO_WEIGHTS_PATH", "weights/dinov3.pth"))
DEFAULT_DINO_REPO = Path(os.environ.get("DINO_REPO_PATH", "third_party/dinov3"))
EXPECTED_DINO_REVISION = "346f38fee679c56a6888f91c51670fae61d364e0"
# Cache-format identifiers inherited from the original implementation.
EXPECTED_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
EXPECTED_SIZE = 1_213_050_671


def validate_source(repo, expected_revision=EXPECTED_DINO_REVISION):
    if not (Path(repo) / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 checkout missing: {repo}; see lam/README.md")


def validate_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"DINOv3 weights missing: {path}; run prepare-dino")


def load_strict_backbone(repo, weights, device):
    import torch
    model = torch.hub.load(str(repo), "dinov3_vitl16", source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False).to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=MODELSCOPE_URL)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"Already available: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                    "--output", str(partial), args.url], check=True)
    partial.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
