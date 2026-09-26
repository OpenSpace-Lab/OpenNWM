"""Read precomputed NavAnywhere manifests without rescanning source/cache trees."""
from __future__ import annotations
import json
import os
from typing import Any
from navanywhere_recipe import load_sampling_recipe


def validate_navanywhere_latent_cache(config: Any, *, recipe_path: str, launch_dir: str) -> dict:
    root = os.path.expanduser(str(config.dataset.precomputed_latents.root))
    if not os.path.isabs(root):
        root = os.path.join(launch_dir, root)
    with open(os.path.join(root, "metadata.json"), encoding="utf-8") as stream:
        metadata = json.load(stream)
    recipe, resolved_recipe, recipe_sha = load_sampling_recipe(
        recipe_path, seed=int(config.seed), context_size=int(config.dataset.context_size),
        goals_per_obs=int(config.dataset.datasets.navanywhere.get("goals_per_obs", 1)),
        frame_offset_range=(-64, 64))
    records = {}
    for source in sorted({str(row["source_id"]) for row in recipe["trajectories"]}):
        with open(os.path.join(root, source, "manifest.jsonl"), encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        records[source] = {str(row["trajectory_id"]): row for row in rows}
    return {"root": root, "metadata": metadata, "manifest_records": records,
            "expected_file_metadata": {"storage_dtype": "bfloat16"}, "recipe": recipe,
            "recipe_path": resolved_recipe, "recipe_sha256": recipe_sha}
