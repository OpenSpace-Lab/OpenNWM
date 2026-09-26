#!/usr/bin/env python3
"""Freeze a deterministic, action-agnostic NavAnywhere sampling recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navanywhere_recipe import (  # noqa: E402
    build_sampling_recipe,
    load_sampling_recipe,
    write_sampling_recipe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the compact JSON recipe shared by TimePT, GeoPT, IDMPT, "
            "and LatentPT."
        )
    )
    parser.add_argument("--root", required=True, help="NavAnywhere source root")
    parser.add_argument("--output", required=True, help="Output recipe JSON")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--goals-per-obs", type=int, default=4)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="0 uses the total usable observation count.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Optional source directory; repeat to select several.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate and print an existing output instead of rewriting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser()
    if output.exists() and args.validate_existing and not args.overwrite:
        recipe, resolved, digest = load_sampling_recipe(
            output,
            seed=args.seed,
            context_size=args.context_size,
            goals_per_obs=args.goals_per_obs,
            frame_offset_range=(-64, 64),
            samples_per_epoch=(
                args.samples_per_epoch if args.samples_per_epoch > 0 else None
            ),
        )
        if args.source_ids is not None and set(recipe["sources"]) != set(
            args.source_ids
        ):
            raise ValueError(
                "Existing recipe sources do not match --source-id values: "
                f"recipe={sorted(recipe['sources'])}, "
                f"requested={sorted(set(args.source_ids))}"
            )
    else:
        recipe = build_sampling_recipe(
            args.root,
            seed=args.seed,
            context_size=args.context_size,
            goals_per_obs=args.goals_per_obs,
            min_frame_offset=-64,
            max_frame_offset=64,
            samples_per_epoch=args.samples_per_epoch,
            source_ids=args.source_ids,
        )
        resolved = write_sampling_recipe(
            output, recipe, overwrite=bool(args.overwrite)
        )
        recipe, resolved, digest = load_sampling_recipe(resolved)
    print(
        json.dumps(
            {
                "recipe": resolved,
                "sha256": digest,
                "seed": recipe["seed"],
                "samples_per_epoch": recipe["samples_per_epoch"],
                "totals": recipe["totals"],
                "sources": recipe["sources"],
                "observation_sampling": recipe["observation_sampling"]["strategy"],
                "goal_sampling": recipe["goal_sampling"]["strategy"],
                "goal_strata": recipe["goal_sampling"]["strata"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
