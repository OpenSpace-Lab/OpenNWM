"""Reproducible evaluation utilities for navigation latent-action models.

The package deliberately stays independent from Lightning's train-only data
module.  Evaluation is split into two phases: a CPU-only, checkpoint-agnostic
manifest build and one or more checkpoint runs consuming that immutable
manifest.
"""

from .config import config_digest, load_evaluation_config

__all__ = ["config_digest", "load_evaluation_config"]
