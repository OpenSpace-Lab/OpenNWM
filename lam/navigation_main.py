"""Dedicated LightningCLI entry point for navigation LAM experiments."""

from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI

from lam.navigation_dataset import NavigationPairDataModule


def main() -> None:
    """Run the navigation LAM CLI without changing the original LAM entry point."""

    LightningCLI(
        LightningModule,
        NavigationPairDataModule,
        subclass_mode_model=True,
        seed_everything_default=32,
        # Resuming into the same isolated experiment directory should refresh
        # the resolved config snapshot instead of failing before checkpoint
        # state restoration begins.
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
