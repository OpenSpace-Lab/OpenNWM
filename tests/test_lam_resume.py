"""Exercise the actual Lightning checkpoint path, including prefetched requests."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lam'))
from lightning import LightningModule, Trainer, seed_everything
from lam.resumable_loader import ResumableDataLoader
from lam.training_runtime import PeriodicEvaluation


@pytest.mark.parametrize('workers', [0, 2])
def test_cursor_skips_completed_not_prefetched_batches(workers):
    dataset = TensorDataset(torch.arange(12))

    def loader():
        return ResumableDataLoader(
            dataset, batch_size=2, num_workers=workers,
            sampler=DistributedSampler(dataset, num_replicas=1, rank=0, seed=7),
            generator=torch.Generator().manual_seed(11))

    original = loader()
    original.sampler.set_epoch(3)
    iterator = iter(original)
    consumed = [next(iterator)[0] for _ in range(2)]
    original.mark_completed(3, 2)
    state = original.state_dict()
    expected = torch.cat([batch[0] for batch in iterator])
    resumed = loader()
    resumed.load_state_dict(state)
    assert len(resumed) == len(original) == 6
    actual = torch.cat([batch[0] for batch in resumed])
    assert torch.equal(actual, expected)
    assert len(set(torch.cat(consumed + [actual]).tolist())) == 12


class _TrainingModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.requests = []

    def training_step(self, batch, batch_idx):
        self.requests.extend(batch[0].tolist())
        target = torch.rand(()) + random.random() + float(np.random.random())
        return (self.weight * (batch[0].float().mean() + 1) - target).square()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class _CheckpointOnly(PeriodicEvaluation):
    def on_fit_start(self, trainer, pl_module):
        (self.run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    def _evaluate(self, trainer, model):
        return {'loss': float(model.weight.detach().square())}


@pytest.mark.parametrize('workers', [0, 2])
def test_mid_epoch_resume_matches_continuous_rng_and_batches(tmp_path, workers):
    def run(directory, steps, checkpoint=None):
        model = _TrainingModel()
        loader = ResumableDataLoader(
            TensorDataset(torch.arange(12)), batch_size=2, num_workers=workers,
            generator=torch.Generator().manual_seed(17))
        callback = _CheckpointOnly(directory, [], every_n_steps=2, save_every_n_steps=2,
                                   periodic_every_n_steps=2)
        trainer = Trainer(accelerator='cpu', devices=1, max_steps=steps,
                          logger=False, enable_checkpointing=False, enable_model_summary=False,
                          enable_progress_bar=False, num_sanity_val_steps=0, callbacks=[callback])
        trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint)
        return model, trainer.optimizers[0].state_dict()

    seed_everything(23, workers=True)
    continuous, optimizer = run(tmp_path / 'continuous', 8)
    seed_everything(23, workers=True)
    first, _ = run(tmp_path / 'split', 2)
    seed_everything(999, workers=True)
    resumed, resumed_optimizer = run(
        tmp_path / 'split', 8, tmp_path / 'split/checkpoints/latest.ckpt')
    assert first.requests + resumed.requests == continuous.requests
    assert torch.equal(resumed.weight, continuous.weight)
    for key, expected in optimizer['state'][0].items():
        actual = resumed_optimizer['state'][0][key]
        assert torch.equal(actual, expected) if torch.is_tensor(expected) else actual == expected
