"""Fixed validation pairs and resumable checkpoint selection for all four LAMs."""
from __future__ import annotations

import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from lightning import Callback
from torch.utils.data import DataLoader, Dataset

from .navigation_dataset import load_navigation_frame
from .navigation_evaluation.catalog import TrajectoryRecord, compute_action
from .navigation_evaluation.io import atomic_write_json


class FixedValidationPairs(Dataset):
    """Read only selected trajectory metadata; use the original frame/action transforms."""
    def __init__(self, records, image_height=240, image_width=320):
        self.records = records
        self.image_height, self.image_width = image_height, image_width

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        videos = torch.stack([
            load_navigation_frame(Path(row[key]), image_height=self.image_height,
                                  image_width=self.image_width)
            for key in ('anchor', 'target')
        ])
        return {'videos': videos, 'actions': torch.tensor(row['action'], dtype=torch.float32).unsqueeze(0),
                'action_mask': torch.tensor(True)}


def validation_records(source_specs, count=192, seed=32, max_offset=8,
                       action_min=(-2.5, -4.), action_max=(5., 4.)):
    """Sample fixed pairs from explicit held-out lists without inspecting all images."""
    rng = random.Random(seed)
    sources = []
    for spec in source_specs:
        names = Path(spec['split_manifest']).read_text().splitlines()
        names = [name.strip() for name in names if name.strip()]
        if not names:
            raise ValueError(f"Empty validation list: {spec['split_manifest']}")
        sources.append((spec, names))
    if not sources or count < 1:
        raise ValueError('Validation needs at least one source and one pair')
    records = []
    for index in range(count):
        spec, names = sources[index % len(sources)]
        # Select metadata only for the trajectory needed by this pair.
        name = names[rng.randrange(len(names))]
        directory = Path(spec.get('data_root', spec.get('root'))) / name
        with (directory / 'traj_data.pkl').open('rb') as handle:
            metadata = pickle.load(handle)
        positions = np.asarray(metadata['position'], dtype=np.float64)
        yaw = np.asarray(metadata['yaw'], dtype=np.float64).reshape(-1)
        trajectory = TrajectoryRecord(str(spec.get('display_name', 'validation')), 'val',
                                      name, directory, float(spec['waypoint_spacing']), positions, yaw)
        if trajectory.length < 2:
            raise ValueError(f'Validation trajectory has fewer than two frames: {directory}')
        limit = min(max_offset, trajectory.length - 1)
        offsets = list(range(-limit, 0)) + list(range(1, limit + 1))
        offset = offsets[index % len(offsets)]
        anchor = rng.randrange(max(0, -offset), trajectory.length - max(0, offset))
        target = anchor + offset
        action = compute_action(trajectory, anchor, target,
                                action_min=action_min, action_max=action_max).normalized
        records.append({'anchor': str(directory / f'{anchor}.jpg'),
                        'target': str(directory / f'{target}.jpg'), 'action': list(action)})
    return records


class PeriodicEvaluation(Callback):
    """Evaluate shared_step without changing training RNG; checkpoint full Trainer state."""
    def __init__(self, run_dir, source_specs, every_n_steps=1000, save_every_n_steps=1000,
                 periodic_every_n_steps=10000, sample_count=192, batch_size=8,
                 num_workers=0, seed=32, image_height=240, image_width=320,
                 max_frame_offset=8, action_min=(-2.5, -4.), action_max=(5., 4.), keep_periodic=3):
        self.run_dir = Path(run_dir)
        self.source_specs = source_specs
        self.every_n_steps, self.save_every_n_steps = every_n_steps, save_every_n_steps
        self.periodic_every_n_steps = periodic_every_n_steps
        self.sample_count, self.batch_size, self.num_workers = sample_count, batch_size, num_workers
        self.seed, self.image_height, self.image_width = seed, image_height, image_width
        self.max_frame_offset, self.action_min, self.action_max = max_frame_offset, action_min, action_max
        if min(every_n_steps, save_every_n_steps, periodic_every_n_steps, sample_count, batch_size) < 1:
            raise ValueError('Evaluation/save intervals and sample/batch counts must be positive')
        self.keep_periodic = keep_periodic
        self.rng_by_rank = []
        self.best_score = float('inf')
        self.last_eval_step = -1
        self.last_saved_step = -1
        self.loader = None

    def state_dict(self):
        return {'best_score': self.best_score, 'last_eval_step': self.last_eval_step,
                'last_saved_step': self.last_saved_step, 'rng_by_rank': self.rng_by_rank}

    def load_state_dict(self, state_dict):
        for name in ('best_score', 'last_eval_step', 'last_saved_step', 'rng_by_rank'):
            if name in state_dict:
                setattr(self, name, state_dict[name])

    def on_fit_start(self, trainer, pl_module):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)
        if trainer.is_global_zero:
            manifest = self.run_dir / 'validation_pairs.json'
            if manifest.exists():
                records = json.loads(manifest.read_text())
            else:
                records = validation_records(self.source_specs, self.sample_count, self.seed,
                                             self.max_frame_offset, self.action_min, self.action_max)
                atomic_write_json(manifest, records)
            self.loader = DataLoader(FixedValidationPairs(records, self.image_height, self.image_width),
                                     batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        trainer.strategy.barrier()

    def _evaluate(self, trainer, model):
        metrics = None
        if trainer.is_global_zero:
            was_training = model.training
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            devices = [model.device.index or 0] if model.device.type == 'cuda' else []
            sums, count = {}, 0
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    model.eval()
                    for batch in self.loader:
                        batch = {key: value.to(model.device) for key, value in batch.items()}
                        with trainer.precision_plugin.forward_context():
                            outputs, loss, auxiliary = model.shared_step(batch)
                            auxiliary = dict(auxiliary)
                            if 'recon' in outputs:
                                auxiliary.update(dict(model._evaluation_metrics(batch, outputs)))
                                if count == 0:
                                    model.log_images(batch, outputs, 'val')
                            if 'action_recon' in outputs:
                                auxiliary['action_mae'] = (outputs['action_recon'] - batch['actions']).abs().mean()
                        size = len(batch['videos'])
                        for key, value in {'loss': loss, **dict(auxiliary)}.items():
                            sums[key] = sums.get(key, 0.) + float(value.detach().float()) * size
                        count += size
                        model.lam.mu_record = None
                metrics = {key: value / count for key, value in sums.items()}
                if not math.isfinite(metrics['loss']):
                    raise RuntimeError(f'Non-finite validation loss: {metrics}')
            finally:
                model.train(was_training)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        return trainer.strategy.broadcast(metrics, src=0)

    def on_train_start(self, trainer, pl_module):
        if self.rng_by_rank:
            state = self.rng_by_rank[min(trainer.global_rank, len(self.rng_by_rank) - 1)]
            random.setstate(state['python'])
            ns = state['numpy']
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            torch.set_rng_state(state['torch'].cpu())
            if pl_module.device.type == 'cuda' and state.get('cuda') is not None:
                torch.cuda.set_rng_state(state['cuda'].cpu(), pl_module.device)

    def _save(self, trainer, name):
        numpy_state = np.random.get_state()
        state = {'python': random.getstate(),
                 'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                 'torch': torch.get_rng_state(),
                 'cuda': (torch.cuda.get_rng_state(trainer.lightning_module.device)
                          if trainer.lightning_module.device.type == 'cuda' else None)}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rng_by_rank = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(self.rng_by_rank, state)
        else:
            self.rng_by_rank = [state]
        path = self.run_dir / 'checkpoints' / name
        temporary = path.with_suffix('.tmp.ckpt')
        trainer.save_checkpoint(temporary, weights_only=False)
        if trainer.is_global_zero:
            temporary.replace(path)
            if name.startswith('step=') and self.keep_periodic > 0:
                periodic = sorted(path.parent.glob('step=*.ckpt'), key=lambda p: int(p.stem.split('=')[1]))
                for expired in periodic[:-self.keep_periodic]:
                    expired.unlink()
        trainer.strategy.barrier()

    def _cycle(self, trainer, model, *, final=False):
        step = trainer.global_step
        if step <= 0:
            return
        evaluate = (final or step % self.every_n_steps == 0) and step != self.last_eval_step
        improved = False
        if evaluate:
            metrics = self._evaluate(trainer, model)
            self.last_eval_step = step
            improved = metrics['loss'] < self.best_score
            if improved:
                self.best_score = metrics['loss']
            if trainer.is_global_zero:
                atomic_write_json(self.run_dir / 'eval' / f'step={step}.json',
                                  {'step': step, 'best_loss': self.best_score, **metrics})
                if trainer.logger:
                    for logger in trainer.loggers:
                        logger.log_metrics({f'val/{key}': value for key, value in metrics.items()}, step=step)
                print(f'[validation] step={step} loss={metrics["loss"]:.7g} best={self.best_score:.7g}', flush=True)
        save = final or evaluate or step % self.save_every_n_steps == 0 or step % self.periodic_every_n_steps == 0
        if save and (step != self.last_saved_step or improved):
            self.last_saved_step = step
            if improved:
                self._save(trainer, 'best.ckpt')
            if step % self.periodic_every_n_steps == 0:
                self._save(trainer, f'step={step}.ckpt')
            self._save(trainer, 'latest.ckpt')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loader = trainer.train_dataloader
        if hasattr(loader, 'mark_completed'):
            loader.mark_completed(trainer.current_epoch, batch_idx + 1)
        self._cycle(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._cycle(trainer, pl_module, final=True)
