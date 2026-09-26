"""Resume deterministic navigation requests without decoding consumed batches."""
from itertools import islice

from torch.utils.data import DataLoader


class _ResumeSampler:
    def __init__(self, sampler, skip):
        self.sampler, self.skip = sampler, skip

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        return islice(iter(self.sampler), skip, None)


class ResumableDataLoader(DataLoader):
    """Lightning saves this cursor before constructing/prefetching the next iterator.

    NavigationPairDataset derives each example from its logical index and epoch,
    so skipping sampler requests also preserves augmentation with multiple workers.
    The callback records completed batches; worker prefetch never advances the cursor.
    """

    def mark_completed(self, epoch, batches):
        self._completed_epoch = int(epoch)
        self._completed_batches = int(batches)

    def state_dict(self):
        return {'epoch': getattr(self, '_completed_epoch', 0),
                'completed_batches': getattr(self, '_completed_batches', 0),
                'batch_size': self.batch_size, 'num_batches': len(self)}

    def load_state_dict(self, state):
        if state['batch_size'] != self.batch_size or state['num_batches'] != len(self):
            raise ValueError('Resume needs the original batch size and per-rank epoch length')
        epoch, batches = state['epoch'], state['completed_batches']
        self.mark_completed(epoch, batches)
        if batches >= len(self):
            epoch, batches = epoch + 1, 0
        # setup_data() creates workers before on_train_epoch_start(). Set the
        # restored epoch now so that prefetched samples use the right epoch.
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        self.batch_sampler.sampler = _ResumeSampler(self.sampler, batches * self.batch_size)
