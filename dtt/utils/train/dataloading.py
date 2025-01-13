from dtt.utils import Config

import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

import os
from pathlib import Path
from typing import List


class BaseShardDataset(IterableDataset):
    """
    Base class for datasets that iteratively load token sequences from numpy file shards.
    To customise the distribution strategy you must:
    - Inherit from this class.
    - Set `nshards_per_epoch` in your `__init__`.
    - Implement your own `_split_shard_paths` and `_split_shard_ids` in order to
    specify how which shards each worker will process and which samples from that shard
    each worker will process.
    - Optionally specify how to seed workers rng by overriding the `seed_worker` method.
    """

    def __init__(
        self, 
        shard_paths: List[Path],
        seq_len: int,
        overlap: int, 
        rank: int,
        world_size: int,
        seed: int = 90
    ):
        self.seq_len = seq_len
        self.idx_step = seq_len - overlap
        self.seed = seed

        self.rank = rank
        self.world_size = world_size
        self.shard_paths = shard_paths
        self.nshards = len(shard_paths)
        
        # The following attributes will be set by the worker_init_fn (iff num_workers > 0):
        self.worker_id = 0
        self.num_workers = 1
        self.worker_rng = None

    def __iter__(self):
        shuffled_shard_paths = self._shuffle_shard_paths()
        worker_shard_paths = self._split_shard_paths(shuffled_shard_paths)

        for shard_path in worker_shard_paths:
            yield from self._shard_iter(shard_path)

    def _shard_iter(self, shard_path):
        """Generator to yield sequences of tokens from a given shard"""
        shard, ids = self._load_shard_and_shuffle_ids(shard_path)
        worker_ids = self._split_shard_ids(ids) 

        for idx in worker_ids:
            sample = shard[idx : idx + self.seq_len]
            targets = shard[idx + 1 : idx + self.seq_len + 1]
            yield sample, targets

    def _split_shard_paths(self, shard_paths):
        raise NotImplementedError()

    def _split_shard_ids(self, shard_ids):
        raise NotImplementedError()

    def _shuffle_shard_paths(self):
        # This shuffle is the same on all ranks:
        shuffled_path_ids = torch.randperm(self.nshards)[: self.nshards_per_epoch]
        shuffled_shard_paths = [self.shard_paths[i] for i in shuffled_path_ids]

        return shuffled_shard_paths

    def _load_shard_and_shuffle_ids(self, shard_path):
        data = np.load(shard_path, allow_pickle=True).astype(np.int64)
        torch_data = torch.from_numpy(data)
        max_idx = len(torch_data) - self.seq_len

        # Valid starting ids for token sequences
        ids = torch.arange(0, max_idx, self.idx_step)
        shuffled_ids = ids[torch.randperm(len(ids), generator=self.worker_rng)]

        return torch_data, shuffled_ids
        
    def __len__(self):
        """Rough estimate of dataset length (currently only used for input to OneCycleLR)"""
        shard_size = np.load(self.shard_paths[0], mmap_mode="r").size
        length = (self.nshards_per_epoch // self.world_size) * (
            shard_size / self.idx_step
        )
        return int(length)

    def seed_worker(self, worker_id):
        """
        This is a worker_init_fn that establishes a single global source of rng for use in dataloader workers.
        This means that the default/global source of rng will perform the same shuffles for all workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = worker_id
        self.num_workers = worker_info.num_workers

        # By default torch will seed each worker with `global_seed + worker_id`
        # We override this to give global rng across all workers
        torch.manual_seed(self.seed)


class UniqueShardDataset(BaseShardDataset):
    """
    This `IterableDataset` ensures no two workers ever load the same shard in the same epoch.

    It is designed for distributed environments where there are more
    shards than ranks. 
    """
        
    def __init__(
        self, 
        shard_paths: List[Path],
        seq_len: int,
        overlap: int, 
        rank: int,
        world_size: int,
    ):
        super().__init__(
            shard_paths,
            seq_len,
            overlap,
            rank,
            world_size
        )

        assert self.nshards >= world_size,(
            f"There are {world_size} processes, but only {self.nshards} shards, "
            "so an even split would mean zero shards on each process."
        )

        self.nshards_per_epoch = world_size * (self.nshards // world_size)
            
    def _split_shard_paths(self, shard_paths):
        return shard_paths[self.rank :: self.world_size][self.worker_id :: self.num_workers]

    def _split_shard_ids(self, shard_ids):
        return shard_ids

    def seed_worker(self, worker_id):
        """
        This is a worker_init_fn that establishes two sources of rng for use in dataloader workers:
        1) The global torch source of rng. This means that all workers on all ranks can perform the same shuffles when needed.
        2) A local source of rng that is different for all workers across all ranks.
        """
        super().seed_worker(worker_id)
        
        # Create second source of rng that is workerwise unique
        self.worker_rng = torch.Generator()
        self.worker_rng.manual_seed(self.seed + 2**self.rank * 3**worker_id)


class SharedShardDataset(BaseShardDataset):
    """
    This `IterableDataset` loads the same shard on all worker processes and partitions that
    shard among those workers.
    """
        
    def __init__(
        self, 
        shard_paths: List[Path],
        seq_len: int,
        overlap: int, 
        rank: int,
        world_size: int,
    ):
        super().__init__(
            shard_paths,
            seq_len,
            overlap,
            rank,
            world_size
        ) 

        self.nshards_per_epoch = self.nshards

    def _split_shard_paths(self, shard_paths):
        return shard_paths

    def _split_shard_ids(self, shard_ids):
        return shard_ids[self.rank :: self.world_size][self.worker_id :: self.num_workers]

    def seed_worker(self, worker_id):
        """
        This is a worker_init_fn that establishes a single source of rng for use in dataloader workers.
        This means that the default/global source of rng will perform the same shuffles for all workers.
        This function also sets the specific `worker_rng` to be the same as the default/global generator.
        This means that any shuffles (i.e `shard_ids`) that depend on the `worker_rng` will be the same 
        for all workers.
        """
        super().seed_worker(worker_id)
        
        self.worker_rng = torch.default_generator


def get_dataloader(path: Path, split: str, cfg: Config) -> DataLoader:
    """
    Creates a `Dataloader` for distributed training.

    The loader will use a dataset that is an implementation of `BaseShardDataloader`.
    The specific implementation will be chosen as follows:

    - If we were to partition the shards as equally as possible between all ranks,
    each rank would not necessarily get the same number of shards (it could differ by
    at most 1).
    - If the difference between the maximum and minimum number of shards* is makes up
    too large a percentage of the minimum number of shards we will use `SharedShardDataset`.
    - Otherwise we use `UniqueShardDataset`.

    The point of this is to try and minimise idle processes/GPUs when the dataset has a
    small number of shards (e.g. the validation dataset), but also minimise the number of
    times we need to load a new shard when the dataset has a large number of shards (e.g.
    the training dataset).
    """
    paths = [path / shard for shard in sorted(os.listdir(path)) if split in shard]
    
    quotient, remainder = divmod(len(paths), cfg.world_size)
    loader_ratio = 1 if remainder==0 else quotient / (quotient + 1)

    if loader_ratio < cfg.min_loader_ratio:
        dataset = SharedShardDataset(
            paths,
            cfg.n_ctx,
            cfg.overlap,
            cfg.rank,
            cfg.world_size
        )

    else:
        dataset = UniqueShardDataset(
            paths,
            cfg.n_ctx,
            cfg.overlap,
            cfg.rank,
            cfg.world_size           
        )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        worker_init_fn=dataset.seed_worker,
    )

    return dataloader
