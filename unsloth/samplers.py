from __future__ import annotations

from torch.utils.data import DistributedSampler
import torch.distributed as dist


class ContextParallelDistributedSampler(DistributedSampler):
    """
    A thin wrapper around DistributedSampler that keeps batches identical
    within every context-parallel group.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        context_parallel_size: int = 1,
        **kwargs,
    ):
        if context_parallel_size < 1:
            raise ValueError("context_parallel_size must be >= 1")
        world_size = num_replicas if num_replicas is not None else dist.get_world_size()
        if world_size % context_parallel_size != 0:
            raise RuntimeError(
                "World size must be divisible by context_parallel_size "
                f"({world_size} % {context_parallel_size} != 0)."
            )
        data_parallel_world = world_size // context_parallel_size
        process_rank = rank if rank is not None else dist.get_rank()
        data_parallel_rank = process_rank // context_parallel_size
        super().__init__(
            dataset,
            num_replicas = data_parallel_world,
            rank = data_parallel_rank,
            **kwargs,
        )
        self.context_parallel_size = context_parallel_size
