from __future__ import annotations

import contextlib
import functools
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple
import sys
import os
import contextvars
import hashlib

import torch
import torch.nn.functional as F
from packaging.version import Version
import torch.distributed as dist

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor import DeviceMesh
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None

from .device_type import DEVICE_TYPE_TORCH

_ACTIVE_MANAGER: contextvars.ContextVar[Optional["ContextParallelManager"]] = (
    contextvars.ContextVar("unsloth_active_cp_manager", default = None)
)


def get_active_context_parallel_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER.get()


def is_context_parallel_active() -> bool:
    manager = get_active_context_parallel_manager()
    return bool(manager and manager.enabled)


def _torch_version() -> Version:
    return Version(torch.__version__.split("+")[0])


@dataclass
class ContextParallelSettings:
    """Lightweight container for the knobs we expose to users."""

    size: int = field(
        default = 1,
        metadata = {
            "help": (
                "Number of ranks that should participate in context parallelism. "
                "Set to >1 only when running under torch.distributed / accelerate."
            )
        },
    )
    seq_dim: int = field(
        default = 1,
        metadata = {
            "help": (
                "Sequence dimension for buffers passed to context parallel. "
                "Use 1 for standard (batch, seq_len) language-model tensors."
            )
        },
    )
    buffer_names: Tuple[str, ...] = field(
        default_factory = lambda: (
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "shift_labels",
        ),
        metadata = {
            "help": (
                "Names inside the Trainer input batch that should be sharded "
                "across the context dimension."
            )
        },
    )
    no_restore_buffer_names: Tuple[str, ...] = field(
        default_factory = lambda: (
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "shift_labels",
        ),
        metadata = {
            "help": (
                "Subset of `buffer_names` that do not need to be restored after the "
                "context parallel region exits. For context parallelism with load "
                "balancing, all buffers should remain sharded."
            )
        },
    )

    def __post_init__(self) -> None:
        self.buffer_names = _ensure_tuple_contains(self.buffer_names, "position_ids")
        self.no_restore_buffer_names = _ensure_tuple_contains(
            self.no_restore_buffer_names,
            "position_ids",
        )

    @classmethod
    def from_args(cls, args: Optional[object]) -> "ContextParallelSettings":
        if args is None:
            return cls()

        def _get(name: str, default):
            return getattr(args, name, default)

        size = int(_get("context_parallel_size", 1))
        seq_dim = int(_get("context_parallel_seq_dim", 1))
        buffer_names = _as_tuple(
            _get(
                "context_parallel_buffer_names",
                (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "shift_labels",
                ),
            )
        )
        buffer_names = _ensure_tuple_contains(buffer_names, "position_ids")
        buffer_names = _ensure_tuple_contains(buffer_names, "shift_labels")
        no_restore = _as_tuple(
            _get(
                "context_parallel_no_restore",
                (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "shift_labels",
                ),
            )
        )
        no_restore = _ensure_tuple_contains(no_restore, "position_ids")
        no_restore = _ensure_tuple_contains(no_restore, "shift_labels")
        return cls(
            size = size,
            seq_dim = seq_dim,
            buffer_names = buffer_names,
            no_restore_buffer_names = no_restore,
        )


def _as_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return tuple(item for item in items if item)
    return tuple(value)


def _ensure_tuple_contains(values: Tuple[str, ...], name: str) -> Tuple[str, ...]:
    if name in values:
        return values
    return values + (name,)


class ContextParallelManager:
    """Encapsulates everything needed to toggle PyTorch context parallelism."""

    def __init__(self, settings: ContextParallelSettings):
        self.settings = settings
        self.enabled = (
            context_parallel is not None
            and DeviceMesh is not None
            and settings.size > 1
        )
        self._mesh: Optional[DeviceMesh] = None
        self._accelerate_mesh: Optional[DeviceMesh] = None
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._cp_rank_index: int = 0
        self._dp_world_size: int = 1
        self._world_size: int = 1
        self._no_restore_lookup = set(settings.no_restore_buffer_names)
        self._cached_num_items: Optional[torch.Tensor | float | int] = None
        self._report_loss: Optional[torch.Tensor] = None
        self._report_tokens: Optional[torch.Tensor] = None
        self._last_global_seq_len: Optional[int] = None
        self._debug_raw_input_ids: Optional[torch.Tensor] = None
        self._verify_environment()
        if self.enabled:
            self._mesh = self._build_mesh()
            self._accelerate_mesh = self._build_accelerate_mesh()

    def _verify_environment(self) -> None:
        if not self.enabled:
            if self.settings.size > 1 and context_parallel is None:
                warnings.warn(
                    "Context parallelism requested but your PyTorch build does not expose "
                    "`torch.distributed.tensor.experimental.context_parallel`. "
                    "Please upgrade to PyTorch 2.4 or newer.",
                    stacklevel = 2,
                )
            return

        min_supported = Version("2.4.0")
        if _torch_version() < min_supported:
            raise RuntimeError(
                f"Context parallelism requires PyTorch >= {min_supported}, got {_torch_version()}."
            )
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is required for context parallelism.")
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed process group must be initialized before enabling context parallelism."
            )

        world_size = torch.distributed.get_world_size()
        self._world_size = world_size
        if world_size < self.settings.size:
            raise RuntimeError(
                f"context_parallel_size={self.settings.size} requires at least that many ranks; got {world_size}."
            )
        if world_size % self.settings.size != 0:
            raise RuntimeError(
                "Number of distributed ranks must be divisible by context_parallel_size "
                f"({world_size} % {self.settings.size} != 0)."
            )

    def _build_mesh(self) -> DeviceMesh:
        rank = torch.distributed.get_rank()
        group_index = rank // self.settings.size
        start = group_index * self.settings.size
        cp_ranks = torch.arange(start, start + self.settings.size, dtype = torch.int64)
        mesh = DeviceMesh(DEVICE_TYPE_TORCH, cp_ranks)
        self._cp_group = mesh.get_group()
        self._cp_rank_index = int(rank - start)
        return mesh

    def _build_accelerate_mesh(self) -> Optional[DeviceMesh]:
        if not self.enabled:
            return None
        world_size = torch.distributed.get_world_size()
        if world_size % self.settings.size != 0:
            return None
        dp_world_size = world_size // self.settings.size
        mesh = torch.arange(world_size, dtype = torch.int64).reshape(
            dp_world_size,
            self.settings.size,
        )
        self._dp_world_size = dp_world_size
        return DeviceMesh(
            DEVICE_TYPE_TORCH,
            mesh,
            mesh_dim_names = ("dp_replicate", "cp"),
        )

    def __bool__(self) -> bool:
        return self.enabled

    @property
    def accelerate_mesh(self) -> Optional[DeviceMesh]:
        return self._accelerate_mesh

    @property
    def data_parallel_world_size(self) -> int:
        return self._dp_world_size

    @property
    def cp_rank_index(self) -> int:
        return self._cp_rank_index

    @property
    def process_group(self) -> Optional[dist.ProcessGroup]:
        return self._cp_group

    @property
    def last_global_seq_len(self) -> Optional[int]:
        return self._last_global_seq_len

    def data_parallel_rank(self) -> int:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return 0
        if not self.enabled or self._dp_world_size == self._world_size:
            return torch.distributed.get_rank()
        return torch.distributed.get_rank() // self.settings.size

    def _collect_buffers(
        self, inputs: dict[str, torch.Tensor]
    ) -> Tuple[list[torch.Tensor], list[int], set[torch.Tensor]]:
        buffers: list[torch.Tensor] = []
        seq_dims: list[int] = []
        no_restore: set[torch.Tensor] = set()
        for name in self.settings.buffer_names:
            tensor = inputs.get(name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim <= self.settings.seq_dim:
                continue
            buffers.append(tensor)
            seq_dims.append(self.settings.seq_dim)
            if name in self._no_restore_lookup:
                no_restore.add(tensor)
        return buffers, seq_dims, no_restore

    def _ensure_position_ids(self, inputs: dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        if isinstance(inputs.get("position_ids"), torch.Tensor):
            return
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return
        if input_ids.ndim <= self.settings.seq_dim:
            return
        seq_len = input_ids.size(self.settings.seq_dim)
        if seq_len <= 0:
            return
        self._last_global_seq_len = seq_len
        device = input_ids.device
        dtype = torch.long
        base = torch.arange(seq_len, dtype = dtype, device = device)
        view_shape = [1] * input_ids.ndim
        view_shape[self.settings.seq_dim] = seq_len
        positions = base.view(view_shape).expand_as(input_ids)
        positions = positions.to(dtype = torch.long)
        inputs["position_ids"] = positions
        if _cp_debug_enabled():
            preview = positions.flatten().tolist()[: min(16, positions.numel())]
            _cp_debug(
                f"[CP-DEBUG][cp-rank={self._cp_rank_index}] synthesized position_ids preview={preview}"
            )

    def _ensure_shift_labels(self, inputs: dict[str, torch.Tensor]) -> None:
        """
        Create pre-shifted labels for context parallelism with load balancing.

        With load balancing, tokens are reordered across ranks. The standard causal LM
        loss shifts labels locally (labels[1:]), but this doesn't work with reordered
        tokens. Instead, we pre-shift labels globally before sharding:
          - Pad labels with -100 at the end: [l0, l1, ..., l199, -100]
          - Take [1:]: [l1, l2, ..., l199, -100]

        After sharding with load balancing, each rank gets the correct "next token"
        labels for its positions, enabling proper loss computation without local shifting.
        """
        if not self.enabled:
            return
        if isinstance(inputs.get("shift_labels"), torch.Tensor):
            return
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            return
        if labels.ndim <= self.settings.seq_dim:
            return
        # Pad labels with -100 at the end, then take [1:] to get shifted labels
        # This matches transformers' approach in _prepare_context_parallel_inputs
        ignore_index = -100
        # Pad on the sequence dimension (last dim for 2D labels)
        padded = F.pad(labels, (0, 1), value = ignore_index)
        shift_labels = padded[:, 1:].contiguous()
        inputs["shift_labels"] = shift_labels
        if _cp_debug_enabled():
            preview = shift_labels.flatten().tolist()[: min(16, shift_labels.numel())]
            _cp_debug(
                f"[CP-DEBUG][cp-rank={self._cp_rank_index}] synthesized shift_labels preview={preview}"
            )

    def _debug_validate_buffers(
        self,
        buffers: list[torch.Tensor],
    ) -> None:
        if not _cp_debug_enabled():
            return
        if not buffers or self._cp_group is None:
            return
        summaries: list[tuple] = []
        for tensor in buffers:
            if not torch.is_tensor(tensor):
                continue
            cpu_tensor = tensor.detach().to("cpu").contiguous()
            checksum = hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()
            shape = tuple(cpu_tensor.shape)
            dtype = str(cpu_tensor.dtype)
            summaries.append((shape, dtype, checksum))
        if not summaries:
            return
        gathered: list[list[tuple]] = [None] * self.settings.size  # type: ignore[list-item]
        dist.all_gather_object(gathered, summaries, group = self._cp_group)
        reference = gathered[0]
        for idx, summary in enumerate(gathered[1:], start = 1):
            if summary != reference:
                raise RuntimeError(
                    f"Context parallel buffers differ across ranks before sharding (rank {idx} mismatch)."
                )
        _cp_debug(
            f"[CP-DEBUG][cp-rank={self._cp_rank_index}] pre-shard buffers match across {self.settings.size} ranks."
        )

    @contextlib.contextmanager
    def apply(self, inputs: dict[str, torch.Tensor]) -> Iterator[None]:
        token = None
        if self.enabled:
            token = _ACTIVE_MANAGER.set(self)
        try:
            if not self.enabled or self._mesh is None:
                yield
                return
            self._adjust_num_items_in_batch(inputs)
            self._ensure_position_ids(inputs)
            self._ensure_shift_labels(inputs)
            buffers, seq_dims, no_restore = self._collect_buffers(inputs)
            self._debug_validate_buffers(buffers)
            if not buffers:
                yield
                return
            with context_parallel(
                self._mesh,
                buffers = buffers,
                buffer_seq_dims = seq_dims,
                no_restore_buffers = no_restore,
            ):
                yield
        finally:
            if token is not None:
                _ACTIVE_MANAGER.reset(token)

    def remember_raw_input_ids(self, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            self._debug_raw_input_ids = None
            return
        self._debug_raw_input_ids = tensor.detach().to("cpu").contiguous()

    def consume_raw_input_ids(self) -> Optional[torch.Tensor]:
        raw = self._debug_raw_input_ids
        self._debug_raw_input_ids = None
        return raw

    @contextlib.contextmanager
    def replay_context(self):
        if not self.enabled:
            yield
            return
        token = _ACTIVE_MANAGER.set(self)
        try:
            yield
        finally:
            _ACTIVE_MANAGER.reset(token)

    def _rank_slice(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor):
            return None
        if tensor.ndim <= self.settings.seq_dim:
            return tensor
        seq_len = tensor.size(self.settings.seq_dim)
        if seq_len % self.settings.size != 0:
            raise RuntimeError(
                "Sequence length must be divisible by context_parallel_size when computing loss weights."
            )
        chunk = seq_len // self.settings.size
        start = self._cp_rank_index * chunk
        return tensor.narrow(self.settings.seq_dim, start, chunk)

    def _loss_weight(self, inputs: dict[str, torch.Tensor], reference: torch.Tensor):
        if not isinstance(reference, torch.Tensor):
            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] _loss_weight: reference is not tensor cp-rank={self._cp_rank_index}"
                )
            return None
        labels = self._rank_slice(inputs.get("labels"))
        if isinstance(labels, torch.Tensor):
            weight = labels.ne(-100).sum()
            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] _loss_weight: labels shape={labels.shape} "
                    f"valid_tokens={weight.item()} cp-rank={self._cp_rank_index}"
                )
            if weight.item() > 0:
                return weight.to(device = reference.device, dtype = reference.dtype)
        attention_mask = self._rank_slice(inputs.get("attention_mask"))
        if isinstance(attention_mask, torch.Tensor):
            weight = attention_mask.sum()
            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] _loss_weight: attention_mask shape={attention_mask.shape} "
                    f"sum={weight.item()} cp-rank={self._cp_rank_index}"
                )
            if weight.item() > 0:
                return weight.to(device = reference.device, dtype = reference.dtype)
        if _cp_debug_enabled():
            _cp_debug(
                f"[CP-DEBUG][focus] _loss_weight: returning None (no valid weight found) cp-rank={self._cp_rank_index}"
            )
        return None

    def _local_valid_token_count(
        self, inputs: dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        labels = self._rank_slice(inputs.get("labels"))
        if isinstance(labels, torch.Tensor):
            return labels.ne(-100).sum()
        attention_mask = self._rank_slice(inputs.get("attention_mask"))
        if isinstance(attention_mask, torch.Tensor):
            return attention_mask.sum()
        return None

    def _adjust_num_items_in_batch(self, inputs: dict[str, torch.Tensor]) -> None:
        if not self.enabled or "num_items_in_batch" not in inputs:
            self._cached_num_items = None
            return
        local_tokens = self._local_valid_token_count(inputs)
        if local_tokens is None:
            self._cached_num_items = None
            return
        value = inputs["num_items_in_batch"]
        if torch.is_tensor(value):
            local_tokens = local_tokens.to(device = value.device, dtype = value.dtype)
            inputs["num_items_in_batch"] = local_tokens
            self._cached_num_items = local_tokens
        else:
            self._cached_num_items = local_tokens.item()
            inputs["num_items_in_batch"] = self._cached_num_items

    def consume_num_items_override(self):
        value = self._cached_num_items
        self._cached_num_items = None
        return value

    def _set_report_loss(self, value: torch.Tensor) -> None:
        self._report_loss = value.detach() if torch.is_tensor(value) else None

    def consume_report_loss(self) -> Optional[torch.Tensor]:
        value = self._report_loss
        self._report_loss = None
        return value

    def _set_report_tokens(self, value: torch.Tensor) -> None:
        self._report_tokens = value.detach() if torch.is_tensor(value) else None

    def consume_report_tokens(self) -> Optional[torch.Tensor]:
        value = self._report_tokens
        self._report_tokens = None
        return value

    def reduce_loss(self, loss, inputs):
        if not self.enabled or self.settings.size <= 1 or self._cp_group is None:
            return loss

        # Debug: Check if labels exist in inputs after context manager
        if _cp_debug_enabled():
            labels_in = inputs.get("labels")
            _cp_debug(
                f"[CP-DEBUG][focus] reduce_loss inputs.labels "
                f"exists={labels_in is not None} "
                f"is_tensor={torch.is_tensor(labels_in)} "
                f"shape={getattr(labels_in, 'shape', None)} "
                f"cp-rank={self._cp_rank_index}"
            )

        def _reduce_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if tensor is None or not torch.is_tensor(tensor):
                zeros = torch.zeros(
                    (),
                    dtype = tensor.dtype if torch.is_tensor(tensor) else torch.float32,
                    device = tensor.device if torch.is_tensor(tensor) else None,
                )
                return tensor, zeros

            # Count valid tokens from sharded shift_labels (stays sharded due to no_restore)
            # This gives us the correct local token count for weighted averaging
            shift_labels = inputs.get("shift_labels")
            if isinstance(shift_labels, torch.Tensor):
                # Count non-ignored tokens (-100 is ignore_index)
                local_weight = (
                    shift_labels.ne(-100)
                    .sum()
                    .to(device = tensor.device, dtype = tensor.dtype)
                )
            else:
                # Fallback: use sharded labels if available
                labels = inputs.get("labels")
                if isinstance(labels, torch.Tensor):
                    local_weight = (
                        labels.ne(-100)
                        .sum()
                        .to(device = tensor.device, dtype = tensor.dtype)
                    )
                else:
                    # Last resort: assume equal distribution
                    local_weight = torch.tensor(
                        1.0, dtype = tensor.dtype, device = tensor.device
                    )

            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] _reduce_tensor: "
                    f"raw_loss={tensor.item()} local_weight={local_weight.item()} "
                    f"cp-rank={self._cp_rank_index}"
                )

            # Convert mean to sum: sum = mean * count
            scaled = tensor * local_weight
            # All-reduce to get global sum and global count
            dist.all_reduce(scaled, op = dist.ReduceOp.SUM, group = self._cp_group)
            dist.all_reduce(local_weight, op = dist.ReduceOp.SUM, group = self._cp_group)

            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] _reduce_tensor after all_reduce: "
                    f"global_sum={scaled.item()} global_weight={local_weight.item()} "
                    f"cp-rank={self._cp_rank_index}"
                )

            return scaled, local_weight

        def _finalize(summed, tokens):
            eps = torch.finfo(summed.dtype).eps
            normalized = summed / torch.clamp(tokens, min = eps)
            self._set_report_loss(normalized)
            self._set_report_tokens(tokens)
            if _cp_debug_enabled():
                _cp_debug(
                    f"[CP-DEBUG][focus] reduce_loss _finalize: summed={summed.item()} tokens={tokens.item()} "
                    f"normalized={normalized.item()} cp-rank={self._cp_rank_index}"
                )
            return normalized

        if isinstance(loss, tuple):
            if not loss:
                return loss
            summed, tokens = _reduce_tensor(loss[0])
            normalized = _finalize(summed, tokens)
            return (normalized, *loss[1:])
        elif torch.is_tensor(loss):
            summed, tokens = _reduce_tensor(loss)
            return _finalize(summed, tokens)
        return loss


def patch_trl_for_context_parallel() -> None:
    """Monkey patches TRL's SFT stack to surface context parallel toggles."""

    if getattr(patch_trl_for_context_parallel, "_applied", False):
        return

    try:
        import trl
    except ImportError:  # pragma: no cover - optional dependency
        return

    config_cls = _patch_sft_config(trl)
    _patch_sft_trainer(trl)
    setattr(patch_trl_for_context_parallel, "_applied", True)
    setattr(config_cls, "__unsloth_context_parallel__", True)


def _patch_sft_config(trl_module):
    base_cls = trl_module.SFTConfig
    if hasattr(base_cls, "context_parallel_size"):
        return base_cls

    @dataclass
    class PatchedSFTConfig(base_cls):  # type: ignore[misc, valid-type]
        shuffle_dataset: bool = field(
            default = True,
            metadata = {
                "help": (
                    "Whether to shuffle the training dataset before each epoch. "
                    "Exposed for context-parallel experiments that require deterministic ordering."
                )
            },
        )
        context_parallel_size: int = field(
            default = 1,
            metadata = {
                "help": (
                    "Number of ranks participating in context parallelism. "
                    "Set to 1 to disable context parallelism."
                )
            },
        )
        context_parallel_seq_dim: int = field(
            default = 1,
            metadata = {
                "help": (
                    "Sequence dimension for buffers that should be sharded when context parallelism is enabled."
                )
            },
        )
        context_parallel_buffer_names: Tuple[str, ...] = field(
            default_factory = lambda: (
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
            ),
            metadata = {
                "help": (
                    "Batch keys whose tensors should be sharded across context parallel ranks. "
                    "Pass a tuple/list of keys."
                )
            },
        )
        context_parallel_no_restore: Tuple[str, ...] = field(
            default_factory = lambda: ("position_ids",),
            metadata = {
                "help": (
                    "Subset of `context_parallel_buffer_names` that do not need to be restored after "
                    "the context parallel block exits."
                )
            },
        )

    PatchedSFTConfig.__name__ = base_cls.__name__
    PatchedSFTConfig.__qualname__ = base_cls.__qualname__
    PatchedSFTConfig.__module__ = base_cls.__module__
    module = sys.modules.get(base_cls.__module__)
    if module is not None:
        setattr(module, base_cls.__name__, PatchedSFTConfig)
    trl_module.SFTConfig = PatchedSFTConfig
    if hasattr(trl_module, "trainer") and hasattr(trl_module.trainer, "sft_trainer"):
        trl_module.trainer.sft_trainer.SFTConfig = PatchedSFTConfig
    return PatchedSFTConfig


def _patch_sft_trainer(trl_module) -> None:
    trainer_cls = trl_module.SFTTrainer
    if hasattr(trainer_cls, "__unsloth_context_parallel__"):
        return

    original_init = trainer_cls.__init__
    original_compute_loss = trainer_cls.compute_loss
    original_prediction_step = trainer_cls.prediction_step
    original_training_step = trainer_cls.training_step
    original_log = trainer_cls.log
    original_get_train_sampler = getattr(trainer_cls, "_get_train_sampler", None)

    def _patch_train_sampler(original_fn):
        @functools.wraps(original_fn)
        def wrapper(self, *args, **kwargs):
            sampler = original_fn(self, *args, **kwargs)
            manager = getattr(self, "_context_parallel_manager", None)
            dataset = args[0] if args else None
            if dataset is None:
                dataset = getattr(self, "train_dataset", None)
            shuffle_dataset = getattr(self.args, "shuffle_dataset", True)
            if (
                manager
                and manager.enabled
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and dataset is not None
            ):
                dp_world = manager.data_parallel_world_size
                world_size = torch.distributed.get_world_size()
                if dp_world != world_size:
                    try:
                        from torch.utils.data.distributed import DistributedSampler
                    except ImportError:
                        return sampler
                    dp_rank = manager.data_parallel_rank()
                    shuffle = shuffle_dataset and not getattr(
                        self.args, "group_by_length", False
                    )
                    return DistributedSampler(
                        dataset,
                        num_replicas = dp_world,
                        rank = dp_rank,
                        shuffle = shuffle,
                        drop_last = getattr(self.args, "dataloader_drop_last", False),
                    )
            if not shuffle_dataset and dataset is not None:
                try:
                    from torch.utils.data import SequentialSampler
                except ImportError:
                    return sampler
                return SequentialSampler(dataset)
            return sampler

        return wrapper

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = ContextParallelManager(
            ContextParallelSettings.from_args(getattr(self, "args", None))
        )
        accelerator = getattr(self, "accelerator", None)
        manager = getattr(self, "_context_parallel_manager", None)
        mesh = getattr(manager, "accelerate_mesh", None) if manager else None
        existing_mesh = (
            getattr(accelerator, "torch_device_mesh", None)
            if accelerator is not None
            else None
        )
        if (
            accelerator is not None
            and mesh is not None
            and (
                existing_mesh is None
                or "cp" not in getattr(existing_mesh, "mesh_dim_names", ())
            )
        ):
            setattr(accelerator.state, "device_mesh", mesh)

        # When using pure context parallelism (dp_world_size=1), disable DDP
        # to avoid gradient checkpointing compatibility issues. DDP is not needed
        # when there's no data parallelism anyway.
        if manager and manager.enabled and manager.data_parallel_world_size == 1:
            try:
                from transformers.training_args import ParallelMode

                args = getattr(self, "args", None)
                if (
                    args is not None
                    and getattr(args, "parallel_mode", None) == ParallelMode.DISTRIBUTED
                ):
                    args.parallel_mode = ParallelMode.NOT_DISTRIBUTED
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(
                            "Context parallelism: disabled DDP for pure CP mode "
                            "(dp_world_size=1, no data parallelism needed)."
                        )
            except ImportError:
                pass

        # Enable sync_each_batch when using CP with gradient accumulation and DDP.
        # This keeps the computation graph constant for DDP + static_graph mode.
        if (
            manager
            and manager.enabled
            and manager.data_parallel_world_size > 1  # Only needed with actual DP
            and accelerator is not None
            and hasattr(accelerator, "gradient_state")
        ):
            grad_accum_steps = getattr(
                getattr(self, "args", None), "gradient_accumulation_steps", 1
            )
            if grad_accum_steps > 1:
                accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        "Context parallelism: enabled sync_each_batch for gradient "
                        "accumulation compatibility (syncing gradients every batch)."
                    )

    def _maybe_enable_ddp_static_graph(trainer):
        ddp_model = getattr(trainer, "model_wrapped", None)
        enable_fn = getattr(ddp_model, "_set_static_graph", None)
        if not callable(enable_fn):
            return
        if getattr(ddp_model, "_unsloth_context_parallel_static_graph", False):
            return
        try:
            enable_fn()
            setattr(ddp_model, "_unsloth_context_parallel_static_graph", True)
            if int(os.environ.get("RANK", "0")) == 0:
                print("Enabled DDP static-graph mode for context parallelism.")
        except Exception:
            pass

    @functools.wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if (
            manager
            and os.environ.get("UNSLOTH_CP_DUMP_BATCH") == "1"
            and isinstance(inputs.get("input_ids"), torch.Tensor)
        ):
            manager.remember_raw_input_ids(inputs["input_ids"])
        if _cp_debug_enabled():
            ids_pre = inputs.get("input_ids")
            preview_pre = (
                ids_pre[0][: min(ids_pre.shape[-1], 16)].tolist()
                if isinstance(ids_pre, torch.Tensor)
                else None
            )
            _cp_debug(
                f"[CP-DEBUG] pre-apply input_ids shape={getattr(ids_pre, 'shape', None)} preview={preview_pre}"
            )
            if os.environ.get("UNSLOTH_CP_DUMP_BATCH") == "1":
                rank_pre = dist.get_rank() if dist.is_initialized() else 0
                gathered = inputs.get("input_ids")
                if isinstance(gathered, torch.Tensor):
                    tokens = gathered.detach().cpu().flatten().tolist()
                    chunk = 64
                    for start in range(0, len(tokens), chunk):
                        end = min(start + chunk, len(tokens))
                        _cp_debug(
                            f"[CP-DEBUG][focus] raw-input_ids rank={rank_pre} idx={start}:{end} tokens={tokens[start:end]}"
                        )
        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            if manager:
                override = manager.consume_num_items_override()
                if override is not None:
                    kwargs["num_items_in_batch"] = override
            rank = 0
            if dist.is_initialized():
                rank = dist.get_rank()
            if _cp_debug_enabled():
                ids = inputs.get("input_ids")
                preview = (
                    ids[0][: min(ids.shape[-1], 16)].tolist()
                    if isinstance(ids, torch.Tensor)
                    else None
                )
                _cp_debug(
                    f"[CP-DEBUG][rank={rank}] before loss input_ids shape={getattr(ids, 'shape', None)} preview={preview}"
                )

            # For context parallelism with shift_labels, compute loss externally
            # to handle load-balanced token distribution correctly
            shift_labels = inputs.get("shift_labels")
            use_external_loss = (
                manager and manager.enabled and isinstance(shift_labels, torch.Tensor)
            )

            if use_external_loss:
                # Remove labels so model doesn't compute loss internally
                saved_labels = inputs.pop("labels", None)
                # Also remove shift_labels from inputs (model doesn't need it)
                inputs.pop("shift_labels", None)

                # Get model outputs (logits only, no loss)
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Compute loss using pre-shifted labels
                # No additional shifting needed - shift_labels already contains
                # the correct "next token" for each position after sharding
                from torch.nn import CrossEntropyLoss

                loss_fct = CrossEntropyLoss()
                # Flatten for loss computation
                vocab_size = logits.size(-1)
                loss = loss_fct(
                    logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                )

                if _cp_debug_enabled():
                    _cp_debug(
                        f"[CP-DEBUG][focus][rank={rank}] external loss: logits shape={logits.shape} "
                        f"shift_labels shape={shift_labels.shape} loss={loss.item()}"
                    )

                # Restore labels for reduce_loss token counting
                if saved_labels is not None:
                    inputs["labels"] = saved_labels

                if return_outputs:
                    loss = (loss, outputs)
            else:
                loss = original_compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs = return_outputs,
                    **kwargs,
                )
        if _cp_debug_enabled():
            _cp_debug(f"[CP-DEBUG][rank={rank}] raw loss={loss}")
        if manager:
            loss = manager.reduce_loss(loss, inputs)
        if _cp_debug_enabled():
            _cp_debug(f"[CP-DEBUG][rank={rank}] reduced loss={loss}")
        return loss

    @functools.wraps(original_prediction_step)
    def patched_prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
        **kwargs,
    ):
        manager = getattr(self, "_context_parallel_manager", None)
        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            return original_prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys,
                **kwargs,
            )

    def _maybe_enable_sync_each_batch(trainer):
        """Enable sync_each_batch at runtime if gradient checkpointing is detected."""
        if getattr(trainer, "_sync_each_batch_checked", False):
            return
        setattr(trainer, "_sync_each_batch_checked", True)

        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is None or not hasattr(accelerator, "gradient_state"):
            return

        # Check if already enabled
        if accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False):
            return

        model = getattr(trainer, "model", None)
        is_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        grad_accum_steps = getattr(trainer.args, "gradient_accumulation_steps", 1)

        if is_checkpointing and grad_accum_steps > 1:
            accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "Context parallelism: enabled sync_each_batch for gradient "
                    "checkpointing + gradient accumulation compatibility."
                )

    @functools.wraps(original_training_step)
    def patched_training_step(self, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        original_n_gpu = getattr(self.args, "n_gpu", 1)
        if manager:
            setattr(self.args, "_n_gpu", manager.data_parallel_world_size)
            _maybe_enable_sync_each_batch(self)
            # Note: static_graph is disabled because it causes internal PyTorch
            # assertion errors with unsloth's gradient checkpointing.
            # sync_each_batch keeps the graph constant instead.
        try:
            loss = original_training_step(self, *args, **kwargs)
        finally:
            if manager:
                setattr(self.args, "_n_gpu", original_n_gpu)
        report_loss = manager.consume_report_loss() if manager else None
        if manager and report_loss is not None:
            self._context_parallel_last_loss = (
                report_loss.detach().item()
                if torch.is_tensor(report_loss)
                else float(report_loss)
            )
            tokens = manager.consume_report_tokens()
            if tokens is not None:
                self._context_parallel_last_tokens = (
                    tokens.detach().item() if torch.is_tensor(tokens) else float(tokens)
                )
        if report_loss is not None:
            return report_loss
        return loss

    @functools.wraps(original_log)
    def patched_log(self, logs, start_time = None):
        manager = getattr(self, "_context_parallel_manager", None)
        if (
            manager
            and logs is not None
            and "loss" in logs
            and hasattr(self, "_context_parallel_last_loss")
        ):
            logs = dict(logs)
            tokens = getattr(self, "_context_parallel_last_tokens", None)
            if tokens is not None and tokens > 0:
                logs["loss"] = self._context_parallel_last_loss
                delattr(self, "_context_parallel_last_tokens")
            else:
                logs["loss"] = self._context_parallel_last_loss
            delattr(self, "_context_parallel_last_loss")
        return original_log(self, logs, start_time)

    def enable_context_parallel(self, **kwargs):
        settings = ContextParallelSettings(**kwargs)
        self._context_parallel_manager = ContextParallelManager(settings)

    trainer_cls.__init__ = patched_init
    trainer_cls.compute_loss = patched_compute_loss
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.training_step = patched_training_step
    trainer_cls.log = patched_log
    trainer_cls.enable_context_parallel = enable_context_parallel
    trainer_cls.__unsloth_context_parallel__ = True
    if original_get_train_sampler is not None:
        trainer_cls._get_train_sampler = _patch_train_sampler(
            original_get_train_sampler
        )


def _cp_debug_enabled() -> bool:
    return bool(os.environ.get("UNSLOTH_CP_DEBUG"))


def _cp_debug(msg: str) -> None:
    if not _cp_debug_enabled():
        return
    mode = os.environ.get("UNSLOTH_CP_DEBUG_MODE", "off").lower()
    if mode == "off":
        return
    if mode == "focused" and "[CP-DEBUG][focus]" not in msg:
        return
    print(msg, flush = True)
