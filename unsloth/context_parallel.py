from __future__ import annotations

import contextlib
import functools
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple
import sys

import torch
from packaging.version import Version
import torch.distributed as dist

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor import DeviceMesh
except (ImportError, AttributeError):  # pragma: no cover - handled at runtime
    context_parallel = None  # type: ignore[assignment]
    DeviceMesh = None  # type: ignore[assignment]

from .device_type import DEVICE_TYPE_TORCH


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
        default_factory = lambda: ("input_ids", "attention_mask", "labels"),
        metadata = {
            "help": (
                "Names inside the Trainer input batch that should be sharded "
                "across the context dimension."
            )
        },
    )
    no_restore_buffer_names: Tuple[str, ...] = field(
        default_factory = tuple,
        metadata = {
            "help": (
                "Subset of `buffer_names` that do not need to be restored after the "
                "context parallel region exits."
            )
        },
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
                ("input_ids", "attention_mask", "labels"),
            )
        )
        no_restore = _as_tuple(_get("context_parallel_no_restore", ()))
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
        self._no_restore_lookup = set(settings.no_restore_buffer_names)
        self._cached_num_items: Optional[torch.Tensor | float | int] = None
        self._report_loss: Optional[torch.Tensor] = None
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

    @contextlib.contextmanager
    def apply(self, inputs: dict[str, torch.Tensor]) -> Iterator[None]:
        if not self.enabled or self._mesh is None:
            yield
            return
        self._adjust_num_items_in_batch(inputs)
        buffers, seq_dims, no_restore = self._collect_buffers(inputs)
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
            return None
        labels = self._rank_slice(inputs.get("labels"))
        if isinstance(labels, torch.Tensor):
            weight = labels.ne(-100).sum()
            if weight.item() > 0:
                return weight.to(device = reference.device, dtype = reference.dtype)
        attention_mask = self._rank_slice(inputs.get("attention_mask"))
        if isinstance(attention_mask, torch.Tensor):
            weight = attention_mask.sum()
            if weight.item() > 0:
                return weight.to(device = reference.device, dtype = reference.dtype)
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
        if self.settings.size > 1 and self._cp_group is not None:
            dist.all_reduce(local_tokens, op = dist.ReduceOp.SUM, group = self._cp_group)
            local_tokens = local_tokens / float(self.settings.size)
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

    def reduce_loss(self, loss, inputs):
        if not self.enabled or self.settings.size <= 1 or self._cp_group is None:
            return loss

        def _reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None or not torch.is_tensor(tensor):
                return tensor
            weight = self._loss_weight(inputs, tensor)
            if weight is None:
                dist.all_reduce(tensor, op = dist.ReduceOp.SUM, group = self._cp_group)
                avg = tensor / float(self.settings.size)
            else:
                scaled = tensor * weight
                dist.all_reduce(scaled, op = dist.ReduceOp.SUM, group = self._cp_group)
                dist.all_reduce(weight, op = dist.ReduceOp.SUM, group = self._cp_group)
                eps = torch.finfo(weight.dtype).eps
                avg = scaled / torch.clamp(weight, min = eps)
            self._set_report_loss(avg)
            return avg * self.settings.size

        if isinstance(loss, tuple):
            if not loss:
                return loss
            reduced = _reduce_tensor(loss[0])
            return (reduced, *loss[1:])
        elif torch.is_tensor(loss):
            return _reduce_tensor(loss)
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
            default_factory = lambda: ("input_ids", "attention_mask", "labels"),
            metadata = {
                "help": (
                    "Batch keys whose tensors should be sharded across context parallel ranks. "
                    "Pass a tuple/list of keys."
                )
            },
        )
        context_parallel_no_restore: Tuple[str, ...] = field(
            default_factory = tuple,
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
    original_get_train_sampler = getattr(trainer_cls, "_get_train_sampler", None)
    original_training_step = trainer_cls.training_step
    original_log = trainer_cls.log

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

    @functools.wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            if manager:
                override = manager.consume_num_items_override()
                if override is not None:
                    kwargs["num_items_in_batch"] = override
            loss = original_compute_loss(
                self,
                model,
                inputs,
                return_outputs = return_outputs,
                **kwargs,
            )
        if manager:
            loss = manager.reduce_loss(loss, inputs)
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

    @functools.wraps(original_training_step)
    def patched_training_step(self, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        original_n_gpu = getattr(self.args, "n_gpu", 1)
        if manager:
            setattr(self.args, "_n_gpu", manager.data_parallel_world_size)
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
