from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor import DeviceMesh
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None

from .device_type import DEVICE_TYPE_TORCH
from .utils.packing import mask_packed_sequence_boundaries

_ACTIVE_MANAGER: contextvars.ContextVar[Optional["ContextParallelManager"]] = (
    contextvars.ContextVar("unsloth_active_cp_manager", default = None)
)


def get_active_context_parallel_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER.get()


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


def _attach_context_parallel_attention_hooks(model: torch.nn.Module) -> list:
    """
    Attach forward_pre_hooks to self_attn modules to ensure correct attention
    behavior during context parallelism with load balancing.

    This mirrors accelerate's approach: remove attention_mask and set is_causal=True
    to ensure ring attention works correctly with reordered tokens.

    Args:
        model: The model to attach hooks to

    Returns:
        List of hook handles that can be used to remove the hooks later
    """
    handles = []

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        # Remove attention_mask and set is_causal=True
        # This ensures ring attention uses causal masking correctly
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
        if "is_causal" in module_kwargs or hasattr(_module, "is_causal"):
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    # Find all self_attn modules - they may be nested in PEFT wrappers
    attn_modules = []
    for name, module in model.named_modules():
        # Attach to modules ending with self_attn (transformers convention)
        if name.endswith("self_attn"):
            attn_modules.append((name, module))

    for name, module in attn_modules:
        handle = module.register_forward_pre_hook(
            _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
        )
        handles.append(handle)

    return handles


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
        self._device_mesh: Optional[DeviceMesh] = None
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._cp_rank_index: int = 0
        self._dp_world_size: int = 1
        self._world_size: int = 1
        self._no_restore_lookup = set(settings.no_restore_buffer_names)
        self._cached_num_items: Optional[torch.Tensor | float | int] = None
        self._report_loss: Optional[torch.Tensor] = None
        self._report_tokens: Optional[torch.Tensor] = None
        self._attention_hook_handles: list = []
        self._verify_environment()
        if self.enabled:
            self._mesh = self._build_mesh()
            self._device_mesh = self._build_device_mesh()

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        """
        Attach hooks to self_attn modules to ensure correct attention behavior
        during context parallelism with load balancing.

        This should be called once after the model is available.
        """
        if not self.enabled:
            return
        if self._attention_hook_handles:
            # Already attached
            return
        self._attention_hook_handles = _attach_context_parallel_attention_hooks(model)

    def _verify_environment(self) -> None:
        if not self.enabled:
            if self.settings.size > 1 and context_parallel is None:
                warnings.warn(
                    "Context parallelism requested but PyTorch >= 2.7 is required.",
                    stacklevel = 2,
                )
            return
        self._world_size = dist.get_world_size()

    def _build_mesh(self) -> DeviceMesh:
        rank = torch.distributed.get_rank()
        group_index = rank // self.settings.size
        start = group_index * self.settings.size
        cp_ranks = torch.arange(start, start + self.settings.size, dtype = torch.int64)
        mesh = DeviceMesh(DEVICE_TYPE_TORCH, cp_ranks)
        self._cp_group = mesh.get_group()
        self._cp_rank_index = int(rank - start)
        return mesh

    def _build_device_mesh(self) -> DeviceMesh:
        self._dp_world_size = self._world_size // self.settings.size
        mesh = torch.arange(self._world_size, dtype = torch.int64).reshape(
            self._dp_world_size, self.settings.size
        )
        return DeviceMesh(
            DEVICE_TYPE_TORCH, mesh, mesh_dim_names = ("dp_replicate", "cp")
        )

    def __bool__(self) -> bool:
        return self.enabled

    @property
    def device_mesh(self) -> Optional[DeviceMesh]:
        return self._device_mesh

    @property
    def data_parallel_world_size(self) -> int:
        return self._dp_world_size

    @property
    def cp_rank_index(self) -> int:
        return self._cp_rank_index

    def data_parallel_rank(self) -> int:
        if not self.enabled:
            return dist.get_rank() if dist.is_initialized() else 0
        return dist.get_rank() // self.settings.size

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
        if "position_ids" in inputs:
            return
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return
        seq_len = input_ids.size(self.settings.seq_dim)
        positions = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)
        inputs["position_ids"] = positions.unsqueeze(0).expand(input_ids.size(0), -1)

    def _ensure_shift_labels(self, inputs: dict[str, torch.Tensor]) -> None:
        """Pre-shift labels globally before sharding for correct next-token prediction."""
        if "shift_labels" in inputs:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        # Pad with -100, then take [1:] to get shifted labels
        shift_labels = F.pad(labels, (0, 1), value = -100)[:, 1:].contiguous()
        packed_seq_lengths = inputs.get("packed_seq_lengths")
        if packed_seq_lengths is not None:
            mask_packed_sequence_boundaries(shift_labels, packed_seq_lengths)
        inputs["shift_labels"] = shift_labels

    @contextlib.contextmanager
    def apply(self, inputs: dict[str, torch.Tensor]) -> Iterator[None]:
        """Wrap training step to shard buffers and patch SDPA for ring attention."""
        if not self.enabled:
            yield
            return
        token = _ACTIVE_MANAGER.set(self)
        self._adjust_num_items_in_batch(inputs)
        self._ensure_position_ids(inputs)
        self._ensure_shift_labels(inputs)
        buffers, seq_dims, no_restore = self._collect_buffers(inputs)
        with context_parallel(
            self._mesh,
            buffers = buffers,
            buffer_seq_dims = seq_dims,
            no_restore_buffers = no_restore,
        ):
            yield
        _ACTIVE_MANAGER.reset(token)

    def _adjust_num_items_in_batch(self, inputs: dict[str, torch.Tensor]) -> None:
        """Cache and remove num_items_in_batch so model uses local token count."""
        self._cached_num_items = None
        value = inputs.pop("num_items_in_batch", None)
        if value is not None:
            self._cached_num_items = value.item() if torch.is_tensor(value) else value

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

        # Extract tensor from tuple if needed
        is_tuple = isinstance(loss, tuple)
        if is_tuple:
            if not loss:
                return loss
            tensor, rest = loss[0], loss[1:]
        elif torch.is_tensor(loss):
            tensor = loss
        else:
            return loss

        # Count valid tokens from sharded shift_labels/labels
        shift_labels = inputs.get("shift_labels")
        if isinstance(shift_labels, torch.Tensor):
            local_tokens = (
                shift_labels.ne(-100).sum().to(device = tensor.device, dtype = tensor.dtype)
            )
        else:
            labels = inputs.get("labels")
            if isinstance(labels, torch.Tensor):
                local_tokens = (
                    labels.ne(-100).sum().to(device = tensor.device, dtype = tensor.dtype)
                )
            else:
                local_tokens = torch.tensor(
                    1.0, dtype = tensor.dtype, device = tensor.device
                )

        # Get global token count
        global_tokens = local_tokens.detach().clone()
        dist.all_reduce(global_tokens, op = dist.ReduceOp.SUM, group = self._cp_group)

        # Weight fraction for gradient scaling
        num_items = self._cached_num_items
        if num_items is None or num_items <= 0:
            num_items = global_tokens.item()
        weight = local_tokens.detach() / num_items
        weighted_loss = tensor * weight

        # Reduce for reporting
        global_loss = weighted_loss.detach().clone()
        dist.all_reduce(global_loss, op = dist.ReduceOp.SUM, group = self._cp_group)

        self._set_report_loss(global_loss)
        self._set_report_tokens(global_tokens)

        return (weighted_loss, *rest) if is_tuple else weighted_loss


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
                    return DistributedSampler(
                        dataset,
                        num_replicas = dp_world,
                        rank = dp_rank,
                        drop_last = getattr(self.args, "dataloader_drop_last", False),
                    )
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
        if manager and manager.enabled:
            print(
                f"Unsloth: Context parallelism enabled with size={manager.settings.size}"
            )
        mesh = getattr(manager, "device_mesh", None) if manager else None
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
                from accelerate.utils import DistributedType

                args = getattr(self, "args", None)
                distributed_state = getattr(args, "distributed_state", None)
                if (
                    distributed_state is not None
                    and distributed_state.distributed_type == DistributedType.MULTI_GPU
                ):
                    distributed_state.distributed_type = DistributedType.NO
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

        # Attach attention hooks for proper ring attention behavior with load balancing.
        # This ensures attention_mask is removed and is_causal=True for all self_attn calls.
        if manager and manager.enabled:
            model = getattr(self, "model", None)
            if model is not None:
                manager.attach_attention_hooks(model)

    @functools.wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)

        # Cache num_items_in_batch from kwargs if present (apply() handles inputs)
        if manager and manager.enabled:
            num_items_val = kwargs.pop("num_items_in_batch", None)
            if num_items_val is not None and manager._cached_num_items is None:
                manager._cached_num_items = (
                    num_items_val.item()
                    if torch.is_tensor(num_items_val)
                    else num_items_val
                )

        # For context parallelism with shift_labels, prefer letting the model
        # handle the pre-shifted targets when it advertises support. Otherwise
        # fall back to an external loss that consumes the sharded tensors.
        shift_labels = inputs.get("shift_labels")
        use_cp_shift_labels = (
            manager and manager.enabled and isinstance(shift_labels, torch.Tensor)
        )
        model_supports_shift_labels = bool(
            getattr(
                model,
                "_unsloth_supports_context_parallel_shift_labels",
                False,
            )
        )

        if use_cp_shift_labels and not model_supports_shift_labels:
            # Remove labels so model doesn't compute loss internally
            saved_labels = inputs.pop("labels", None)
            # Also remove shift_labels from inputs (model doesn't expect it)
            local_shift_labels = inputs.pop("shift_labels", None)

            # Get model outputs (logits only, no loss)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Compute loss using pre-shifted labels
            # No additional shifting needed - shift_labels already contains
            # the correct "next token" for each position after sharding
            from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

            loss = fast_cross_entropy_loss(
                logits = logits,
                labels = local_shift_labels,
            )

            # Restore labels for reduce_loss token counting
            if saved_labels is not None:
                inputs["labels"] = saved_labels
            if local_shift_labels is not None:
                inputs["shift_labels"] = local_shift_labels

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

        if manager and manager.enabled:
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

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        original_n_gpu = getattr(self.args, "n_gpu", 1)
        if manager:
            setattr(self.args, "_n_gpu", manager.data_parallel_world_size)
            _maybe_enable_sync_each_batch(self)
            # Attach attention hooks if not already done (model may not be ready at init)
            if manager.enabled and not manager._attention_hook_handles:
                m = getattr(self, "model", None)
                if m is not None:
                    manager.attach_attention_hooks(m)

        # Wrap entire training step (forward + backward) in context_parallel
        # This keeps SDPA patched and buffers sharded throughout, including
        # during gradient checkpoint recomputation in backward pass.
        cp_context = (
            manager.apply(inputs)
            if manager and manager.enabled
            else contextlib.nullcontext()
        )
        try:
            with cp_context:
                loss = original_training_step(self, model, inputs, *args, **kwargs)
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
        # Don't override the loss - let the Trainer accumulate losses correctly.
        # The per-batch losses returned from training_step are now correct
        # (normalized by total GA tokens), so the Trainer's accumulation will
        # produce the same result as CP=1.
        manager = getattr(self, "_context_parallel_manager", None)
        if manager and hasattr(self, "_context_parallel_last_loss"):
            delattr(self, "_context_parallel_last_loss")
        if manager and hasattr(self, "_context_parallel_last_tokens"):
            delattr(self, "_context_parallel_last_tokens")
        return original_log(self, logs, start_time)

    trainer_cls.__init__ = patched_init
    trainer_cls.compute_loss = patched_compute_loss
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.training_step = patched_training_step
    trainer_cls.log = patched_log
    trainer_cls.__unsloth_context_parallel__ = True
    if original_get_train_sampler is not None:
        trainer_cls._get_train_sampler = _patch_train_sampler(
            original_get_train_sampler
        )
