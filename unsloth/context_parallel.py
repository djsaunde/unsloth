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

    try:
        # Allow overriding load balance via env (default True matches PyTorch)
        from torch.distributed.tensor.experimental._attention import _cp_options

        env_lb = os.environ.get("UNSLOTH_CP_LB")
        if env_lb is not None:
            _cp_options.enable_load_balance = env_lb not in ("0", "false", "False")
    except Exception:
        pass

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


def _run_cp_sanity_check(model, manager) -> None:
    """
    One-time sanity check to verify model outputs are consistent across CP ranks.
    Run with UNSLOTH_CP_SANITY_CHECK=1 to enable.
    """
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create identical test input on all ranks
    torch.manual_seed(42)
    seq_len = 32 * manager.settings.size  # Ensure divisible by CP size
    test_input = torch.randint(100, 1000, (1, seq_len), device = "cuda")

    # Broadcast from rank 0 to ensure identical input
    dist.broadcast(test_input, src = 0)

    print(
        f"[CP-SANITY][rank={rank}] Running sanity check with input shape {test_input.shape}"
    )

    # Run model WITHOUT context parallel to get reference output
    # Force SDPA path by temporarily disabling Flash Attention
    # This ensures reference and CP runs use the same attention implementation
    import unsloth.models.llama as llama_module

    original_has_flash = getattr(llama_module, "HAS_FLASH_ATTENTION", False)
    llama_module.HAS_FLASH_ATTENTION = False
    print(f"[CP-SANITY][rank={rank}] Disabled Flash Attention for fair comparison")

    # Create position_ids for reference too (ensures same RoPE code path)
    full_pos = torch.arange(seq_len, device = "cuda").unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # Capture layer outputs for debugging
        layer_outputs_ref = {}

        def make_hook(name, storage):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                storage[name] = out.detach().clone()

            return hook

        def make_pre_hook(name, storage):
            def hook(module, input):
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                else:
                    inp = input
                if isinstance(inp, torch.Tensor):
                    storage[name] = inp.detach().clone()

            return hook

        # Register hooks on first few layers
        hooks = []
        # Navigate to the inner model (handles PEFT wrapping, etc.)
        base_model = model
        print(f"[CP-SANITY][rank={rank}] model type: {type(model).__name__}")
        if hasattr(model, "model"):
            print(
                f"[CP-SANITY][rank={rank}] model.model type: {type(model.model).__name__}"
            )
            if hasattr(model.model, "model"):
                print(
                    f"[CP-SANITY][rank={rank}] model.model.model type: {type(model.model.model).__name__}"
                )
                base_model = model.model.model
            elif hasattr(model.model, "layers"):
                base_model = model.model
        print(f"[CP-SANITY][rank={rank}] base_model type: {type(base_model).__name__}")
        print(
            f"[CP-SANITY][rank={rank}] has embed_tokens: {hasattr(base_model, 'embed_tokens')}"
        )
        print(f"[CP-SANITY][rank={rank}] has layers: {hasattr(base_model, 'layers')}")
        if hasattr(base_model, "embed_tokens"):
            hooks.append(
                base_model.embed_tokens.register_forward_hook(
                    make_hook("embed", layer_outputs_ref)
                )
            )
        if hasattr(base_model, "layers") and len(base_model.layers) > 0:
            hooks.append(
                base_model.layers[0].register_forward_hook(
                    make_hook("layer0", layer_outputs_ref)
                )
            )
            # Also hook the attention module inside layer 0
            if hasattr(base_model.layers[0], "self_attn"):
                hooks.append(
                    base_model.layers[0].self_attn.register_forward_pre_hook(
                        make_pre_hook("layer0_attn_in", layer_outputs_ref)
                    )
                )
                hooks.append(
                    base_model.layers[0].self_attn.register_forward_hook(
                        make_hook("layer0_attn", layer_outputs_ref)
                    )
                )
            # And the input layernorm
            if hasattr(base_model.layers[0], "input_layernorm"):
                hooks.append(
                    base_model.layers[0].input_layernorm.register_forward_hook(
                        make_hook("layer0_ln", layer_outputs_ref)
                    )
                )
        print(f"[CP-SANITY][rank={rank}] Registered {len(hooks)} hooks")

        ref_output = model(test_input, position_ids = full_pos.clone())
        ref_logits = ref_output.logits
        ref_sum = ref_logits.sum().item()
        ref_first = ref_logits[0, 0, :5].tolist()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Log reference layer checksums
        print(
            f"[CP-SANITY][rank={rank}] layer_outputs_ref keys: {list(layer_outputs_ref.keys())}"
        )
        for name, tensor in layer_outputs_ref.items():
            checksum = tensor.float().sum().item()
            print(
                f"[CP-SANITY][rank={rank}] REF {name}: shape={tuple(tensor.shape)} checksum={checksum:.4f}"
            )

        print(
            f"[CP-SANITY][rank={rank}] Reference (no CP, SDPA): logits_sum={ref_sum:.4f} first_5={[f'{x:.4f}' for x in ref_first]}"
        )

        # Now run WITH context parallel
        # DON'T manually shard - let context_parallel do it!

        # Check if model is compiled (torch.compile can bypass TorchFunctionMode)
        is_compiled = hasattr(model, "_orig_mod") or hasattr(model, "__wrapped__")
        print(f"[CP-SANITY][rank={rank}] Model is_compiled={is_compiled}")

        # Try to detect if ring attention will be used
        try:
            from torch.distributed.tensor.experimental._attention import _cp_options

            print(
                f"[CP-SANITY][rank={rank}] CP options: enable_load_balance={_cp_options.enable_load_balance}"
            )
        except Exception as e:
            print(f"[CP-SANITY][rank={rank}] Could not check CP options: {e}")

        # Pass FULL tensors to context_parallel - it will shard them
        buffers = [test_input, full_pos]
        print(
            f"[CP-SANITY][rank={rank}] Passing full input shape={tuple(test_input.shape)} to context_parallel"
        )

        # Set the global seq_len so position_ids adjustment in the model works correctly
        manager._last_global_seq_len = seq_len

        with context_parallel(
            manager._mesh,
            buffers = buffers,
            buffer_seq_dims = [1, 1],
            no_restore_buffers = set(buffers),
        ):
            # IMPORTANT: Also set unsloth's _ACTIVE_MANAGER so cp_active is True
            # This ensures unsloth uses SDPA (which gets intercepted by ring attention)
            # instead of Flash Attention (which would bypass ring attention).
            with manager.replay_context():
                # Check if SDPA function is patched
                import torch.nn.functional as _F

                _sdpa_fn = _F.scaled_dot_product_attention
                print(
                    f"[CP-SANITY][rank={rank}] SDPA function: {_sdpa_fn.__name__ if hasattr(_sdpa_fn, '__name__') else type(_sdpa_fn)}"
                )
                print(
                    f"[CP-SANITY][rank={rank}] SDPA function module: {_sdpa_fn.__module__ if hasattr(_sdpa_fn, '__module__') else 'N/A'}"
                )
                # After entering context, buffers are sharded in-place
                print(
                    f"[CP-SANITY][rank={rank}] Inside context_parallel, input now shape={tuple(test_input.shape)}"
                )
                print(
                    f"[CP-SANITY][rank={rank}] position_ids shape={tuple(full_pos.shape)} values={full_pos[0, :8].tolist()}...{full_pos[0, -4:].tolist()}"
                )
                print(
                    f"[CP-SANITY][rank={rank}] input_ids first 8: {test_input[0, :8].tolist()}"
                )
                print(
                    f"[CP-SANITY][rank={rank}] cp_active check: manager.enabled={manager.enabled}"
                )

                # Register hooks for CP run
                layer_outputs_cp = {}
                hooks_cp = []
                if hasattr(base_model, "embed_tokens"):
                    hooks_cp.append(
                        base_model.embed_tokens.register_forward_hook(
                            make_hook("embed", layer_outputs_cp)
                        )
                    )
                if hasattr(base_model, "layers") and len(base_model.layers) > 0:
                    hooks_cp.append(
                        base_model.layers[0].register_forward_hook(
                            make_hook("layer0", layer_outputs_cp)
                        )
                    )
                    # Also hook the attention module inside layer 0
                    if hasattr(base_model.layers[0], "self_attn"):
                        hooks_cp.append(
                            base_model.layers[0].self_attn.register_forward_pre_hook(
                                make_pre_hook("layer0_attn_in", layer_outputs_cp)
                            )
                        )
                        hooks_cp.append(
                            base_model.layers[0].self_attn.register_forward_hook(
                                make_hook("layer0_attn", layer_outputs_cp)
                            )
                        )
                    # And the input layernorm
                    if hasattr(base_model.layers[0], "input_layernorm"):
                        hooks_cp.append(
                            base_model.layers[0].input_layernorm.register_forward_hook(
                                make_hook("layer0_ln", layer_outputs_cp)
                            )
                        )

                cp_output = model(input_ids = test_input, position_ids = full_pos)
                cp_logits = cp_output.logits
                local_sum = cp_logits.sum()

                # Remove hooks
                for h in hooks_cp:
                    h.remove()

                # Log CP layer checksums and compare with reference
                for name, tensor in layer_outputs_cp.items():
                    local_checksum = tensor.float().sum()
                    # Gather checksums from all ranks
                    all_checksums = [
                        torch.zeros_like(local_checksum) for _ in range(world_size)
                    ]
                    dist.all_gather(all_checksums, local_checksum)
                    total_checksum = sum(c.item() for c in all_checksums)
                    ref_checksum = (
                        layer_outputs_ref[name].float().sum().item()
                        if name in layer_outputs_ref
                        else 0
                    )
                    diff = abs(total_checksum - ref_checksum)
                    print(
                        f"[CP-SANITY][rank={rank}] CP {name}: local_shape={tuple(tensor.shape)} total_checksum={total_checksum:.4f} ref={ref_checksum:.4f} diff={diff:.4f}"
                    )
                print(
                    f"[CP-SANITY][rank={rank}] Model returned, logits shape={tuple(cp_logits.shape)}"
                )

                # Gather sums from all ranks
                all_sums = [torch.zeros_like(local_sum) for _ in range(world_size)]
                dist.all_gather(all_sums, local_sum)
                cp_total_sum = sum(s.item() for s in all_sums)

                print(
                    f"[CP-SANITY][rank={rank}] CP local logits_sum={local_sum.item():.4f}"
                )
                print(
                    f"[CP-SANITY][rank={rank}] CP total logits_sum={cp_total_sum:.4f} (ref={ref_sum:.4f})"
                )

                # Check if they match
                diff = abs(cp_total_sum - ref_sum)
                if diff < 0.01:
                    print(
                        f"[CP-SANITY][rank={rank}] ✓ PASS: Outputs match (diff={diff:.6f})"
                    )
                else:
                    print(
                        f"[CP-SANITY][rank={rank}] ✗ FAIL: Outputs differ (diff={diff:.6f})"
                    )
                    print(
                        f"[CP-SANITY][rank={rank}] This indicates ring attention may not be working correctly"
                    )

    # Restore Flash Attention flag
    llama_module.HAS_FLASH_ATTENTION = original_has_flash

    model.train()
    dist.barrier()


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

    if handles and int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Context parallelism: attached attention hooks to {len(handles)} self_attn modules"
        )

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
        self._accelerate_mesh: Optional[DeviceMesh] = None
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._cp_rank_index: int = 0
        self._dp_world_size: int = 1
        self._world_size: int = 1
        self._no_restore_lookup = set(settings.no_restore_buffer_names)
        self._cached_num_items: Optional[torch.Tensor | float | int] = None
        self._cached_ga_steps: int = 1
        self._report_loss: Optional[torch.Tensor] = None
        self._report_tokens: Optional[torch.Tensor] = None
        self._last_global_seq_len: Optional[int] = None
        self._attention_hook_handles: list = []
        self._verify_environment()
        if self.enabled:
            self._mesh = self._build_mesh()
            self._accelerate_mesh = self._build_accelerate_mesh()

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

    def remove_attention_hooks(self) -> None:
        """Remove previously attached attention hooks."""
        for handle in self._attention_hook_handles:
            handle.remove()
        self._attention_hook_handles = []

    def enter_sdpa_patch(self) -> None:
        """
        Manually enter SDPA patching for context parallel.

        This patches F.scaled_dot_product_attention to use ring attention
        and enables the CP dispatcher. Call exit_sdpa_patch() when done.

        This is needed to keep SDPA patched during backward pass with
        gradient checkpointing, which re-runs forward outside the normal
        context_parallel context manager.

        Also initializes deferred buffer restoration - buffers sharded in
        apply() will be restored in exit_sdpa_patch() instead of immediately,
        ensuring buffers stay sharded during backward pass.
        """
        if not self.enabled or self._mesh is None:
            return
        if getattr(self, "_sdpa_patch_active", False):
            return  # Already patched

        # Initialize deferred buffer restoration list
        # This will be populated by apply() and processed in exit_sdpa_patch()
        self._pending_buffer_restores: list[tuple[torch.Tensor, torch.Tensor]] = []

        try:
            from torch.distributed.tensor.experimental._attention import (
                _distribute_function,
                _enable_cp_dispatcher,
            )
            from torch.distributed.tensor import DTensor, Shard
            import itertools

            seq_dim = 2  # Standard seq dimension for attention
            mesh = self._mesh

            def attention_input_fn(mesh, *args, **kwargs):
                placement = [Shard(seq_dim)]
                all_args = []
                for arg in itertools.chain(args, kwargs.values()):
                    if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                        arg = DTensor.from_local(arg, mesh, placement, run_check = False)
                    all_args.append(arg)
                new_args = tuple(all_args[0 : len(args)])
                new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
                return new_args, new_kwargs

            def attention_output_fn(mesh, outputs):
                new_outputs = []
                for output in (
                    [outputs] if isinstance(outputs, torch.Tensor) else outputs
                ):
                    output = (
                        output.to_local() if isinstance(output, DTensor) else output
                    )
                    new_outputs.append(output)
                if isinstance(outputs, torch.Tensor):
                    return new_outputs[0]
                return tuple(new_outputs)

            # Patch SDPA
            _distribute_function(
                F.scaled_dot_product_attention,
                F,
                mesh,
                attention_input_fn,
                attention_output_fn,
            )

            # Enable CP dispatcher
            self._cp_dispatcher_ctx = _enable_cp_dispatcher()
            self._cp_dispatcher_ctx.__enter__()

            self._sdpa_patch_active = True

        except Exception:
            pass

    def exit_sdpa_patch(self) -> None:
        """Exit SDPA patching that was entered with enter_sdpa_patch().

        Also restores any buffers that were deferred during apply().
        This ensures buffers stay sharded throughout both forward and backward
        passes when gradient checkpointing is enabled.
        """
        if not getattr(self, "_sdpa_patch_active", False):
            return

        try:
            from torch.distributed.tensor.experimental._attention import (
                _restore_function,
            )

            # Restore deferred buffers FIRST, before exiting CP context
            # This ensures backward pass (with gradient checkpointing) sees sharded buffers
            pending = getattr(self, "_pending_buffer_restores", [])
            if pending:
                for buffer, original in pending:
                    buffer.resize_(original.shape)
                    buffer.copy_(original)
                self._pending_buffer_restores = []

            # Exit CP dispatcher
            if (
                hasattr(self, "_cp_dispatcher_ctx")
                and self._cp_dispatcher_ctx is not None
            ):
                self._cp_dispatcher_ctx.__exit__(None, None, None)
                self._cp_dispatcher_ctx = None

            # Restore SDPA
            _restore_function(F.scaled_dot_product_attention, F)

            self._sdpa_patch_active = False

        except Exception:
            pass

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
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return
        if input_ids.ndim <= self.settings.seq_dim:
            return
        seq_len = input_ids.size(self.settings.seq_dim)
        if seq_len <= 0:
            return
        # Always set _last_global_seq_len for position_ids adjustment in model
        self._last_global_seq_len = seq_len
        # If position_ids already provided, don't synthesize new ones
        if isinstance(inputs.get("position_ids"), torch.Tensor):
            return
        device = input_ids.device
        dtype = torch.long
        base = torch.arange(seq_len, dtype = dtype, device = device)
        view_shape = [1] * input_ids.ndim
        view_shape[self.settings.seq_dim] = seq_len
        positions = base.view(view_shape).expand_as(input_ids)
        positions = positions.to(dtype = torch.long)
        inputs["position_ids"] = positions

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

    def _debug_validate_buffers(
        self,
        buffers: list[torch.Tensor],
    ) -> None:
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
            # Debug: verify load balance setting at runtime
            # If SDPA is already patched (via enter_sdpa_patch), do buffer sharding only
            # to avoid context_parallel's __exit__ from unpatching SDPA before backward
            sdpa_already_patched = getattr(self, "_sdpa_patch_active", False)

            if sdpa_already_patched:
                # Manual buffer sharding without SDPA patching
                try:
                    from torch.distributed.tensor.experimental._attention import (
                        _context_parallel_buffers,
                    )

                    # Shard buffers in-place
                    original_buffers = [
                        None if b in no_restore else b.clone() for b in buffers
                    ]
                    chunks = _context_parallel_buffers(self._mesh, buffers, seq_dims)
                    for buffer, chunk in zip(buffers, chunks):
                        chunk = chunk.clone()
                        buffer.resize_(chunk.shape)
                        buffer.copy_(chunk)

                    yield

                    # Defer buffer restoration to exit_sdpa_patch()
                    # This ensures buffers stay sharded during backward pass
                    # (critical for gradient checkpointing which re-runs forward)
                    for buffer, orig in zip(buffers, original_buffers):
                        if orig is not None:
                            self._pending_buffer_restores.append((buffer, orig))

                except ImportError:
                    # Fallback to context_parallel if imports fail
                    with context_parallel(
                        self._mesh,
                        buffers = buffers,
                        buffer_seq_dims = seq_dims,
                        no_restore_buffers = no_restore,
                    ):
                        yield
            else:
                # Normal path: use context_parallel which handles both SDPA patching and buffer sharding
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
        # Cache and remove num_items_in_batch so the model computes loss using LOCAL token count.
        # We'll use the cached value in reduce_loss for proper gradient accumulation support.
        #
        # With context parallelism, each rank has a shard of the sequence. If we pass
        # the GLOBAL num_items_in_batch, the model computes: loss = local_sum / global_count
        # Then reduce_loss tries to recover: scaled = loss * local_count, but this gives:
        #   (local_sum / global_count) * local_count ≠ local_sum
        #
        # By removing num_items_in_batch, the model computes: loss = local_sum / local_count
        # Then reduce_loss correctly recovers: scaled = loss * local_count = local_sum
        # The final reduction uses the CACHED num_items_in_batch (total across GA steps)
        # for proper gradient accumulation: global_sum / num_items_in_batch
        # Note: Don't reset _cached_num_items if already set (may have been cached from kwargs
        # in patched_compute_loss before this is called)
        if not self.enabled or "num_items_in_batch" not in inputs:
            return
        # Cache the global value for use in reduce_loss (only if not already cached)
        value = inputs.get("num_items_in_batch")
        if value is not None and self._cached_num_items is None:
            self._cached_num_items = value.item() if torch.is_tensor(value) else value
        # Remove so model uses local token count
        inputs.pop("num_items_in_batch", None)

    def consume_num_items_override(self):
        # Return None so the model uses local token count for loss computation.
        # Keep _cached_num_items intact - reduce_loss will use it for proper
        # gradient accumulation support.
        return None

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

        # Enhanced CP Loss Debug
        def _reduce_tensor(
            tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if tensor is None or not torch.is_tensor(tensor):
                zeros = torch.zeros(
                    (),
                    dtype = tensor.dtype if torch.is_tensor(tensor) else torch.float32,
                    device = tensor.device if torch.is_tensor(tensor) else None,
                )
                return tensor, zeros, zeros

            # Count valid tokens from sharded shift_labels (stays sharded due to no_restore)
            shift_labels = inputs.get("shift_labels")
            if isinstance(shift_labels, torch.Tensor):
                local_tokens = (
                    shift_labels.ne(-100)
                    .sum()
                    .to(device = tensor.device, dtype = tensor.dtype)
                )
            else:
                labels = inputs.get("labels")
                if isinstance(labels, torch.Tensor):
                    local_tokens = (
                        labels.ne(-100)
                        .sum()
                        .to(device = tensor.device, dtype = tensor.dtype)
                    )
                else:
                    local_tokens = torch.tensor(
                        1.0, dtype = tensor.dtype, device = tensor.device
                    )

            # Get global token count WITHOUT gradients
            global_tokens = local_tokens.detach().clone()
            dist.all_reduce(global_tokens, op = dist.ReduceOp.SUM, group = self._cp_group)

            # For backward: scale by local_tokens / num_items_in_batch (total across GA)
            # This matches CP=1 which divides by num_items_in_batch for each micro-batch.
            # For logging: use global_tokens (per-batch mean) to show meaningful loss.
            num_items = self._cached_num_items
            if num_items is None or num_items <= 0:
                num_items = global_tokens.item()  # fallback to per-batch tokens

            # Each rank backwards on: local_loss * (local_tokens / num_items_in_batch)
            weight_fraction_for_backward = local_tokens.detach() / num_items
            weighted_loss = tensor * weight_fraction_for_backward

            # For reporting: sum of weighted_loss across ranks = total_CE / num_items
            # This matches CP=1's per-batch loss (each normalized by total GA tokens)
            global_loss_for_report = weighted_loss.detach().clone()
            dist.all_reduce(
                global_loss_for_report, op = dist.ReduceOp.SUM, group = self._cp_group
            )

            # Return weighted_loss (for backward) and global_loss (for reporting)
            return weighted_loss, global_loss_for_report, global_tokens

        def _finalize(weighted_loss, global_loss, global_tokens):
            # weighted_loss: used for backward (gradients flow correctly per-rank)
            # global_loss: used for reporting (same value on all ranks)
            self._set_report_loss(global_loss)
            self._set_report_tokens(global_tokens)
            # Return weighted_loss for backward - gradients will be correctly scaled
            return weighted_loss

        if isinstance(loss, tuple):
            if not loss:
                return loss
            weighted_loss, global_loss, global_tokens = _reduce_tensor(loss[0])
            result = _finalize(weighted_loss, global_loss, global_tokens)
            return (result, *loss[1:])
        elif torch.is_tensor(loss):
            weighted_loss, global_loss, global_tokens = _reduce_tensor(loss)
            return _finalize(weighted_loss, global_loss, global_tokens)
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
                from accelerate.utils import DistributedType

                args = getattr(self, "args", None)
                distributed_state = getattr(args, "distributed_state", None)
                if (
                    distributed_state is not None
                    and distributed_state.distributed_type == DistributedType.MULTI_GPU
                ):
                    distributed_state.distributed_type = DistributedType.NO
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

        # Attach attention hooks for proper ring attention behavior with load balancing.
        # This ensures attention_mask is removed and is_causal=True for all self_attn calls.
        if manager and manager.enabled:
            model = getattr(self, "model", None)
            if model is not None:
                manager.attach_attention_hooks(model)
            elif int(os.environ.get("RANK", "0")) == 0:
                print(
                    "Context parallelism: WARNING - model not available at init time for hook attachment"
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
        # Cache num_items_in_batch and GA steps for reduce_loss, then remove from inputs
        # so model uses local token count
        if manager and manager.enabled:
            # Reset cached values from previous step
            manager._cached_num_items = None
            # Cache gradient accumulation steps for reduce_loss
            manager._cached_ga_steps = getattr(
                getattr(self, "args", None), "gradient_accumulation_steps", 1
            )
            # Cache num_items_in_batch before removing
            num_items_val = kwargs.get("num_items_in_batch")
            if num_items_val is not None:
                manager._cached_num_items = (
                    num_items_val.item()
                    if torch.is_tensor(num_items_val)
                    else num_items_val
                )
            else:
                # num_items_in_batch not provided (model signature check may have failed)
                # This can happen when unsloth's _unsloth_get_batch_samples determines
                # the model doesn't accept **kwargs. Debug to understand why.
                # Fallback: compute from labels (approximation for GA)
                labels = inputs.get("labels")
                if isinstance(labels, torch.Tensor):
                    token_count = labels[..., 1:] != -100
                    attention_mask = inputs.get("attention_mask")
                    if isinstance(attention_mask, torch.Tensor):
                        token_count = token_count & (attention_mask[..., 1:] != 0)
                    local_count = token_count.sum()
                    ga_steps = manager._cached_ga_steps
                    # Note: This is an approximation assuming uniform batch sizes
                    manager._cached_num_items = local_count.item() * ga_steps

            # Debug: verify num_items_in_batch and dataset items
            kwargs.pop("num_items_in_batch", None)
            inputs.pop("num_items_in_batch", None)
        # One-time sanity check to verify model outputs match between CP and non-CP
        if (
            manager
            and manager.enabled
            and os.environ.get("UNSLOTH_CP_SANITY_CHECK") == "1"
            and not getattr(manager, "_sanity_check_done", False)
        ):
            manager._sanity_check_done = True
            try:
                _run_cp_sanity_check(model, manager)
            except Exception as e:
                print(f"[CP-SANITY] Error running sanity check: {e}")

        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            if manager:
                override = manager.consume_num_items_override()
                if override is not None:
                    kwargs["num_items_in_batch"] = override
            rank = 0
            if dist.is_initialized():
                rank = dist.get_rank()
            # Enhanced CP Loss Debug
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

            # Debug: trace the compute_loss path
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
                # Use unsloth's fast CE for numerical consistency with CP=1
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
        # Log CP=1 losses for comparison with CP=2
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
            # Attach attention hooks if not already done (model may not be ready at init)
            if manager.enabled and not manager._attention_hook_handles:
                model = getattr(self, "model", None)
                if model is not None:
                    manager.attach_attention_hooks(model)
            # Enter SDPA patch for the entire training step (forward + backward)
            # This is needed because gradient checkpointing re-runs forward during
            # backward, and we need SDPA to remain patched throughout.
            manager.enter_sdpa_patch()
            # Note: static_graph is disabled because it causes internal PyTorch
            # assertion errors with unsloth's gradient checkpointing.
            # sync_each_batch keeps the graph constant instead.
        try:
            loss = original_training_step(self, *args, **kwargs)
        finally:
            if manager:
                # Exit SDPA patch after backward completes
                manager.exit_sdpa_patch()
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
