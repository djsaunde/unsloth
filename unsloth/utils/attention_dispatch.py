# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from torch import Tensor
import torch.nn.functional as F

from ..models._utils import *
from ..context_parallel import get_cp_manager
from ..utils.packing import (
    build_sdpa_packed_attention_mask,
    build_xformers_block_causal_mask,
)
from ..utils.ring_attention import (
    is_ring_flash_attn_available,
    get_ring_attn_func,
    RingAttnVariant,
    llama3_flash_attn_prepare_cu_seqlens,
)

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
HAS_XFORMERS = xformers is not None
BlockDiagonalCausalMask = None
if HAS_XFORMERS:
    BlockDiagonalCausalMask = xformers.attn_bias.BlockDiagonalCausalMask
SDPA_HAS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")

FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"
RING_FLASH_DENSE = "ring_flash_dense"
RING_FLASH_VARLEN = "ring_flash_varlen"

_CP_SDPA_FALLBACK_LOGGED = False


XFORMERS_BLOCK_DIAG_CLS = (
    xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None
)


@dataclass
class AttentionConfig:
    """
    Per-layer attention metadata.

    NOTE(djsaunde): I had originally intended this to be populated once per layer, but
        we're currently constructing it on every forward pass since it can possibly be
        invalid from one forward pass to the next (e.g., switching from training to
        inference). For now, I'm keeping separate from AttentionContext for the sake of
        better grouping of params.
    """

    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AttentionContext:
    """Per-call info required to run attention."""

    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None


def select_attention_backend(use_varlen: bool = False) -> str:
    """Return attention backend based on availability / priority order."""

    cp_manager = get_cp_manager()
    if cp_manager is not None:
        # Context parallelism active - check for ring-flash-attn
        if cp_manager.use_ring_flash_attn:
            # Ring-flash-attention available - use it
            if use_varlen:
                return RING_FLASH_VARLEN
            return RING_FLASH_DENSE
        else:
            # Fall back to SDPA-based CP (no varlen support)
            if use_varlen:
                raise ValueError(
                    "Context parallelism with SDPA does not support varlen/packing mode. "
                    "Install ring-flash-attention (`pip install ring-flash-attn`) for packing support, "
                    "or disable packing, or set context_parallel_size=1."
                )
            global _CP_SDPA_FALLBACK_LOGGED
            if not _CP_SDPA_FALLBACK_LOGGED:
                print(
                    "Unsloth: Context parallelism using SDPA backend (ring-flash-attn not installed)."
                )
                _CP_SDPA_FALLBACK_LOGGED = True
            return SDPA

    # Standard single-GPU path
    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    if HAS_XFORMERS:
        return XFORMERS
    return SDPA


def run_attention(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    """
    Run attention using config / context info.

    Backend choice is prioritized for speed: FlashAttention when installed
    (`flash_varlen` for packed/variable-length inputs with `seq_info`, otherwise dense
    flash), then xFormers if flash is unavailable, with PyTorch SDPA as the final
    fallback (e.g., CPU or no fused kernels).

    Varlen flash is preferred when packing metadata is present because it avoids padding
    and keeps peak memory low. xFormers and SDPA can also handle packed batches (we
    pass a block-diagonal mask into each).
    """

    backend = config.backend
    if backend == FLASH_VARLEN and context.seq_info is None:
        backend = FLASH_DENSE if HAS_FLASH_ATTENTION else SDPA
    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    requires_grad = context.requires_grad
    sliding_window = context.sliding_window

    if backend == FLASH_VARLEN:
        Q_f = Q.transpose(1, 2).reshape(bsz * q_len, n_heads, head_dim)
        K_f = K.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        V_f = V.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        _, cu_seqlens, max_seqlen = context.seq_info
        return flash_attn_varlen_func(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **flash_varlen_kwargs,
        ).view(bsz, q_len, n_heads, head_dim)
    elif backend == FLASH_DENSE:
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        return flash_attn_func(Q_t, K_t, V_t, **flash_dense_kwargs).reshape(
            bsz, q_len, n_heads, head_dim
        )
    elif backend == RING_FLASH_DENSE:
        cp_manager = get_cp_manager()
        ring_attn_fn = get_ring_attn_func(use_varlen = False)

        # Ring-flash-attn expects (B, S, H, D) layout with contiguous tensors
        Q_t = Q.transpose(1, 2).contiguous()
        K_t = K.transpose(1, 2).contiguous()
        V_t = V.transpose(1, 2).contiguous()

        # Debug: Check for nan/inf in inputs
        import os
        if os.environ.get("UNSLOTH_DEBUG_RING_ATTN"):
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            q_has_nan = Q_t.isnan().any().item()
            k_has_nan = K_t.isnan().any().item()
            v_has_nan = V_t.isnan().any().item()
            print(f"[Rank {rank}] RING_FLASH_DENSE input shapes: Q={Q_t.shape}, K={K_t.shape}, V={V_t.shape}")
            print(f"[Rank {rank}] Input has nan: Q={q_has_nan}, K={k_has_nan}, V={v_has_nan}")
            print(f"[Rank {rank}] Q range: [{Q_t.min().item():.4f}, {Q_t.max().item():.4f}]")
            print(f"[Rank {rank}] cp_group={cp_manager.cp_group}, cp_rank={cp_manager.cp_rank_index}")

        out = ring_attn_fn(
            Q_t,
            K_t,
            V_t,
            dropout_p = flash_dense_kwargs.get("dropout_p", 0.0),
            causal = flash_dense_kwargs.get("causal", True),
            group = cp_manager.cp_group,
        )

        # Debug: Check for nan/inf in output
        if os.environ.get("UNSLOTH_DEBUG_RING_ATTN"):
            out_has_nan = out.isnan().any().item()
            out_has_inf = out.isinf().any().item()
            print(f"[Rank {rank}] RING_FLASH_DENSE output has nan={out_has_nan}, inf={out_has_inf}")
            if not out_has_nan and not out_has_inf:
                print(f"[Rank {rank}] Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

        return out.reshape(bsz, q_len, n_heads, head_dim)
    elif backend == RING_FLASH_VARLEN:
        cp_manager = get_cp_manager()
        ring_attn_fn = get_ring_attn_func(use_varlen = True)

        # Get global cu_seqlens from cp_manager (computed before sharding)
        cu_seqlens_global = cp_manager.varlen_cu_seqlens
        if cu_seqlens_global is None:
            raise ValueError("RING_FLASH_VARLEN requires cu_seqlens for packed sequences")

        # Ensure cu_seqlens is on the same device as Q/K/V
        cu_seqlens_global = cu_seqlens_global.to(device = Q.device)

        causal = flash_varlen_kwargs.get("causal", True)
        world_size = cp_manager.settings.size

        # Q/K/V are already LOCAL (pre-sharded by context_parallel.apply)
        # Reshape to (local_tokens, heads, head_dim) for varlen format
        local_tokens = bsz * q_len
        Q_local = Q.transpose(1, 2).reshape(local_tokens, n_heads, head_dim).contiguous()
        K_local = K.transpose(1, 2).reshape(local_tokens, config.n_kv_heads, head_dim).contiguous()
        V_local = V.transpose(1, 2).reshape(local_tokens, config.n_kv_heads, head_dim).contiguous()

        # Use llama3_flash_attn_prepare_cu_seqlens to compute local cu_seqlens
        # from global cu_seqlens for this rank
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_global,
            causal = causal,
            rank = cp_manager.cp_rank_index,
            world_size = world_size,
        )

        # Debug: Check for nan/inf in inputs
        import os
        if os.environ.get("UNSLOTH_DEBUG_RING_ATTN"):
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            q_has_nan = Q_local.isnan().any().item()
            k_has_nan = K_local.isnan().any().item()
            v_has_nan = V_local.isnan().any().item()
            print(f"[Rank {rank}] RING_FLASH_VARLEN local shapes: Q={Q_local.shape}, K={K_local.shape}, V={V_local.shape}")
            print(f"[Rank {rank}] Input has nan: Q={q_has_nan}, K={k_has_nan}, V={v_has_nan}")
            print(f"[Rank {rank}] cu_seqlens_global={cu_seqlens_global.tolist()}")
            print(f"[Rank {rank}] cu_seqlens_q={cu_seqlens_q.tolist()}, cu_seqlens_k={cu_seqlens_k.tolist()}")
            print(f"[Rank {rank}] max_seqlen_q={max_seqlen_q}, max_seqlen_k={max_seqlen_k}")
            print(f"[Rank {rank}] local_k_slice={local_k_slice}")

        out = ring_attn_fn(
            Q_local,
            K_local,
            V_local,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride = 1,
            local_k_slice = local_k_slice,
            dropout_p = flash_varlen_kwargs.get("dropout_p", 0.0),
            causal = causal,
            group = cp_manager.cp_group,
        )

        # Debug: Check for nan/inf in output
        if os.environ.get("UNSLOTH_DEBUG_RING_ATTN"):
            out_has_nan = out.isnan().any().item()
            out_has_inf = out.isinf().any().item()
            print(f"[Rank {rank}] RING_FLASH_VARLEN output has nan={out_has_nan}, inf={out_has_inf}")
            if not out_has_nan and not out_has_inf:
                print(f"[Rank {rank}] Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

        # Output has shape (local_tokens, n_heads, head_dim)
        return out.view(bsz, q_len, n_heads, head_dim)
    elif backend == XFORMERS:
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window = sliding_window,
            base_mask = context.causal_mask,
        )

        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        K_mod = K_t
        V_mod = V_t
        Q_mod = Q_t

        if config.n_groups != 1:
            K_mod = K_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            V_mod = V_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            K_mod = K_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )
            V_mod = V_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )

            if requires_grad:
                K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q_mod = Q_t.view(
                    bsz, q_len, config.n_kv_heads, config.n_groups, head_dim
                )

        has_block = XFORMERS_BLOCK_DIAG_CLS is not None and isinstance(
            attn_bias, XFORMERS_BLOCK_DIAG_CLS
        )

        if config.n_groups != 1 and has_block:
            if not requires_grad:
                Q_mod = Q_mod.view(
                    1, bsz * q_len, config.n_kv_heads, config.n_groups, head_dim
                )
                K_mod = K_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
                V_mod = V_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
            else:
                Q_mod = Q_mod.view(1, bsz * q_len, n_heads, head_dim)
                K_mod = K_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)

        out = xformers_attention(
            Q_mod,
            K_mod,
            V_mod,
            attn_bias = attn_bias,
            **xformers_kwargs,
        )

        if config.n_groups != 1 and not requires_grad:
            if has_block:
                out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            else:
                out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            out = out.reshape(bsz, q_len, n_heads, head_dim)
        else:
            if has_block:
                out = out.view(bsz, q_len, n_heads, head_dim)
            else:
                out = out.view(bsz, q_len, n_heads, head_dim)
        return out
    else:
        local_mask = context.attention_mask
        is_causal_local = False
        if context.seq_info is not None and local_mask is None:
            local_mask = build_sdpa_packed_attention_mask(
                context.seq_info,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
            )
        else:
            q_len_local = Q.shape[-2]
            k_len_local = K.shape[-2]
            is_causal_local = local_mask is None and q_len_local == k_len_local

        kwargs = dict(sdpa_kwargs)
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        if SDPA_HAS_GQA:
            kwargs.setdefault("enable_gqa", config.n_groups != 1)
            out = F.scaled_dot_product_attention(Q, K, V, **kwargs)
            return out.transpose(1, 2)

        K_mod = K
        V_mod = V
        if config.n_groups != 1:
            K_mod = K[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            V_mod = V[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            K_mod = K_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V_mod = V_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)

        out = F.scaled_dot_product_attention(
            Q.contiguous(),
            K_mod.contiguous(),
            V_mod.contiguous(),
            **kwargs,
        )
        return out.transpose(1, 2).contiguous()


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "run_attention",
    "FLASH_VARLEN",
    "FLASH_DENSE",
    "XFORMERS",
    "SDPA",
    "RING_FLASH_DENSE",
    "RING_FLASH_VARLEN",
]
