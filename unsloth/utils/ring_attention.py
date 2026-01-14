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

"""Ring-flash-attention integration for context parallelism."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional


class RingAttnVariant(Enum):
    """Ring attention variant selection."""

    ZIGZAG = "zigzag"  # ~70% of single-GPU FA, best load balance
    AUTO = "auto"  # Automatically select zigzag


# Try importing ring-flash-attention functions
HAS_RING_FLASH_ATTN = False
zigzag_ring_flash_attn_func = None
zigzag_ring_flash_attn_varlen_func = None

try:
    from ring_flash_attn import (
        zigzag_ring_flash_attn_func as _zigzag_dense,
        zigzag_ring_flash_attn_varlen_func as _zigzag_varlen,
    )

    zigzag_ring_flash_attn_func = _zigzag_dense
    zigzag_ring_flash_attn_varlen_func = _zigzag_varlen
    HAS_RING_FLASH_ATTN = True
except ImportError:
    pass


def is_ring_flash_attn_available() -> bool:
    """Check if ring-flash-attention is installed and usable."""
    return HAS_RING_FLASH_ATTN


def get_ring_attn_func(
    use_varlen: bool, variant: RingAttnVariant = RingAttnVariant.AUTO
) -> Optional[Callable]:
    """
    Get the appropriate ring attention function based on mode and variant.

    Args:
        use_varlen: Whether to use variable-length (packed) attention.
        variant: Which ring attention variant to use.

    Returns:
        The ring attention function, or None if not available.
    """
    if not HAS_RING_FLASH_ATTN:
        return None

    # Currently only zigzag is implemented
    if use_varlen:
        return zigzag_ring_flash_attn_varlen_func
    return zigzag_ring_flash_attn_func


def parse_ring_attn_variant(variant_str: str) -> RingAttnVariant:
    """Parse a string variant name into RingAttnVariant enum."""
    variant_str = variant_str.lower().strip()
    if variant_str == "zigzag":
        return RingAttnVariant.ZIGZAG
    return RingAttnVariant.AUTO


__all__ = [
    "HAS_RING_FLASH_ATTN",
    "RingAttnVariant",
    "is_ring_flash_attn_available",
    "get_ring_attn_func",
    "parse_ring_attn_variant",
    "zigzag_ring_flash_attn_func",
    "zigzag_ring_flash_attn_varlen_func",
]
