"""Utilities for enabling sample packing across Unsloth entry points."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for sample packing on an existing trainer."""

    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    def _mark_allow_overlength(module):
        if hasattr(module, "max_seq_length"):
            setattr(module, "_unsloth_allow_packed_overlength", True)
        for child in module.children():
            _mark_allow_overlength(child)

    _mark_allow_overlength(model)

    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "flash_attention_2"
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.attn_implementation = "flash_attention_2"

    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_packing_wrapped", False):
        return

    original_torch_call = collator.torch_call

    def torch_call_with_lengths(examples: Sequence[dict]):
        batch = original_torch_call(examples)
        if examples and isinstance(examples[0], dict):
            seq_lengths: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    seq_lengths.extend(int(length) for length in lengths)
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(seq_lengths, dtype=torch.int32)
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True
