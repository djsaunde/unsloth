import pytest

try:
    from unsloth import DEVICE_TYPE
    from unsloth.models.llama import FastLlamaModel, LlamaAttention, LlamaModel
    import torch
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
except NotImplementedError:
    pytest.skip("Jagged tensor tests require a GPU-enabled environment", allow_module_level=True)


def _device():
    if DEVICE_TYPE in ("cuda", "hip", "xpu"):
        return torch.device(DEVICE_TYPE)
    return torch.device("cuda")


def _build_config(**overrides):
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=32000,
        rope_theta=10000.0,
    )
    defaults.update(overrides)
    return LlamaConfig(**defaults)


def _jagged_long(seqs, device):
    return torch.nested.nested_tensor([torch.tensor(seq, dtype=torch.long, device=device) for seq in seqs], layout=torch.jagged)


def _jagged_hidden(seqs, device, dtype):
    return torch.nested.nested_tensor([torch.tensor(seq, dtype=dtype, device=device) for seq in seqs], layout=torch.jagged)


def test_llama_attention_accepts_jagged_inputs():
    device = _device()
    FastLlamaModel.pre_patch()
    config = _build_config(num_hidden_layers=1)
    attention = LlamaAttention(config).to(device)

    hidden = _jagged_hidden([[[0.1]*64]*7, [[0.2]*64]*5], device, attention.q_proj.weight.dtype)
    output, attn_weights, cache = attention(hidden_states=hidden, causal_mask=None, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False)

    assert output.is_nested
    assert attn_weights is None
    assert cache is None


def test_llama_model_forward_returns_jagged_hidden_state():
    device = _device()
    FastLlamaModel.pre_patch()
    config = _build_config()
    model = LlamaModel(config).to(device)

    input_ids = _jagged_long([[1, 2, 3, 4], [5, 6]], device)
    outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=False, return_dict=True)

    assert outputs.last_hidden_state.is_nested


def test_llama_causallm_handles_jagged_labels():
    device = _device()
    FastLlamaModel.pre_patch()
    config = _build_config()
    model = LlamaForCausalLM(config).to(device)

    sequences = [[1, 2, 3], [4, 5]]
    input_ids = _jagged_long(sequences, device)
    labels = _jagged_long(sequences, device)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )

    assert outputs.loss is not None
    assert outputs.logits.shape[0] == len(sequences)
    assert outputs.logits.shape[1] == max(len(seq) for seq in sequences)
