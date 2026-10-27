"""
engine/models.py - which model architectures the custom engine runs, and how they differ.

Like vLLM's and SGLang's model registries, but small: the custom engine reuses each transformers
model's own modules and weights (projections, norms, rotary embeddings, MLP / MoE blocks) and only
replaces attention and the KV cache. So any decoder that follows the common "pre-norm Llama"
layout works, as long as we know how it prepares queries and keys:

    x = x + attention(input_layernorm(x))          rotary positions, optional q/k norms
    x = x + mlp(post_attention_layernorm(x))       dense MLP or a Mixture-of-Experts block

What differs between architectures, and is detected here from the loaded model:
    qk_norm   "none"      Llama, Mistral, Qwen2
              "per_head"  Qwen3, Qwen3-MoE: RMSNorm over each head's head_dim, after the reshape
              "full"      OLMoE: RMSNorm over the whole q/k projection, before the reshape
    moe       True when the MLP is a Mixture-of-Experts block (OLMoE, Qwen3-MoE); the fused MoE
              kernels and expert quantization then apply
    GQA, head_dim != hidden/heads, q/k/v biases (Qwen2), clip_qkv (OLMoE), rope scaling (Llama 3):
              handled generically (the paged kernel supports GQA; the projections carry their biases;
              rotary embeddings come from the model's own module)

Not supported (rejected at startup with a clear message): sliding-window attention (Mistral v0.1,
Gemma, Qwen with use_sliding_window) and architectures with a different layer layout (OLMo 2
normalizes after attention, Gemma adds extra norms, DeepSeek uses MLA).

Short names usable as MODEL=... are in config.MODELS.
"""

# architecture -> what we've tested it with (tests/test_models.py builds a tiny version of each)
SUPPORTED = {
    "OlmoeForCausalLM": "OLMoE (MoE, q/k norm over the full projection)",
    "Qwen3ForCausalLM": "Qwen3 dense (GQA, per-head q/k norm)",
    "Qwen3MoeForCausalLM": "Qwen3-MoE (GQA, per-head q/k norm, fused MoE applies)",
    "Qwen2ForCausalLM": "Qwen2 / Qwen2.5 (GQA, q/k/v biases)",
    "LlamaForCausalLM": "Llama 2 / 3.x (GQA, rope scaling)",
    "MistralForCausalLM": "Mistral without sliding window (e.g. v0.3)",
}


def describe(model, max_model_len):
    """Check the model is supported and return what the runner needs to know about it."""
    cfg = model.config
    arch = (getattr(cfg, "architectures", None) or [type(model).__name__])[0]
    if arch not in SUPPORTED:
        raise ValueError(f"The custom engine doesn't support {arch} yet. Supported: "
                         f"{', '.join(SUPPORTED)} (ENGINE=hf runs any model through transformers).")

    layer_types = set(getattr(cfg, "layer_types", None) or [])
    window = getattr(cfg, "sliding_window", None)
    if any("sliding" in t for t in layer_types) or (
            getattr(cfg, "use_sliding_window", True) and window and window < max_model_len):
        raise ValueError(f"{arch} here uses sliding-window attention (window {window}), which the custom "
                         f"engine's KV cache and kernels don't implement. Use ENGINE=hf for this model.")

    attn = model.model.layers[0].self_attn
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    qk_norm = "none"
    if getattr(attn, "q_norm", None) is not None:
        size = attn.q_norm.weight.numel()
        qk_norm = "per_head" if size == head_dim else "full"
        if qk_norm == "full" and size != heads * head_dim:
            raise ValueError(f"{arch}: q_norm has {size} weights, expected {head_dim} or {heads * head_dim}")

    moe = any(getattr(cfg, k, None) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    return {"architecture": arch, "qk_norm": qk_norm, "moe": bool(moe),
            "heads": heads, "kv_heads": getattr(cfg, "num_key_value_heads", None) or heads, "head_dim": head_dim}