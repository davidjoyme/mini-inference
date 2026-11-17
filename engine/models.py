SUPPORTED = {
    "OlmoeForCausalLM": "OLMoE (MoE, q/k norm over the full projection)",
    "Qwen3ForCausalLM": "Qwen3 dense (GQA, per-head q/k norm)",
    "Qwen3MoeForCausalLM": "Qwen3-MoE (GQA, per-head q/k norm, fused MoE applies)",
    "Qwen2ForCausalLM": "Qwen2 / Qwen2.5 (GQA, q/k/v biases)",
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