import config
from engine.hf_engine import HFEngine
from engine.custom_engine import CustomEngine

def create_engine():
    if config.ENGINE == "hf":
        return HFEngine()
    if config.ENGINE == "custom":
        return CustomEngine()
    raise ValueError(f"Unknown ENGINE '{config.ENGINE}' (use 'hf' or 'custom')")


def engine_label(info: dict) -> str:
    """Short name for result tables and charts, in the same format bench/sweep.py accepts:
    'hf', 'hf+fused-moe', 'continuous/paged', 'continuous/paged+fused-moe+prefix+kv-fp8', ..."""
    e = info.get("engine") or {}
    fused = "+fused-moe" if e.get("moe_kernel") == "triton" else ""
    if e.get("engine") != "custom":
        return "hf" + fused
    label = f"{e.get('scheduler')}/{e.get('kv_cache')}" + fused
    if e.get("attn_backend") == "triton":
        label += "+paged-attn"
    if e.get("cuda_graphs"):
        label += "+cuda-graphs"
    if e.get("fused_proj"):
        label += "+fused-proj"
    if e.get("prefix_caching"):
        label += "+prefix"
    if e.get("kv_cache_dtype") in ("fp8", "int8"):
        label += f"+kv-{e['kv_cache_dtype']}"
    if e.get("chunked_prefill"):
        label += f"+chunked+budget-{e.get('max_batched_tokens')}"
    return label