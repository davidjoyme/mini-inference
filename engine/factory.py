"""
engine/factory.py - build the engine selected in config.py (ENGINE, or the ENGINE env variable).

    "hf"      engine.hf_engine.HFEngine          transformers pipeline, one request at a time
    "custom"  engine.custom_engine.CustomEngine  own scheduler / KV cache / attention (engine/core/)

Both implement engine.base.Engine, so callers never need to know which one they got.
"""
import config


def create_engine():
    if config.ENGINE == "hf":
        from engine.hf_engine import HFEngine
        return HFEngine()
    if config.ENGINE == "custom":
        from engine.custom_engine import CustomEngine
        return CustomEngine()
    raise ValueError(f"Unknown ENGINE '{config.ENGINE}' (use 'hf' or 'custom')")


def engine_label(info: dict) -> str:
    """Short name for result tables and charts, in the same format bench/sweep.py accepts:
    'hf', 'continuous/paged', 'continuous/paged+prefix+kv-fp8', ..."""
    e = info.get("engine") or {}
    if e.get("engine") != "custom":
        return "hf"
    label = f"{e.get('scheduler')}/{e.get('kv_cache')}"
    if e.get("prefix_caching"):
        label += "+prefix"
    if e.get("kv_cache_dtype") in ("fp8", "int8"):
        label += f"+kv-{e['kv_cache_dtype']}"
    if e.get("chunked_prefill"):
        label += f"+chunked+budget-{e.get('max_batched_tokens')}"
    return label