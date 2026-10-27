"""
engine/quantization/loader.py - load a model with quantized weights.

First start with a given setting ("fresh"):
    1. load the bf16 model into CPU RAM
    2. for each decoder layer: move it to the GPU, quantize its weights there,
       drop the bf16 copy (so the GPU never has to hold the full bf16 model)
    3. move the rest (embeddings, final norm, lm_head) to the GPU
    4. save the quantized weights to QUANT_CACHE_DIR

Later starts with the same setting ("cache"):
    1. build an empty model skeleton (no memory used) with quantized layers in place
    2. load the saved quantized weights straight onto the GPU
"""
import ctypes
import gc
import json
import os
import time

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from engine.quantization.modules import QuantExperts, QuantLinear, is_fused_experts


def _set_module(root: nn.Module, name: str, new: nn.Module):
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def _decoder_layers(model):
    try:
        return model.model.layers
    except AttributeError:
        raise RuntimeError("Expected the decoder layers at model.model.layers; "
                           "adapt _decoder_layers() for this architecture.")


def _free_memory(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)   # hand freed CPU RAM back to the OS (Linux)
    except OSError:
        pass


def quantize_model(model, method, targets, device):
    """Replace eligible layers with quantized ones, layer by layer, on `device`.
    Returns counts of what was quantized. Also works on device='meta' (shapes only)."""
    stats = {"experts": 0, "linear": 0, "lm_head": 0, "skipped": [],
             "quantized_params": 0}

    for i, layer in enumerate(_decoder_layers(model)):
        layer.to(device)
        for name, module in list(layer.named_modules()):
            full_name = f"model.layers.{i}.{name}"
            if "experts" in targets and is_fused_experts(module):
                if method.supports(module.gate_up_proj) and method.supports(module.down_proj):
                    stats["quantized_params"] += module.gate_up_proj.numel() + module.down_proj.numel()
                    _set_module(layer, name, QuantExperts(module, method))
                    stats["experts"] += 1
                else:
                    stats["skipped"].append(full_name)
            elif "linear" in targets and isinstance(module, nn.Linear):
                if method.supports(module.weight):
                    stats["quantized_params"] += module.weight.numel()
                    _set_module(layer, name, QuantLinear(module, method))
                    stats["linear"] += 1
                else:
                    stats["skipped"].append(full_name)
        if device != "meta":
            _free_memory(device)

    if "lm_head" in targets and isinstance(getattr(model, "lm_head", None), nn.Linear):
        if method.supports(model.lm_head.weight):
            stats["quantized_params"] += model.lm_head.weight.numel()
            model.lm_head = QuantLinear(model.lm_head.to(device), method)
            stats["lm_head"] = 1
        else:
            stats["skipped"].append("lm_head")

    model.to(device)   # everything not quantized: embeddings, norms, router, rotary
    return stats


def cache_path(cache_dir, model_name, method, targets):
    safe = model_name.strip("/").replace("/", "__")
    tag = method.name + (f"_g{method.group_size}" if hasattr(method, "group_size") else "")
    return os.path.join(cache_dir, f"{safe}__{tag}__{'-'.join(sorted(targets))}.safetensors")


def save_cache(model, path, info):
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Tied weights (e.g. Qwen3-4B's lm_head IS its embedding matrix) are one tensor under two
    # names; safetensors refuses shared tensors, so save it once and record the other name.
    state, aliases, seen = {}, {}, {}
    for k, v in model.state_dict().items():
        key = (v.data_ptr(), tuple(v.shape), v.dtype, v.device)
        if key in seen:
            aliases[k] = seen[key]
            continue
        seen[key] = k
        state[k] = v.detach().to("cpu").contiguous()
    tmp = path + ".tmp"
    save_file(state, tmp, metadata={"info": json.dumps(info), "aliases": json.dumps(aliases)})
    os.replace(tmp, path)   # only appears under its real name once fully written
    del state
    _free_memory("cpu")


def _reinit_meta_buffers(model, device):
    """Non-persistent buffers (e.g. rotary embedding frequencies) aren't saved in the
    checkpoint, so after loading they're still empty. Rebuild those modules from config."""
    for name, module in list(model.named_modules()):
        if any(b.is_meta for b in module.buffers(recurse=False)):
            try:
                rebuilt = type(module)(config=model.config)
                _set_module(model, name, rebuilt.to(device))
            except TypeError as e:
                raise RuntimeError(f"Can't rebuild {name} ({type(module).__name__}) after loading "
                                   f"from cache: {e}. Set QUANT_USE_CACHE=0.") from e
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(f"Weights missing after cache load (e.g. {leftover[:3]}). "
                           f"Delete the cache file or set QUANT_USE_CACHE=0.")


def load_from_cache(path, model_name, method, targets, device, dtype):
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    stats = quantize_model(model, method, targets, "meta")   # puts empty quantized layers in place

    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        aliases = json.loads((f.metadata() or {}).get("aliases", "{}"))
    state = load_file(path, device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    for name, source in aliases.items():                # re-tie shared weights
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, model.get_parameter(source))
    missing = [m for m in missing if m not in aliases]
    if missing or unexpected:
        raise RuntimeError(f"Cache {path} doesn't match the model "
                           f"(missing {missing[:3]}, unexpected {unexpected[:3]}). Delete it and restart.")
    del state
    _reinit_meta_buffers(model, device)
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    except OSError:
        pass
    return model.eval(), stats


def load_quantized_model(model_name, method, targets, device, dtype, cache_dir=None, use_cache=True):
    start = time.perf_counter()
    path = cache_path(cache_dir, model_name, method, targets) if cache_dir else None

    if use_cache and path and os.path.exists(path):
        print(f"Loading quantized weights from cache: {path}")
        model, stats = load_from_cache(path, model_name, method, targets, device, dtype)
        source = "cache"
    else:
        print(f"Loading bf16 weights into CPU RAM, then quantizing layer by layer onto {device} ...")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.eval()
        stats = quantize_model(model, method, targets, device)
        source = "fresh"
        if use_cache and path:
            print(f"Saving quantized weights to {path} (next start will be faster) ...")
            save_cache(model, path, {"model": model_name, **method.describe(), "targets": targets})

    stats["load_source"] = source
    stats["cache_path"] = path
    stats["load_time_s"] = round(time.perf_counter() - start, 1)
    return model, stats


def weight_memory(model) -> dict:
    """Bytes held by parameters and buffers, split into quantized layers and the rest."""
    quant = other = 0
    quant_ids = set()
    for m in model.modules():
        if isinstance(m, (QuantLinear, QuantExperts)):
            for t in list(m.parameters()) + list(m.buffers()):
                quant_ids.add(id(t))
                quant += t.numel() * t.element_size()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in quant_ids:
            other += t.numel() * t.element_size()
    mb = 1024 ** 2
    return {"weights_total_mb": round((quant + other) / mb, 1),
            "weights_quantized_layers_mb": round(quant / mb, 1),
            "weights_other_mb": round(other / mb, 1)}