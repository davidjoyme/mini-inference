"""engine/model_loader.py - loads the model (optionally quantized) and reports where it ended up."""
import time
import torch
from transformers import AutoTokenizer, pipeline
import config

DTYPES = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
          "float16": torch.float16, "float32": torch.float32}


def quantization_enabled() -> bool:
    return str(config.QUANTIZATION).lower() not in ("none", "", "bf16", "off")


def load_model():
    """Returns (pipeline, load_info)."""
    if config.QUIET_TRANSFORMERS:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    print(f"Loading model: {config.MODEL_NAME} (quantization: {config.QUANTIZATION}) ...")
    start = time.perf_counter()

    if not quantization_enabled():
        generator = pipeline(
            "text-generation",
            model=config.MODEL_NAME,
            device_map=config.DEVICE_MAP,
            dtype=config.DTYPE,   # on older transformers versions, rename this to torch_dtype
        )
        load_info = {"quantization": {"method": "none"}}
    else:
        from engine.quantization import get_method
        from engine.quantization.loader import load_quantized_model

        # Quantized models go on a single device (multi-GPU splitting isn't supported here)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = DTYPES.get(config.DTYPE, torch.bfloat16)
        method = get_method(config.QUANTIZATION, device=device, compute_dtype=dtype,
                            group_size=config.QUANT_GROUP_SIZE, fp8_backend=config.FP8_BACKEND)
        model, stats = load_quantized_model(
            config.MODEL_NAME, method, config.QUANT_TARGETS, device, dtype,
            cache_dir=config.QUANT_CACHE_DIR, use_cache=config.QUANT_USE_CACHE,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device)
        load_info = {"quantization": {**method.describe(), "targets": config.QUANT_TARGETS, **stats}}

    from engine.quantization.loader import weight_memory
    # Fused MoE kernels (kernels/fused_moe): swap the experts modules, sharing their weights
    load_info["moe_kernel"] = "default"
    if config.MOE_KERNEL == "triton":
        from kernels.fused_moe.module import align_backend, apply_fused_moe
        swapped = apply_fused_moe(generator.model, align=config.MOE_ALIGN)
        load_info["moe_kernel"] = "triton" if swapped else "default"
        if swapped:
            load_info["moe_align"] = align_backend(config.MOE_ALIGN)   # compiles the CUDA kernel now
        print(f"Fused MoE kernel: {swapped} experts modules replaced"
              + (f", moe_align: {load_info['moe_align']}" if swapped else
                 " (none eligible - a dense model, or quantized experts: using the default path)"))
    elif config.MOE_KERNEL not in ("default", "", None):
        raise ValueError(f"Unknown MOE_KERNEL '{config.MOE_KERNEL}' (use default or triton)")

    load_info["memory"] = weight_memory(generator.model)
    load_info["load_time_s"] = round(time.perf_counter() - start, 1)

    print(f"Model loaded in {load_info['load_time_s']}s")
    print_diagnostics(generator, load_info)
    return generator, load_info


def print_diagnostics(generator, load_info):
    print("\n===== Model diagnostics =====")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            used = torch.cuda.memory_allocated(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} (compute {cap[0]}.{cap[1]}) | "
                  f"model uses {used / 1e9:.2f} GB of {total / 1e9:.2f} GB")
    print("Model device:", generator.model.device)
    print("Model dtype:", generator.model.dtype)
    cfg = generator.model.config
    print("Attention implementation:", getattr(cfg, "_attn_implementation", None))
    print("MoE kernel:", load_info.get("moe_kernel", "default"))
    print("Experts implementation:", getattr(cfg, "_experts_implementation", None),
          "(replaced by QuantExperts' per-expert loop)" if quantization_enabled() else "")
    device_map = getattr(generator.model, "hf_device_map", None)
    print("Device map:", device_map)
    if device_map and any(str(d) in ("cpu", "disk") for d in device_map.values()):
        print("WARNING: part of the model is offloaded to CPU/disk - generation will be very slow.")

    q = load_info["quantization"]
    print("Quantization:", {k: v for k, v in q.items() if k not in ("skipped", "cache_path")})
    if q.get("skipped"):
        print(f"  left in bf16 (shape not supported): {len(q['skipped'])} layers, e.g. {q['skipped'][:3]}")
    print("Weight memory:", load_info["memory"])
    print("=============================\n")