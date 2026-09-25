"""engine/model_loader.py - loads the model and reports where it ended up."""
import time
import torch
from transformers import pipeline

import config


def load_model():
    if config.QUIET_TRANSFORMERS:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    print(f"Loading model: {config.MODEL_NAME} ...")
    start = time.perf_counter()
    generator = pipeline(
        "text-generation",
        model=config.MODEL_NAME,
        device_map=config.DEVICE_MAP,
        dtype=config.DTYPE,   # on older transformers versions, rename this to torch_dtype
    )
    print(f"Model loaded in {time.perf_counter() - start:.1f}s")
    print_diagnostics(generator)
    return generator


def print_diagnostics(generator):
    print("\n===== Model diagnostics =====")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            used = torch.cuda.memory_allocated(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} | "
                  f"model uses {used / 1e9:.2f} GB of {total / 1e9:.2f} GB")
    print("Model device:", generator.model.device)
    print("Model dtype:", generator.model.dtype)
    device_map = getattr(generator.model, "hf_device_map", None)
    print("Device map:", device_map)
    if device_map and any(str(d) in ("cpu", "disk") for d in device_map.values()):
        print("WARNING: part of the model is offloaded to CPU/disk - generation will be very slow.")
    print("=============================\n")