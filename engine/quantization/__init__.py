"""
engine/quantization - weight quantization methods and the code that applies them.

Pick a method in config.py (QUANTIZATION) or with the QUANT environment variable.
"""
import torch

from engine.quantization.fp8 import FP8WeightAct
from engine.quantization.int4_wo import Int4WeightOnly
from engine.quantization.int8_wo import Int8WeightOnly

METHODS = {
    "int8_wo": Int8WeightOnly,
    "int4_wo": Int4WeightOnly,
    "fp8": FP8WeightAct,
}


def get_method(name: str, device="cpu", compute_dtype=torch.bfloat16, group_size=128, fp8_backend="auto"):
    if name not in METHODS:
        raise ValueError(f"Unknown quantization '{name}'. Options: none, {', '.join(METHODS)}")
    if name == "int4_wo":
        return Int4WeightOnly(group_size=group_size, compute_dtype=compute_dtype)
    if name == "fp8":
        return FP8WeightAct(backend=fp8_backend, device=device, compute_dtype=compute_dtype)
    return METHODS[name](compute_dtype=compute_dtype)