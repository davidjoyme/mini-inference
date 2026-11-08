"""
engine/quantization/int8_wo.py - INT8 weight-only quantization (W8A16).

Each output channel (row of the weight) gets one scale:
    scale = max(|row|) / 127
    q     = round(w / scale)           stored as int8 in [-127, 127]
    w    ~= q * scale
Activations stay in bfloat16. Memory: 1 byte per weight + a tiny scale per row.
"""
import torch
from engine.quantization.base import QuantMethod


class Int8WeightOnly(QuantMethod):
    name = "int8_wo"

    def quantize(self, weight: torch.Tensor) -> dict:
        w = weight.float()
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0   # [..., out, 1]
        q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        return {"qweight": q, "scale": scale.to(self.compute_dtype)}

    def dequantize(self, qweight, scale) -> torch.Tensor:
        return qweight.to(scale.dtype) * scale

    def linear(self, x, qweight, scale):
        # Multiplying the output by the scale (instead of scaling the weight first)
        # gives the same result with less work: (x @ q.T) * scale.T
        y = torch.nn.functional.linear(x, qweight.to(x.dtype))
        return y * scale.squeeze(-1).to(x.dtype)