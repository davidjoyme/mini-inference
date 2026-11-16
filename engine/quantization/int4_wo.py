import torch
from engine.quantization.base import QuantMethod


class Int4WeightOnly(QuantMethod):
    name = "int4_wo"

    def __init__(self, group_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def supports(self, weight):
        return weight.shape[-1] % self.group_size == 0 and self.group_size % 2 == 0

    def quantize(self, weight):
        *lead, out_f, in_f = weight.shape
        g = self.group_size
        w = weight.float().reshape(*lead, out_f, in_f // g, g)

        w_min = w.amin(dim=-1, keepdim=True)
        w_max = w.amax(dim=-1, keepdim=True)
        scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        q = torch.round((w - w_min) / scale).clamp(0, 15).to(torch.uint8)
        q = q.reshape(*lead, out_f, in_f)

        packed = q[..., 0::2] | (q[..., 1::2] << 4)          # [..., out, in/2] uint8
        return {
            "qweight": packed,
            "scale": scale.squeeze(-1).to(self.compute_dtype),   # [..., out, in/g]
            "zero": w_min.squeeze(-1).to(self.compute_dtype),    # [..., out, in/g]
        }

    def dequantize(self, qweight, scale, zero):
        low = qweight & 0x0F
        high = qweight >> 4
        q = torch.stack((low, high), dim=-1).reshape(*qweight.shape[:-1], qweight.shape[-1] * 2)

        *lead, out_f, in_f = q.shape
        g = in_f // scale.shape[-1]
        q = q.reshape(*lead, out_f, in_f // g, g).to(scale.dtype)
        w = q * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        return w.reshape(*lead, out_f, in_f)

    def describe(self):
        return {"method": self.name, "group_size": self.group_size}