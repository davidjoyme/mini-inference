"""
engine/quantization/modules.py - drop-in replacements for the model's layers.

QuantLinear   replaces an nn.Linear (OLMoE's attention projections, optionally lm_head)
QuantExperts  replaces OLMoE's fused experts module, whose weights are two 3D tensors:
                gate_up_proj [num_experts, 2 * intermediate, hidden]
                down_proj    [num_experts, hidden, intermediate]

Quantized tensors are registered as buffers, so model.to(device) and state_dict()
handle them like normal weights. Never call model.to(dtype) after quantizing: it
would cast the FP8 weights and the scales.
"""
import torch
from torch import nn
from profiling.nvtx import nvtx_range


def _register(module: nn.Module, prefix: str, tensors: dict):
    for key, tensor in tensors.items():
        module.register_buffer(f"{prefix}{key}", tensor)


def _collect(module: nn.Module, prefix: str, keys) -> dict:
    return {key: getattr(module, f"{prefix}{key}") for key in keys}


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, method):
        super().__init__()
        self.method = method
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q = method.quantize(linear.weight.data)
        self._keys = list(q)
        _register(self, "", q)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data, requires_grad=False)

    def forward(self, x):
        y = self.method.linear(x, **_collect(self, "", self._keys))
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, method={self.method.name}"


class QuantExperts(nn.Module):
    """Same routing logic as transformers' OlmoeExperts.forward, with quantized matmuls."""

    def __init__(self, experts: nn.Module, method):
        super().__init__()
        self.method = method
        self.num_experts = experts.gate_up_proj.shape[0]
        self.act_fn = experts.act_fn

        gate_up = method.quantize(experts.gate_up_proj.data)
        down = method.quantize(experts.down_proj.data)
        self._keys = list(gate_up)
        _register(self, "gate_up_", gate_up)
        _register(self, "down_", down)

    def _expert_weights(self, prefix, e):
        # Every quantized tensor has the expert dimension first, so [e] picks one expert
        return {k: v[e] for k, v in _collect(self, prefix, self._keys).items()}

    def forward(self, hidden_states, top_k_index, top_k_weights):
        with nvtx_range("experts"):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                e = int(expert_idx[0])
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e])
                current = hidden_states[token_idx]
                gate, up = self.method.linear(current, **self._expert_weights("gate_up_", e)).chunk(2, dim=-1)
                current = self.act_fn(gate) * up
                current = self.method.linear(current, **self._expert_weights("down_", e))
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))

            return final_hidden_states

    def extra_repr(self):
        return f"num_experts={self.num_experts}, method={self.method.name}"


def is_fused_experts(module: nn.Module) -> bool:
    """OLMoE-style experts: two 3D weight tensors with the expert dimension first."""
    gu = getattr(module, "gate_up_proj", None)
    dn = getattr(module, "down_proj", None)
    return (isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter)
            and gu.dim() == 3 and dn.dim() == 3)