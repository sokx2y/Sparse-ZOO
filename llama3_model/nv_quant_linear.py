import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


SVD_ZOO_ROOT = Path(__file__).resolve().parents[1] / "SVD-ZOO-Quant"
if not SVD_ZOO_ROOT.is_dir():
    raise FileNotFoundError(f"SVD-ZOO-Quant directory not found: {SVD_ZOO_ROOT}")
if str(SVD_ZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(SVD_ZOO_ROOT))

from smoothquant.fake_quant import (
    quantize_activation_nvfp8,
    quantize_weight_nvfp4,
    quantize_weight_nvfp8,
)


NV_GROUP_SIZE = 16


class NVQuantLinear(nn.Linear):
    weight_format = None

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if in_features % NV_GROUP_SIZE != 0:
            raise ValueError(
                f"NV quantization requires in_features divisible by {NV_GROUP_SIZE}, "
                f"got {in_features}"
            )

    @classmethod
    def from_float(cls, module: nn.Linear):
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        new_module.weight = module.weight
        if module.bias is not None:
            new_module.bias = module.bias
        new_module.requantize_weight_()
        return new_module

    @torch.no_grad()
    def requantize_weight_(self):
        if self.weight_format == "nvfp4":
            quantized = quantize_weight_nvfp4(self.weight, group_size=NV_GROUP_SIZE)
        elif self.weight_format == "nvfp8":
            quantized = quantize_weight_nvfp8(self.weight, group_size=NV_GROUP_SIZE)
        else:
            raise ValueError(f"Unsupported NV weight format: {self.weight_format}")
        self.weight.copy_(quantized.to(device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_input = quantize_activation_nvfp8(
            input,
            group_size=NV_GROUP_SIZE,
        )
        return F.linear(quantized_input.to(dtype=self.weight.dtype), self.weight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_format={self.weight_format}, "
            f"activation_format=nvfp8, group_size={NV_GROUP_SIZE}"
        )


class NVW4A8Linear(NVQuantLinear):
    weight_format = "nvfp4"


class NVW8A8Linear(NVQuantLinear):
    weight_format = "nvfp8"


def _linear_class_for_mode(mode: str):
    normalized_mode = mode.lower()
    if normalized_mode == "w4a8":
        return NVW4A8Linear
    if normalized_mode == "w8a8":
        return NVW8A8Linear
    raise ValueError(f"Unsupported NV quantization mode: {mode}")


def replace_opt_linears_with_nv(model: nn.Module, mode: str) -> int:
    if getattr(getattr(model, "config", None), "model_type", None) != "opt":
        raise ValueError("replace_opt_linears_with_nv requires an OPT model")
    return _replace_decoder_linears(model, mode, "model.decoder.layers")


def replace_llama_linears_with_nv(model: nn.Module, mode: str) -> int:
    if getattr(getattr(model, "config", None), "model_type", None) != "llama":
        raise ValueError("replace_llama_linears_with_nv requires a Llama model")
    return _replace_decoder_linears(model, mode, "model.layers")


def _replace_decoder_linears(model: nn.Module, mode: str, decoder_prefix: str) -> int:
    linear_class = _linear_class_for_mode(mode)
    replaced = 0

    def replace_module(module, prefix=""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            in_decoder_layers = full_name.startswith(decoder_prefix)

            if (
                in_decoder_layers
                and isinstance(child, nn.Linear)
                and not isinstance(child, NVQuantLinear)
            ):
                setattr(module, name, linear_class.from_float(child))
                replaced += 1
                continue

            replace_module(child, full_name)

    replace_module(model)
    if replaced == 0:
        raise RuntimeError(f"No Linear modules were replaced under {decoder_prefix}")
    return replaced