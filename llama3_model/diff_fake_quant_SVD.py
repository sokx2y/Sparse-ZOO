import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from diff_fake_quant_mx import (
    diffEmbedding,
    diffLinear,
    diffLlamaRMSNorm,
)
from forward_delta_debug import (
    record_forward_delta_profile,
    record_svd_terms_profile,
    svd_term_profile_enabled,
)


SVD_ZOO_ROOT = Path(__file__).resolve().parents[1] / "SVD-ZOO-Quant"
if SVD_ZOO_ROOT.exists() and str(SVD_ZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(SVD_ZOO_ROOT))

try:
    from smoothquant.fake_quant import (
        _quantize_groupwise_fp,
        quantize_activation_mxfp4,
        quantize_activation_mxfp8,
        quantize_activation_nvfp4,
        quantize_activation_nvfp8,
        quantize_weight_mxint4,
        quantize_weight_mxfp4,
        quantize_weight_mxfp8,
        quantize_weight_nvint4,
        quantize_weight_nvfp4,
        quantize_weight_nvfp8,
    )
except Exception:
    _quantize_groupwise_fp = None
    quantize_activation_mxfp4 = None
    quantize_activation_mxfp8 = None
    quantize_activation_nvfp4 = None
    quantize_activation_nvfp8 = None
    quantize_weight_mxint4 = None
    quantize_weight_mxfp4 = None
    quantize_weight_mxfp8 = None
    quantize_weight_nvint4 = None
    quantize_weight_nvfp4 = None
    quantize_weight_nvfp8 = None


def svd_lora_output(input: torch.Tensor, u: Optional[torch.Tensor], v: Optional[torch.Tensor]):
    if u is None or v is None:
        return None
    flat = input.reshape(-1, input.shape[-1]).to(dtype=v.dtype)
    out = flat.matmul(v).matmul(u.t())
    return out.view(*input.shape[:-1], u.shape[0])


def lowrank_direction_from_provider(provider, param_name, param, inference_count):
    if provider is None or param is None or not param.requires_grad:
        return None, None
    left, right, scale = provider(
        param_name,
        param.shape,
        param.device,
        param.dtype,
        inference_count,
    )
    return left.matmul(right.t()), scale


def quantize_svd_lora_factor_nvfp8(tensor: Optional[torch.Tensor], return_raw: bool = False):
    if tensor is None:
        return (None, None) if return_raw else None
    if quantize_weight_nvfp8 is None:
        raise RuntimeError("SVD-LoRA factor FP8 quantization requires smoothquant.fake_quant to be importable")
    quantized = quantize_weight_nvfp8(tensor, group_size=None)
    return (quantized, quantized) if return_raw else quantized


def _finite_stats(name, value):
    if value is None:
        return f"{name}=None"
    value_f = value.detach().float()
    finite = torch.isfinite(value_f)
    if finite.any():
        finite_vals = value_f[finite]
        return (
            f"{name}:finite={bool(finite.all().item())}"
            f",min={float(finite_vals.min().cpu()):.6e}"
            f",max={float(finite_vals.max().cpu()):.6e}"
        )
    return f"{name}:finite=False,all_nonfinite=True"


def default_group_size_for_format(quant_format: Optional[str]):
    quant_format = (quant_format or "none").lower()
    if quant_format in {"nvint4", "nvfp4", "nvfp8"}:
        return 16
    if quant_format in {"mxint4", "mxfp4", "mxfp8"}:
        return 32
    return None


def _effective_group_size(quant_format: Optional[str], group_size: Optional[int]):
    return group_size if group_size and group_size > 0 else default_group_size_for_format(quant_format)


@torch.no_grad()
def quantize_activation_nvfp8(t: torch.Tensor, group_size=16):
    if _quantize_groupwise_fp is None:
        raise RuntimeError("nvfp8 activation quantization requires smoothquant.fake_quant to be importable")
    t = t.contiguous()
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    return _quantize_groupwise_fp(
        t,
        n_bits=8,
        e_bits=4,
        m_bits=3,
        group_size=group_size,
        dim=-1,
        scale_quant=True,
    ).view(t_shape)


@torch.no_grad()
def quantize_activation_mxfp8(t: torch.Tensor, group_size=32):
    if _quantize_groupwise_fp is None:
        raise RuntimeError("mxfp8 activation quantization requires smoothquant.fake_quant to be importable")
    t = t.contiguous()
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    return _quantize_groupwise_fp(
        t,
        n_bits=8,
        e_bits=4,
        m_bits=3,
        group_size=group_size,
        dim=-1,
        e8_scale=True,
        e8_scale_op="ceil",
    ).view(t_shape)


def activation_quantizers_from_format(quant_format: Optional[str], group_size: Optional[int]):
    quant_format = (quant_format or "none").lower()
    backend = quant_backend_from_format(quant_format)
    if backend == "none":
        return lambda x: x, lambda x: x
    group_size = _effective_group_size(quant_format, group_size)
    if backend == "nv":
        if quantize_activation_nvfp4 is None:
            raise RuntimeError("nvfp4 activation quantization requires smoothquant.fake_quant to be importable")
        if quant_format == "nvfp8" and quantize_activation_nvfp8 is None:
            raise RuntimeError("nvfp8 activation quantization requires smoothquant.fake_quant to be importable")
        return (
            lambda x: quantize_activation_nvfp8(x, group_size=group_size),
            lambda x: quantize_activation_nvfp4(x, group_size=group_size),
        )
    if backend == "mx":
        if quantize_activation_mxfp4 is None:
            raise RuntimeError("mxfp4 activation quantization requires smoothquant.fake_quant to be importable")
        if quant_format == "mxfp8" and quantize_activation_mxfp8 is None:
            raise RuntimeError("mxfp8 activation quantization requires smoothquant.fake_quant to be importable")
        return (
            lambda x: quantize_activation_mxfp8(x, group_size=group_size),
            lambda x: quantize_activation_mxfp4(x, group_size=group_size),
        )
    raise ValueError(f"Unsupported SVD-LoRA quant_format: {quant_format}")


def quantize_frozen_base_weight(weight: torch.Tensor, quant_format: Optional[str], group_size: int):
    quant_format = (quant_format or "none").lower()
    if quant_format in {None, "", "none"}:
        return weight
    group_size = _effective_group_size(quant_format, group_size)
    if quant_format == "nvint4":
        if quantize_weight_nvint4 is None:
            raise RuntimeError("nvint4 requires smoothquant.fake_quant to be importable")
        return quantize_weight_nvint4(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    if quant_format == "nvfp4":
        if quantize_weight_nvfp4 is None:
            raise RuntimeError("nvfp4 requires smoothquant.fake_quant to be importable")
        return quantize_weight_nvfp4(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    if quant_format == "nvfp8":
        if quantize_weight_nvfp8 is None:
            raise RuntimeError("nvfp8 requires smoothquant.fake_quant to be importable")
        return quantize_weight_nvfp8(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    if quant_format == "mxint4":
        if quantize_weight_mxint4 is None:
            raise RuntimeError("mxint4 requires smoothquant.fake_quant to be importable")
        return quantize_weight_mxint4(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    if quant_format == "mxfp4":
        if quantize_weight_mxfp4 is None:
            raise RuntimeError("mxfp4 requires smoothquant.fake_quant to be importable")
        return quantize_weight_mxfp4(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    if quant_format == "mxfp8":
        if quantize_weight_mxfp8 is None:
            raise RuntimeError("mxfp8 requires smoothquant.fake_quant to be importable")
        return quantize_weight_mxfp8(weight.detach(), group_size=group_size).to(
            device=weight.device, dtype=weight.dtype
        )
    raise ValueError(f"Unsupported frozen base quant format: {quant_format}")


def quant_backend_from_format(quant_format: Optional[str]):
    quant_format = (quant_format or "none").lower()
    if quant_format in {"none", "fp", "fp16", "fp32"}:
        return "none"
    if quant_format in {"nvint4", "nvfp4", "nvfp8"}:
        return "nv"
    if quant_format in {"mxint4", "mxfp4", "mxfp8"}:
        return "mx"
    raise ValueError(f"Unsupported SVD-LoRA quant_format: {quant_format}")


def set_svd_lora(
    module,
    u,
    v,
    trainable=True,
    freeze_base=True,
    base_quant_format=None,
    group_size=16,
    residual_weight=None,
    lora_scaling: float = 1.0,
    sanity_check: bool = False,
):
    if u.shape[0] != module.out_features or v.shape[0] != module.in_features or u.shape[1] != v.shape[1]:
        raise ValueError(
            f"SVD-LoRA shape mismatch for {getattr(module, 'layer_name', '<unnamed>')}: "
            f"u={tuple(u.shape)} v={tuple(v.shape)} linear=({module.out_features}, {module.in_features})"
        )
    u = u * float(lora_scaling)
    sanity = None
    if residual_weight is not None:
        if residual_weight.shape != module.weight.shape:
            raise ValueError(
                f"SVD-LoRA residual base shape mismatch for {getattr(module, 'layer_name', '<unnamed>')}: "
                f"residual={tuple(residual_weight.shape)} weight={tuple(module.weight.shape)}"
            )
        if sanity_check:
            smooth_weight = module.weight.detach().float().cpu()
            residual_cpu = residual_weight.detach().float().cpu()
            s_actual = u.detach().float().cpu().matmul(v.detach().float().cpu().t())
            err = residual_cpu + s_actual - smooth_weight
            sanity = {
                "smooth_norm": float(smooth_weight.norm().item()),
                "residual_norm": float(residual_cpu.norm().item()),
                "s_actual_norm": float(s_actual.norm().item()),
                "reconstruction_error_norm": float(err.norm().item()),
            }
        with torch.no_grad():
            module.weight.data.copy_(residual_weight.to(device=module.weight.device, dtype=module.weight.dtype))
    if base_quant_format not in {None, "", "none"}:
        with torch.no_grad():
            module.weight.data.copy_(quantize_frozen_base_weight(module.weight.data, base_quant_format, group_size))
    if freeze_base:
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
    module.svd_lora_u = nn.Parameter(
        u.to(device=module.weight.device, dtype=module.weight.dtype).contiguous(),
        requires_grad=trainable,
    )
    module.svd_lora_v = nn.Parameter(
        v.to(device=module.weight.device, dtype=module.weight.dtype).contiguous(),
        requires_grad=trainable,
    )
    return sanity


def svd_lora_forward_delta(module, input, diff_input):
    u = getattr(module, "svd_lora_u", None)
    v = getattr(module, "svd_lora_v", None)
    if u is None or v is None:
        return None, None

    du, u_scale = lowrank_direction_from_provider(
        module.uv_provider,
        module.layer_name + ".svd_lora_u",
        u,
        module.inference_count,
    )
    dv, v_scale = lowrank_direction_from_provider(
        module.uv_provider,
        module.layer_name + ".svd_lora_v",
        v,
        module.inference_count,
    )
    if getattr(module, "quantize_svd_lora_factors_fp8", False):
        # QdiffSVDLinear locally FP8-quantizes U/V/dU/dV before SVD-LoRA forward-delta math.
        if getattr(module, "debug_nonfinite", False):
            u, u_q = quantize_svd_lora_factor_nvfp8(u, return_raw=True)
            v, v_q = quantize_svd_lora_factor_nvfp8(v, return_raw=True)
            du, du_q = quantize_svd_lora_factor_nvfp8(du, return_raw=True)
            dv, dv_q = quantize_svd_lora_factor_nvfp8(dv, return_raw=True)
            quantized_factors = (u_q, v_q, du_q, dv_q)
            if (
                any(t is not None and not torch.isfinite(t).all() for t in quantized_factors)
                and not getattr(module.__class__, "_printed_factor_nonfinite_debug", False)
            ):
                module.__class__._printed_factor_nonfinite_debug = True
                print(
                    "[QDIFF-SVD-FACTOR-NONFINITE]",
                    f"layer={module.layer_name}",
                    f"inference_count={module.inference_count}",
                    _finite_stats("u", getattr(module, "svd_lora_u", None)),
                    _finite_stats("v", getattr(module, "svd_lora_v", None)),
                    _finite_stats("u_q", u_q),
                    _finite_stats("v_q", v_q),
                    _finite_stats("du_q", du_q),
                    _finite_stats("dv_q", dv_q),
                    flush=True,
                )
        else:
            u = quantize_svd_lora_factor_nvfp8(u)
            v = quantize_svd_lora_factor_nvfp8(v)
            du = quantize_svd_lora_factor_nvfp8(du)
            dv = quantize_svd_lora_factor_nvfp8(dv)

    plus = svd_lora_output(input, u, v)
    if plus is None:
        return None, None
    if du is None and dv is None:
        svd_diff = svd_lora_output(diff_input, u, v)
        if svd_term_profile_enabled(module.inference_count):
            zero = torch.zeros_like(svd_diff)
            record_svd_terms_profile(
                module.layer_name,
                module,
                module.inference_count,
                {
                    "x_dv_u": zero,
                    "x_v_du": zero,
                    "x_dv_du": zero,
                    "dx_v_u": svd_diff,
                    "dx_dv_u": zero,
                    "dx_v_du": zero,
                    "dx_dv_du": zero,
                },
                svd_diff,
                {
                    "first_order": svd_diff,
                    "no_dx_factor_perturb": svd_diff,
                    "only_second_order_and_dx_base": svd_diff,
                },
            )
        return plus, svd_diff

    delta_u = torch.zeros_like(u) if du is None else u_scale * du
    delta_v = torch.zeros_like(v) if dv is None else v_scale * dv
    use_appro_svd = getattr(module, "appro_SVD", False)
    profile_svd_terms = svd_term_profile_enabled(module.inference_count)

    if use_appro_svd and not profile_svd_terms:
        svd_diff = (
            svd_lora_output(input, u, delta_v)
            + svd_lora_output(input, delta_u, v)
            + svd_lora_output(diff_input, u, v)
            + svd_lora_output(diff_input, delta_u, v)
        )
        return plus, svd_diff

    u_minus = u + delta_u
    v_minus = v + delta_v
    minus = svd_lora_output(input, u_minus, v_minus)
    diff_minus = svd_lora_output(diff_input, u_minus, v_minus)
    if diff_minus is not None:
        minus = minus + diff_minus
    full_svd_diff = minus - plus
    svd_diff = full_svd_diff

    if profile_svd_terms:
        terms = {
            "x_dv_u": svd_lora_output(input, u, delta_v),
            "x_v_du": svd_lora_output(input, delta_u, v),
            "x_dv_du": svd_lora_output(input, delta_u, delta_v),
            "dx_v_u": svd_lora_output(diff_input, u, v),
            "dx_dv_u": svd_lora_output(diff_input, u, delta_v),
            "dx_v_du": svd_lora_output(diff_input, delta_u, v),
            "dx_dv_du": svd_lora_output(diff_input, delta_u, delta_v),
        }
        approximations = {
            "first_order": terms["x_dv_u"] + terms["x_v_du"] + terms["dx_v_u"],
            "appro_SVD": terms["x_dv_u"] + terms["x_v_du"] + terms["dx_v_u"] + terms["dx_v_du"],
            "no_dx_factor_perturb": terms["x_dv_u"] + terms["x_v_du"] + terms["x_dv_du"] + terms["dx_v_u"],
            "only_second_order_and_dx_base": terms["x_dv_du"] + terms["dx_v_u"] + terms["dx_dv_du"],
            "sum_all_terms": sum(terms.values()),
        }
        if getattr(module, "appro_SVD", False):
            svd_diff = approximations["appro_SVD"]
        record_svd_terms_profile(
            module.layer_name,
            module,
            module.inference_count,
            terms,
            full_svd_diff,
            approximations,
        )
    return plus, svd_diff


class diffSVDLinear(diffLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        svd_out = svd_lora_output(
            input,
            getattr(self, "svd_lora_u", None),
            getattr(self, "svd_lora_v", None),
        )
        return output if svd_out is None else output + svd_out

    def forward_delta(self, input: torch.Tensor, diff_input: torch.Tensor):
        self.inference_count += 1
        output = F.linear(input, self.weight, self.bias)
        svd_output, svd_diff = svd_lora_forward_delta(self, input, diff_input)
        if svd_output is not None:
            output = output + svd_output

        diff_output = F.linear(diff_input, self.weight, None)
        
        if svd_diff is not None:
            diff_output = diff_output + svd_diff
        record_forward_delta_profile(
            "svd",
            self.layer_name,
            self,
            self.inference_count,
            output,
            diff_output,
        )
        return output, diff_output


class QdiffSVDLinear(nn.Linear):
    _printed_nonfinite_debug = False
    _printed_factor_nonfinite_debug = False

    def __init__(
        self,
        layer_name: str,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        uv_provider=None,
        z_provider=None,
        quant_format: str = "nvfp8",
        quant_format_wdx: Optional[str] = None,
        group_size: int = 0,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.inference_count = 0
        self.layer_name = layer_name
        self.uv_provider = uv_provider
        self.z_provider = z_provider
        self.outlier_profiler = None
        self.perturb_distribution_profiler = None
        self.quant_format = (quant_format or "none").lower()
        self.quant_format_wdx = (quant_format_wdx or "none").lower()
        # uv -> fp8?
        self.quantize_svd_lora_factors_fp8 = True
        self.appro_SVD = False
        self.debug_nonfinite = False
        # usually, we can keep group_size = 0 and use _effective_group_size to decide self.group-size automatically.
        self.group_size = _effective_group_size(self.quant_format, group_size)
        self.group_size_wdx = _effective_group_size(self.quant_format_wdx, group_size)
        # activation_quantizers_from_format func always keep x->fp8, dx->fp4 in this version.
        self.act_quant, self.diff_act_quant = activation_quantizers_from_format(
            self.quant_format,
            self.group_size,
        )

    def _base_linear(self, input, weight=None, bias=None, specs=None):
        del specs
        weight = self.weight if weight is None else weight
        return F.linear(input.to(dtype=weight.dtype), weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_q = self.act_quant(input)
        output = self._base_linear(input_q, self.weight, self.bias)
        svd_out = svd_lora_output(
            input_q,
            getattr(self, "svd_lora_u", None),
            getattr(self, "svd_lora_v", None),
        )
        return output if svd_out is None else output + svd_out

    def forward_delta(self, input: torch.Tensor, diff_input: torch.Tensor):
        self.inference_count += 1
        input_q = self.act_quant(input)
        diff_input_q = self.diff_act_quant(diff_input)
        output = self._base_linear(input_q, self.weight, self.bias)
        svd_output, svd_diff = svd_lora_forward_delta(self, input_q, diff_input_q)
        if svd_output is not None:
            output = output + svd_output

        weight_wdx = (
            self.weight
            if self.quant_format_wdx in {"", "none"}
            else quantize_frozen_base_weight(self.weight, self.quant_format_wdx, self.group_size_wdx)
        )
        diff_output = F.linear(diff_input_q.to(dtype=weight_wdx.dtype), weight_wdx, None)
        
        if svd_diff is not None:
            diff_output = diff_output + svd_diff
        if self.debug_nonfinite and not QdiffSVDLinear._printed_nonfinite_debug and (
            not torch.isfinite(output).all() or not torch.isfinite(diff_output).all()
        ):
            QdiffSVDLinear._printed_nonfinite_debug = True
            with torch.no_grad():
                def finite_stats(name, value):
                    if value is None:
                        return f"{name}=None"
                    value_f = value.detach().float()
                    finite = torch.isfinite(value_f)
                    if finite.any():
                        finite_vals = value_f[finite]
                        return (
                            f"{name}:finite={bool(finite.all().item())}"
                            f",min={float(finite_vals.min().cpu()):.6e}"
                            f",max={float(finite_vals.max().cpu()):.6e}"
                        )
                    return f"{name}:finite=False,all_nonfinite=True"

                print(
                    "[QDIFF-SVD-NONFINITE]",
                    f"layer={self.layer_name}",
                    f"inference_count={self.inference_count}",
                    finite_stats("input", input),
                    finite_stats("diff_input", diff_input),
                    finite_stats("input_q", input_q),
                    finite_stats("diff_input_q", diff_input_q),
                    finite_stats("weight", self.weight),
                    finite_stats("weight_wdx", weight_wdx),
                    finite_stats("svd_output", svd_output),
                    finite_stats("svd_diff", svd_diff),
                    finite_stats("output", output),
                    finite_stats("diff_output", diff_output),
                    flush=True,
                )
        record_forward_delta_profile(
            "svd_quant",
            self.layer_name,
            self,
            self.inference_count,
            output,
            diff_output,
            extra={
                "quant_format": self.quant_format,
                "quant_format_wdx": self.quant_format_wdx,
                "group_size": self.group_size,
            },
        )
        return output, diff_output


def attach_svd_lora_to_diff_layers(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    adapter_name: str = "default",
    svd_lora_rank: Optional[int] = None,
    trainable: bool = True,
    freeze_base: bool = True,
    base_quant_format: Optional[str] = None,
    group_size: int = 16,
    compensation_mode: str = "quant_error",
    lora_scaling: float = 1.0,
    strict: bool = False,
) -> int:
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    compensation_mode = (compensation_mode or "quant_error").lower()
    if compensation_mode == "residual":
        print("[SVD-LoRA] compensation_mode=residual")
        print("[SVD-LoRA] using base formula: Q(W_residual) + S")
    else:
        print(f"[SVD-LoRA] compensation_mode={compensation_mode}")
        print("[SVD-LoRA] using base formula: Q(W_smooth) + S")

    attached = 0
    missing = []
    missing_residual = []
    for name, module in model.named_modules():
        if not isinstance(module, (diffSVDLinear, QdiffSVDLinear)):
            continue
        key_a = f"{name}.lora_A.{adapter_name}.weight"
        key_b = f"{name}.lora_B.{adapter_name}.weight"
        if key_a not in state_dict or key_b not in state_dict:
            missing.append(name)
            continue
        lora_a = state_dict[key_a]
        lora_b = state_dict[key_b]
        if svd_lora_rank is not None and (lora_a.shape[0] != svd_lora_rank or lora_b.shape[1] != svd_lora_rank):
            raise ValueError(
                f"SVD-LoRA rank mismatch for {name}: checkpoint rank "
                f"A={lora_a.shape[0]} B={lora_b.shape[1]}, svd_lora_rank={svd_lora_rank}"
            )
        residual_weight = None
        if compensation_mode == "residual":
            residual_key = f"{name}.svd_residual_weight"
            if residual_key not in state_dict:
                missing_residual.append(name)
                continue
            residual_weight = state_dict[residual_key]
        sanity = set_svd_lora(
            module,
            u=lora_b,
            v=lora_a.t().contiguous(),
            trainable=trainable,
            freeze_base=freeze_base,
            base_quant_format=base_quant_format,
            group_size=group_size,
            residual_weight=residual_weight,
            lora_scaling=lora_scaling,
            sanity_check=(compensation_mode == "residual" and attached == 0),
        )
        if compensation_mode == "residual":
            print(f"[SVD-LoRA] loaded residual base weight for {name}")
            if sanity is not None:
                print(
                    "[SVD-LoRA] residual sanity "
                    f"layer={name} norm(W_smooth)={sanity['smooth_norm']:.6e} "
                    f"norm(W_residual)={sanity['residual_norm']:.6e} "
                    f"norm(S_actual)={sanity['s_actual_norm']:.6e} "
                    f"norm(W_residual+S_actual-W_smooth)={sanity['reconstruction_error_norm']:.6e}"
                )
        attached += 1

    if strict and missing:
        raise KeyError(f"Missing SVD-LoRA checkpoint entries for {len(missing)} layers, first={missing[0]}")
    if missing_residual:
        raise KeyError(
            "Residual SVD-LoRA checkpoint is missing per-layer residual base weights. "
            "Regenerate it with the patched SVD-ZOO-Quant/experiments/init_lora_from_svd.py. "
            f"Missing {len(missing_residual)} layers, first={missing_residual[0]}"
        )
    if missing:
        print(f"[SVD-LoRA] warning: skipped {len(missing)} layers without checkpoint factors")
    if attached == 0:
        print("[SVD-LoRA] warning: no SVD-aware diff layers matched the checkpoint")
    else:
        print(f"[SVD-LoRA] attached factors to {attached} SVD-aware diff layers")
    return attached


def QuantizeLlamaForLOZOSVD(
    model,
    uv_provider=None,
    z_provider=None,
    quant_format="nvfp8",
    quant_format_wdx=None,
    group_size=0,
):
    quant_backend = quant_backend_from_format(quant_format)

    def make_linear(full_name, child):
        cls = diffSVDLinear if quant_backend == "none" else QdiffSVDLinear
        kwargs = dict(
            layer_name=full_name,
            in_features=child.in_features,
            out_features=child.out_features,
            bias=(child.bias is not None),
            device="meta",
            dtype=child.weight.dtype,
            uv_provider=uv_provider,
            z_provider=z_provider,
        )
        if cls is QdiffSVDLinear:
            kwargs.update(
                quant_format=quant_format,
                quant_format_wdx=quant_format_wdx,
                group_size=group_size,
            )
        new_linear = cls(**kwargs)
        new_linear.weight = child.weight
        if child.bias is not None:
            new_linear.bias = child.bias
        return new_linear

    def replace_llama_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            is_embed_tokens = full_name == "model.embed_tokens"
            in_decoder_layers = full_name.startswith("model.layers")
            is_lm_head = full_name == "lm_head"

            if is_embed_tokens and isinstance(child, nn.Embedding) and not isinstance(child, diffEmbedding):
                new_emb = diffEmbedding(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse,
                    layer_name=full_name,
                    uv_provider=uv_provider,
                    device="meta",
                    dtype=child.weight.dtype,
                )
                new_emb.weight = child.weight
                setattr(module, name, new_emb)
                print(f"Replace {full_name} with diffEmbedding")
                continue

            if in_decoder_layers and isinstance(child, nn.Linear) and not isinstance(child, (diffSVDLinear, QdiffSVDLinear)):
                setattr(module, name, make_linear(full_name, child))
                print(f"Replace decoder linear {full_name} with SVD-aware diff linear")
                continue

            if is_lm_head and isinstance(child, nn.Linear) and not isinstance(child, diffLinear):
                provider_layer_name = full_name
                try:
                    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and child.weight is model.model.embed_tokens.weight:
                        provider_layer_name = "model.embed_tokens"
                except Exception:
                    provider_layer_name = full_name
                new_head = diffLinear(
                    layer_name=provider_layer_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    device="meta",
                    dtype=child.weight.dtype,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
                new_head.weight = child.weight
                if child.bias is not None:
                    new_head.bias = child.bias
                setattr(module, name, new_head)
                print(f"Replace {full_name} with diffLinear, provider layer_name={provider_layer_name}")
                continue

            if child.__class__.__name__ == "LlamaRMSNorm":
                new_norm = diffLlamaRMSNorm(
                    hidden_size=child.weight.shape[0],
                    eps=child.variance_epsilon,
                    layer_name=full_name,
                    z_provider=z_provider,
                    device="meta",
                    dtype=child.weight.dtype,
                )
                new_norm.weight = child.weight
                setattr(module, name, new_norm)
                print(f"Replace {full_name} with diffLlamaRMSNorm")
                continue

            replace_llama_module(child, full_name)

    if getattr(model, "config", None) is not None and model.config.model_type == "llama":
        replace_llama_module(model)
    else:
        print("Model is not a Llama model, skip QuantizeLlamaForLOZOSVD.")


def QuantizeOPTForLOZOSVD(*args, **kwargs):
    raise NotImplementedError("SVD-aware LOZO replacement is currently implemented for Llama only.")




