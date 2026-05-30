import csv
import logging
import math
import re
from pathlib import Path

import numpy as np
import torch

from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op
from mx.specs import finalize_mx_specs


logger = logging.getLogger(__name__)

DEFAULT_LAYER_REGEX = (
    r"^model\.layers\.(0|7|14|21|27)\..*"
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

SUMMARY_FIELDS = (
    "layer_name",
    "tensor_type",
    "shape",
    "mean",
    "std",
    "min",
    "max",
    "abs_p50",
    "abs_p95",
    "abs_p99",
    "abs_p999",
    "zero_ratio",
    "positive_ratio",
    "negative_ratio",
    "skewness",
    "kurtosis",
    "sampled",
    "sampled_numel",
    "original_numel",
)


def make_linear_weight_mx_specs(weight_elem_format):
    """Build the weight-side MX specs used by QdiffLinear."""
    return finalize_mx_specs(
        {
            "w_elem_format": weight_elem_format,
            "a_elem_format": None,
            "scale_bits": 8,
            "block_size": 16,
            "bfloat": 16,
            "custom_cuda": True,
            "quantize_backprop": False,
        }
    )


def quantize_weight_like_mx_linear(weight, mx_specs):
    """Quantize one weight-side tensor with the real MX linear weight path."""
    if mx_specs is None:
        return weight

    # mx.linear.LinearFunction does not expose Q(weight) as a standalone API.
    # Replay its exact weight prefix here: bfloat/elemwise rounding first, then
    # block MX quantization on the input-feature axis with w_elem_format.
    bf_weight = quantize_elemwise_op(
        weight,
        mx_specs=mx_specs,
        round=mx_specs["round_weight"],
    )
    return quantize_mx_op(
        bf_weight,
        mx_specs,
        elem_format=mx_specs["w_elem_format"],
        axes=[-1],
        round=mx_specs["round_mx_output"],
    )


class PerturbDistributionProfiler:
    def __init__(
        self,
        output_dir="linear_outlier_plots/perturb_distribution",
        layer_regex=DEFAULT_LAYER_REGEX,
        direct_weight_mx_specs=None,
        delta_weight_mx_specs=None,
        max_sample_elements=200_000,
        log_y=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_regex = layer_regex or DEFAULT_LAYER_REGEX
        self._regex = re.compile(self.layer_regex)
        self.direct_weight_mx_specs = direct_weight_mx_specs
        self.delta_weight_mx_specs = delta_weight_mx_specs
        self.max_sample_elements = max(1, int(max_sample_elements))
        self.log_y = bool(log_y)
        self._attached_layers = []
        self._profiled_layers = set()
        self._summary_rows = []

    def matches(self, layer_name):
        return self._regex.search(str(layer_name)) is not None

    def note_attached_layers(self, layer_names):
        self._attached_layers = list(layer_names)

    def record_forward_delta(self, layer_name, module, u, v, scale, call_id=None):
        layer_name = str(layer_name or getattr(module, "layer_name", "linear"))
        if call_id is None or int(call_id) != 30:
            return
        if not self.matches(layer_name) or layer_name in self._profiled_layers:
            return
        if u is None or v is None or scale is None:
            raise RuntimeError(
                "[PERTURB_DISTRIBUTION_PROVIDER_MISS] "
                f"Selected layer {layer_name!r} did not provide u, v, and scale."
            )
        if getattr(module, "weight", None) is None:
            raise RuntimeError(
                "[PERTURB_DISTRIBUTION_WEIGHT_MISS] "
                f"Selected layer {layer_name!r} has no weight tensor."
            )

        weight = module.weight.detach()
        if u.ndim != 2 or v.ndim != 2:
            raise RuntimeError(
                "[PERTURB_DISTRIBUTION_PROVIDER_SHAPE] "
                f"Expected rank-2 u/v for {layer_name!r}, got {tuple(u.shape)} "
                f"and {tuple(v.shape)}."
            )

        direct_specs, delta_specs = self._module_mx_specs(module)
        with torch.no_grad():
            # This is the same low-rank delta tensor implied by diffLinear:
            # tmp @ u.T with tmp = x @ v corresponds to x @ (u @ v.T).T.
            raw_dw = torch.matmul(u, v.t()).mul(scale)
            if tuple(raw_dw.shape) != tuple(weight.shape):
                raise RuntimeError(
                    "[PERTURB_DISTRIBUTION_PROVIDER_SHAPE] "
                    f"raw_dw shape {tuple(raw_dw.shape)} does not match "
                    f"{layer_name!r} weight shape {tuple(weight.shape)}."
                )

            indices = self._sample_indices(weight.numel(), weight.device)
            sampled = indices is not None
            raw_sample = self._sample_tensor(raw_dw, indices)
            weight_sample = self._sample_tensor(weight, indices)

            qw = quantize_weight_like_mx_linear(weight, direct_specs)
            qw_sample = self._sample_tensor(qw, indices)

            qw_plus_dw = quantize_weight_like_mx_linear(weight + raw_dw, direct_specs)
            qw_plus_dw_sample = self._sample_tensor(qw_plus_dw, indices)
            direct_eff_sample = qw_plus_dw_sample - qw_sample

            delta_eff_dw = quantize_weight_like_mx_linear(raw_dw, delta_specs)
            delta_eff_sample = self._sample_tensor(delta_eff_dw, indices)

        arrays = {
            "raw_dw": raw_sample,
            "direct_eff_dw": direct_eff_sample,
            "delta_eff_dw": delta_eff_sample,
        }
        stem = self._layer_stem(layer_name)
        plot_path = self._save_plot(stem, layer_name, arrays)
        npz_path = self._save_npz(
            stem=stem,
            layer_name=layer_name,
            shape=tuple(weight.shape),
            indices=indices,
            sampled=sampled,
            raw_dw=raw_sample,
            direct_eff_dw=direct_eff_sample,
            delta_eff_dw=delta_eff_sample,
            weight=weight_sample,
            qw=qw_sample,
            qw_plus_dw=qw_plus_dw_sample,
        )
        for tensor_type, array in arrays.items():
            self._summary_rows.append(
                self._summary_row(
                    layer_name=layer_name,
                    tensor_type=tensor_type,
                    shape=tuple(weight.shape),
                    array=array,
                    sampled=sampled,
                    original_numel=weight.numel(),
                )
            )
        self._profiled_layers.add(layer_name)
        self.flush()
        logger.info(
            "Saved perturb distribution profile for %s from call %s: %s, %s "
            "(sampled=%s, samples=%d/%d)",
            layer_name,
            call_id,
            plot_path,
            npz_path,
            sampled,
            raw_sample.size,
            weight.numel(),
        )

    def flush(self):
        csv_path = self.output_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(self._summary_rows)
        return csv_path

    def _module_mx_specs(self, module):
        if module.__class__.__name__ == "QdiffLinear":
            return module.mx_specs0, module.mx_specs2
        return self.direct_weight_mx_specs, self.delta_weight_mx_specs

    def _sample_indices(self, numel, device):
        if numel <= self.max_sample_elements:
            return None
        stride = int(math.ceil(float(numel) / float(self.max_sample_elements)))
        return torch.arange(
            0,
            numel,
            stride,
            device=device,
            dtype=torch.long,
        )[: self.max_sample_elements]

    @staticmethod
    def _sample_tensor(tensor, indices):
        flat = tensor.detach().reshape(-1)
        if indices is not None:
            flat = flat.index_select(0, indices)
        values = flat.to(dtype=torch.float32, device="cpu").numpy()
        if indices is None:
            return values.reshape(tuple(tensor.shape))
        return values

    def _save_plot(self, stem, layer_name, arrays):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panel_defs = (
            ("raw_dw", "raw dW"),
            ("direct_eff_dw", "Q(W+dW)-Q(W)"),
            ("delta_eff_dw", "Q(dW)"),
        )
        x_min, x_max = self._shared_plot_range(arrays.values())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        for axis, (tensor_type, title) in zip(axes, panel_defs):
            values = self._finite_flat(arrays[tensor_type])
            axis.hist(
                values,
                bins=160,
                range=(x_min, x_max),
                density=True,
                color="#31688e",
                alpha=0.88,
            )
            axis.axvline(0.0, color="#b23a48", linewidth=1.0)
            axis.set_title(title)
            axis.set_xlim(x_min, x_max)
            axis.set_xlabel("value")
            axis.set_ylabel("density")
            if self.log_y:
                axis.set_yscale("log")
            stats = self._stats(values)
            axis.text(
                0.03,
                0.97,
                self._annotation(stats),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
        fig.suptitle(
            "Distribution comparison of LOZO perturbations under MX quantization\n"
            f"{layer_name}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        path = self.output_dir / f"{stem}_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_npz(
        self,
        stem,
        layer_name,
        shape,
        indices,
        sampled,
        raw_dw,
        direct_eff_dw,
        delta_eff_dw,
        weight,
        qw,
        qw_plus_dw,
    ):
        suffix = "sampled" if sampled else "full"
        path = self.output_dir / f"{stem}_perturb_tensors_{suffix}.npz"
        sample_indices = (
            indices.detach().to(device="cpu").numpy()
            if indices is not None
            else np.empty((0,), dtype=np.int64)
        )
        np.savez_compressed(
            path,
            layer_name=np.asarray(layer_name),
            original_shape=np.asarray(shape, dtype=np.int64),
            tensors_are_sampled=np.asarray(sampled),
            sample_indices=sample_indices,
            raw_dw=raw_dw,
            direct_eff_dw=direct_eff_dw,
            delta_eff_dw=delta_eff_dw,
            W=weight,
            QW=qw,
            QW_plus_dw=qw_plus_dw,
        )
        return path

    def _summary_row(
        self,
        layer_name,
        tensor_type,
        shape,
        array,
        sampled,
        original_numel,
    ):
        stats = self._stats(self._finite_flat(array))
        row = {
            "layer_name": layer_name,
            "tensor_type": tensor_type,
            "shape": "x".join(str(dim) for dim in shape),
            "sampled": bool(sampled),
            "sampled_numel": int(np.asarray(array).size),
            "original_numel": int(original_numel),
        }
        row.update(stats)
        return row

    @staticmethod
    def _finite_flat(array):
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        return flat[np.isfinite(flat)]

    def _shared_plot_range(self, arrays):
        values = [self._finite_flat(array) for array in arrays]
        finite_values = [value for value in values if value.size > 0]
        if not finite_values:
            return -1.0, 1.0
        combined = np.concatenate(finite_values)
        x_min, x_max = np.percentile(combined, [0.1, 99.9])
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return -1.0, 1.0
        if x_min == x_max:
            margin = abs(float(x_min)) * 0.05 or 1.0
            return float(x_min - margin), float(x_max + margin)
        return float(x_min), float(x_max)

    @staticmethod
    def _stats(values):
        if values.size == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "abs_p50": float("nan"),
                "abs_p95": float("nan"),
                "abs_p99": float("nan"),
                "abs_p999": float("nan"),
                "zero_ratio": float("nan"),
                "positive_ratio": float("nan"),
                "negative_ratio": float("nan"),
                "skewness": float("nan"),
                "kurtosis": float("nan"),
            }

        mean = float(np.mean(values))
        std = float(np.std(values))
        abs_percentiles = np.percentile(np.abs(values), [50, 95, 99, 99.9])
        if std == 0.0:
            skewness = float("nan")
            kurtosis = float("nan")
        else:
            centered = (values - mean) / std
            skewness = float(np.mean(centered**3))
            # Fisher excess kurtosis. A normal sample is near 0.
            kurtosis = float(np.mean(centered**4) - 3.0)
        return {
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "abs_p50": float(abs_percentiles[0]),
            "abs_p95": float(abs_percentiles[1]),
            "abs_p99": float(abs_percentiles[2]),
            "abs_p999": float(abs_percentiles[3]),
            "zero_ratio": float(np.mean(values == 0)),
            "positive_ratio": float(np.mean(values > 0)),
            "negative_ratio": float(np.mean(values < 0)),
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    @staticmethod
    def _annotation(stats):
        return (
            f"mean {stats['mean']:.3e}\n"
            f"std  {stats['std']:.3e}\n"
            f"zero {stats['zero_ratio']:.3f}\n"
            f"pos  {stats['positive_ratio']:.3f}\n"
            f"neg  {stats['negative_ratio']:.3f}\n"
            f"skew {stats['skewness']:.3f}\n"
            f"kurt {stats['kurtosis']:.3f}"
        )

    @staticmethod
    def _layer_stem(layer_name):
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name)
        leaf_name = re.sub(r"[^A-Za-z0-9_]+", "_", layer_name.split(".")[-1])
        if layer_match is not None:
            return f"layer_{int(layer_match.group(1)):02d}_{leaf_name}"
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", layer_name).strip("_")
        return safe_name or "linear"


def attach_perturb_distribution_profiler(model, profiler):
    selected = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ not in ("diffLinear", "QdiffLinear"):
            continue
        layer_name = str(getattr(module, "layer_name", module_name) or module_name)
        if not profiler.matches(layer_name):
            continue
        if getattr(module, "uv_provider", None) is None:
            raise RuntimeError(
                "[PERTURB_DISTRIBUTION_PROVIDER_MISS] "
                f"Selected layer {layer_name!r} has no uv_provider."
            )
        module.perturb_distribution_profiler = profiler
        selected.append(layer_name)

    if not selected:
        raise RuntimeError(
            "[PERTURB_DISTRIBUTION_NO_LAYERS] No diffLinear/QdiffLinear "
            f"layers matched regex {profiler.layer_regex!r}."
        )

    profiler.note_attached_layers(selected)
    return profiler