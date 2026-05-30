import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn


class LinearOutlierProfiler:
    def __init__(
        self,
        output_dir="linear_profile",
        max_calls_per_layer=2,
        block_size=16,
        token_stride=1,
        channel_stride=4,
        weight_stride=8,
        layer_regex="",
        save_3d_plot=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls_per_layer = int(max_calls_per_layer)
        self.block_size = int(block_size)
        self.token_stride = max(1, int(token_stride))
        self.channel_stride = max(1, int(channel_stride))
        self.weight_stride = max(1, int(weight_stride))
        self.layer_regex = layer_regex or ""
        self.save_3d_plot = bool(save_3d_plot)
        self._regex = re.compile(self.layer_regex) if self.layer_regex else None
        self._counts = {}
        self._records = []
        self._flush_pos = 0
        self._handles = []

    def should_record(self, layer_name):
        if self.max_calls_per_layer <= 0:
            return False
        if self._regex is not None and self._regex.search(layer_name) is None:
            return False
        return self._counts.get(layer_name, 0) < self.max_calls_per_layer

    def record_forward(self, layer_name, module, input_tensor, call_id=None):
        self.record_forward_delta(
            layer_name=layer_name,
            module=module,
            input_tensor=input_tensor,
            diff_input=None,
            u=None,
            v=None,
            scale=None,
            call_id=call_id,
        )

    def record_forward_delta(
        self,
        layer_name,
        module,
        input_tensor,
        diff_input,
        u=None,
        v=None,
        scale=None,
        call_id=None,
    ):
        layer_name = str(layer_name or getattr(module, "layer_name", "linear"))
        if not self.should_record(layer_name):
            return

        recorded_call = self._counts.get(layer_name, 0) + 1
        self._counts[layer_name] = recorded_call
        inference_count = call_id
        if inference_count is None:
            inference_count = getattr(module, "inference_count", recorded_call)

        try:
            x_plot = self._activation_plot_array(input_tensor)
            dx_plot = self._activation_plot_array(diff_input) if diff_input is not None else None
            w_plot = self._weight_plot_array(module.weight)
            delta_w_plot = self._delta_w_sample_array(u, v, scale)
            x_signed = self._activation_signed_array(input_tensor)
            dx_signed = self._activation_signed_array(diff_input) if diff_input is not None else None
            w_signed = self._weight_signed_array(module.weight)
            dw_signed = self._delta_w_signed_sample_array(u, v, scale)

            stats = {}
            stats["x"] = self._tensor_stats(
                "x", self._activation_stats_matrix(input_tensor), channel_axis_values=None
            )
            if diff_input is not None:
                stats["dx"] = self._tensor_stats(
                    "dx", self._activation_stats_matrix(diff_input), channel_axis_values=None
                )
            else:
                stats["dx"] = None
            stats["w"] = self._tensor_stats(
                "w", self._weight_stats_matrix(module.weight), channel_axis_values=None
            )
            if delta_w_plot is not None:
                in_idx = np.arange(0, v.shape[0], self.weight_stride, dtype=np.int64)
                stats["uv_delta_w"] = self._tensor_stats(
                    "uv_delta_w", delta_w_plot, channel_axis_values=in_idx
                )
                stats["uv_delta_w"]["sampled"] = True
                stats["uv_delta_w"]["sample_stride"] = self.weight_stride
            else:
                stats["uv_delta_w"] = None

            png_path = None
            if self.save_3d_plot:
                png_path = self._save_plot(
                    layer_name,
                    recorded_call,
                    inference_count,
                    x_plot,
                    dx_plot,
                    w_plot,
                    delta_w_plot,
                )
            curve_paths = self._save_simple_curves(
                layer_name,
                recorded_call,
                inference_count,
                x_plot,
                dx_plot,
                w_plot,
                delta_w_plot,
            )
            npy_paths = self._save_signed_npy(
                layer_name,
                recorded_call,
                inference_count,
                x_signed,
                dx_signed,
                w_signed,
                dw_signed,
            )
            hist_path = self._save_hist_plot(
                layer_name,
                recorded_call,
                inference_count,
                x_signed,
                dx_signed,
                w_signed,
                dw_signed,
            )
            hist_full_path = self._save_hist_full_plot(
                layer_name,
                recorded_call,
                inference_count,
                x_signed,
                dx_signed,
                w_signed,
                dw_signed,
            )
            self._records.append(
                {
                    "layer_name": layer_name,
                    "call_id": int(recorded_call),
                    "inference_count": self._safe_number(inference_count),
                    "plot_path": str(png_path) if png_path is not None else None,
                    "curve_paths": curve_paths,
                    "npy_paths": npy_paths,
                    "hist_path": str(hist_path) if hist_path is not None else None,
                    "hist_full_path": str(hist_full_path) if hist_full_path is not None else None,
                    "block_size": self.block_size,
                    "token_stride": self.token_stride,
                    "channel_stride": self.channel_stride,
                    "weight_stride": self.weight_stride,
                    "stats": stats,
                }
            )
        except Exception as exc:
            self._records.append(
                {
                    "layer_name": layer_name,
                    "call_id": int(recorded_call),
                    "inference_count": self._safe_number(inference_count),
                    "error": repr(exc),
                }
            )

    def flush(self):
        if self._flush_pos >= len(self._records):
            return
        jsonl_path = self.output_dir / "linear_outlier_report.jsonl"
        csv_path = self.output_dir / "linear_outlier_report.csv"
        new_records = self._records[self._flush_pos :]

        with jsonl_path.open("a", encoding="utf-8") as f:
            for record in new_records:
                f.write(json.dumps(self._json_safe(record), ensure_ascii=False) + "\n")

        csv_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "layer_name",
                    "call_id",
                    "inference_count",
                    "tensor",
                    "abs_p50",
                    "abs_p95",
                    "abs_p99",
                    "abs_p999",
                    "abs_max",
                    "max_over_p99",
                    "channel_absmax_max_over_median",
                    "top_channels_by_absmax",
                    "block_max_over_p95_p50",
                    "block_max_over_p95_p95",
                    "block_max_over_p95_max",
                    "block_max_over_median_p50",
                    "block_max_over_median_p95",
                    "block_max_over_median_max",
                    "dominant_frac_p95",
                    "dominant_frac_max",
                    "small_frac_p95",
                    "plot_path",
                ],
            )
            if not csv_exists:
                writer.writeheader()
            for record in new_records:
                for tensor_name, stats in record.get("stats", {}).items():
                    if stats is None:
                        continue
                    row = {
                        "layer_name": record.get("layer_name"),
                        "call_id": record.get("call_id"),
                        "inference_count": record.get("inference_count"),
                        "tensor": tensor_name,
                        "plot_path": record.get("plot_path"),
                    }
                    for key in writer.fieldnames:
                        if key in stats:
                            row[key] = stats[key]
                    row["top_channels_by_absmax"] = json.dumps(
                        stats.get("top_channels_by_absmax", [])
                    )
                    writer.writerow(self._json_safe(row))

        self._flush_pos = len(self._records)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _activation_plot_array(self, tensor):
        if tensor is None or not torch.is_tensor(tensor) or getattr(tensor, "is_meta", False):
            return None
        with torch.no_grad():
            t = tensor.detach()
            if t.dim() == 3:
                sampled = t[:, :: self.token_stride, :: self.channel_stride]
                arr = sampled.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()
                return arr.mean(axis=0)
            if t.dim() == 2:
                sampled = t[:: self.token_stride, :: self.channel_stride]
                return sampled.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()
            if t.dim() == 1:
                sampled = t[:: self.channel_stride].reshape(1, -1)
                return sampled.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()
            sampled = t.reshape(-1, t.shape[-1])[:: self.token_stride, :: self.channel_stride]
            return sampled.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()

    def _activation_signed_array(self, tensor):
        if tensor is None or not torch.is_tensor(tensor) or getattr(tensor, "is_meta", False):
            return None
        with torch.no_grad():
            t = tensor.detach()
            if t.dim() == 3:
                sampled = t[:, :: self.token_stride, :: self.channel_stride]
                sampled = sampled.reshape(-1, sampled.shape[-1])
                return sampled.to(device="cpu", dtype=torch.float32, copy=True).numpy()
            if t.dim() == 2:
                sampled = t[:: self.token_stride, :: self.channel_stride]
                return sampled.to(device="cpu", dtype=torch.float32, copy=True).numpy()
            if t.dim() == 1:
                sampled = t[:: self.channel_stride].reshape(1, -1)
                return sampled.to(device="cpu", dtype=torch.float32, copy=True).numpy()
            if t.dim() == 0:
                return None
            sampled = t.reshape(-1, t.shape[-1])[:: self.token_stride, :: self.channel_stride]
            return sampled.to(device="cpu", dtype=torch.float32, copy=True).numpy()

    def _activation_stats_matrix(self, tensor):
        if tensor is None or not torch.is_tensor(tensor) or getattr(tensor, "is_meta", False):
            return None
        with torch.no_grad():
            t = tensor.detach().reshape(-1, tensor.shape[-1])
            return t.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()

    def _weight_plot_array(self, weight):
        if weight is None or not torch.is_tensor(weight) or getattr(weight, "is_meta", False):
            return None
        with torch.no_grad():
            sampled = weight.detach()[:: self.weight_stride, :: self.weight_stride]
            return sampled.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()

    def _weight_signed_array(self, weight):
        if weight is None or not torch.is_tensor(weight) or getattr(weight, "is_meta", False):
            return None
        with torch.no_grad():
            sampled = weight.detach()[:: self.weight_stride, :: self.weight_stride]
            return sampled.to(device="cpu", dtype=torch.float32, copy=True).numpy()

    def _weight_stats_matrix(self, weight):
        if weight is None or not torch.is_tensor(weight) or getattr(weight, "is_meta", False):
            return None
        with torch.no_grad():
            return weight.detach().to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()

    def _delta_w_sample_array(self, u, v, scale):
        if u is None or v is None or scale is None:
            return None
        if not torch.is_tensor(u) or not torch.is_tensor(v):
            return None
        if getattr(u, "is_meta", False) or getattr(v, "is_meta", False):
            return None
        with torch.no_grad():
            out_idx = torch.arange(0, u.shape[0], self.weight_stride, device=u.device)
            in_idx = torch.arange(0, v.shape[0], self.weight_stride, device=v.device)
            u_sample = u.detach().index_select(0, out_idx)
            v_sample = v.detach().index_select(0, in_idx)
            delta = u_sample.matmul(v_sample.t()).mul(scale)
            return delta.to(device="cpu", dtype=torch.float32, copy=True).abs_().numpy()

    def _delta_w_signed_sample_array(self, u, v, scale):
        if u is None or v is None or scale is None:
            return None
        if not torch.is_tensor(u) or not torch.is_tensor(v):
            return None
        if getattr(u, "is_meta", False) or getattr(v, "is_meta", False):
            return None
        with torch.no_grad():
            out_idx = torch.arange(0, u.shape[0], self.weight_stride, device=u.device)
            in_idx = torch.arange(0, v.shape[0], self.weight_stride, device=v.device)
            u_sample = u.detach().index_select(0, out_idx)
            v_sample = v.detach().index_select(0, in_idx)
            delta = u_sample.matmul(v_sample.t()).mul(scale)
            return delta.to(device="cpu", dtype=torch.float32, copy=True).numpy()

    def _tensor_stats(self, tensor_name, matrix, channel_axis_values=None):
        if matrix is None or matrix.size == 0:
            return None
        arr = np.asarray(matrix, dtype=np.float32)
        flat = arr.reshape(-1)
        abs_p50 = self._percentile(flat, 50)
        abs_p95 = self._percentile(flat, 95)
        abs_p99 = self._percentile(flat, 99)
        abs_p999 = self._percentile(flat, 99.9)
        abs_max = self._float(np.max(flat))

        channel_absmax = np.max(arr, axis=0)
        channel_median = self._float(np.median(channel_absmax))
        top_channels = self._top_channels(channel_absmax, channel_axis_values)
        block_stats = self._block_stats(arr)

        stats = {
            "tensor": tensor_name,
            "shape": list(arr.shape),
            "abs_p50": abs_p50,
            "abs_p95": abs_p95,
            "abs_p99": abs_p99,
            "abs_p999": abs_p999,
            "abs_max": abs_max,
            "max_over_p99": self._ratio(abs_max, abs_p99),
            "channel_absmax_max_over_median": self._ratio(
                self._float(np.max(channel_absmax)), channel_median
            ),
            "top_channels_by_absmax": top_channels,
        }
        stats.update(block_stats)
        return stats

    def _block_stats(self, arr):
        max_over_p95 = []
        max_over_median = []
        dominant_frac = []
        small_frac = []
        width = arr.shape[-1]
        for start in range(0, width, self.block_size):
            block = arr[:, start : start + self.block_size].reshape(-1)
            if block.size == 0:
                continue
            block_max = self._float(np.max(block))
            block_p95 = self._percentile(block, 95)
            block_median = self._float(np.median(block))
            block_sum = self._float(np.sum(block))
            max_over_p95.append(self._ratio(block_max, block_p95))
            max_over_median.append(self._ratio(block_max, block_median))
            dominant_frac.append(self._ratio(block_max, block_sum))
            small_frac.append(self._float(np.mean(block < (block_max / 16.0))))

        return {
            "block_max_over_p95_p50": self._list_percentile(max_over_p95, 50),
            "block_max_over_p95_p95": self._list_percentile(max_over_p95, 95),
            "block_max_over_p95_max": self._list_max(max_over_p95),
            "block_max_over_median_p50": self._list_percentile(max_over_median, 50),
            "block_max_over_median_p95": self._list_percentile(max_over_median, 95),
            "block_max_over_median_max": self._list_max(max_over_median),
            "dominant_frac_p95": self._list_percentile(dominant_frac, 95),
            "dominant_frac_max": self._list_max(dominant_frac),
            "small_frac_p95": self._list_percentile(small_frac, 95),
        }

    def _top_channels(self, channel_absmax, channel_axis_values):
        if channel_absmax.size == 0:
            return []
        top_k = min(8, channel_absmax.size)
        order = np.argsort(channel_absmax)[-top_k:][::-1]
        channels = []
        for idx in order:
            channel_idx = int(channel_axis_values[idx]) if channel_axis_values is not None else int(idx)
            channels.append({"index": channel_idx, "absmax": self._float(channel_absmax[idx])})
        return channels

    def _save_plot(self, layer_name, recorded_call, inference_count, x_plot, dx_plot, w_plot, delta_w_plot):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 10))
        panels = [
            ("|x|", x_plot, "Channel", "Token"),
            ("|dx|", dx_plot, "Channel", "Token"),
            ("|W|", w_plot, "In Channel", "Out Channel"),
            ("|delta_W|", delta_w_plot, "In Channel", "Out Channel"),
        ]
        safe_name = self._safe_filename(layer_name)
        for i, (name, data, x_label, y_label) in enumerate(panels, start=1):
            ax = fig.add_subplot(2, 2, i, projection="3d")
            title = f"{layer_name} call={recorded_call} inference={inference_count} {name}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel("Absolute value")
            if data is None or data.size == 0:
                ax.text2D(0.25, 0.5, "not available", transform=ax.transAxes)
                continue
            y = np.arange(data.shape[0])
            x = np.arange(data.shape[1])
            xx, yy = np.meshgrid(x, y)
            ax.plot_surface(xx, yy, data, cmap="viridis", linewidth=0, antialiased=False)
        fig.tight_layout()
        png_path = self.output_dir / f"{safe_name}_call{recorded_call}_inf{inference_count}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path

    def _save_simple_curves(
        self,
        layer_name,
        recorded_call,
        inference_count,
        x_plot,
        dx_plot,
        w_plot,
        delta_w_plot,
    ):
        curve_paths = {}
        w_dw_path = self._save_w_dw_curve(
            layer_name,
            recorded_call,
            inference_count,
            w_plot,
            delta_w_plot,
        )
        if w_dw_path is not None:
            curve_paths["w_dw_curve"] = str(w_dw_path)

        x_dx_path = self._save_x_dx_curve(
            layer_name,
            recorded_call,
            inference_count,
            x_plot,
            dx_plot,
        )
        if x_dx_path is not None:
            curve_paths["x_dx_curve"] = str(x_dx_path)
        return curve_paths

    def _save_signed_npy(
        self,
        layer_name,
        recorded_call,
        inference_count,
        x_signed,
        dx_signed,
        w_signed,
        dw_signed,
    ):
        npy_dir = self.output_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_filename(layer_name)
        npy_paths = {}
        arrays = (
            ("x", x_signed),
            ("dx", dx_signed),
            ("w", w_signed),
            ("dw", dw_signed),
        )
        for tensor_name, array in arrays:
            if array is None:
                continue
            npy_path = (
                npy_dir
                / f"{safe_name}_call{recorded_call}_inf{inference_count}_{tensor_name}.npy"
            )
            np.save(npy_path, array, allow_pickle=False)
            npy_paths[tensor_name] = str(npy_path)
        return npy_paths

    def _save_hist_plot(
        self,
        layer_name,
        recorded_call,
        inference_count,
        x_signed,
        dx_signed,
        w_signed,
        dw_signed,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        safe_name = self._safe_filename(layer_name)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        try:
            arrays = (
                ("x", x_signed),
                ("dx", dx_signed),
                ("w", w_signed),
                ("dw", dw_signed),
            )
            for ax, (tensor_name, array) in zip(axes.reshape(-1), arrays):
                if array is None or array.size == 0:
                    ax.text(0.25, 0.5, "not available", transform=ax.transAxes)
                    ax.set_title(f"{tensor_name} not available", fontsize=9)
                    continue

                arr = np.asarray(array, dtype=np.float32)
                values = arr.reshape(-1)
                if values.size > 1_000_000:
                    idx = np.linspace(0, values.size - 1, num=1_000_000, dtype=np.int64)
                    values = values[idx]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    ax.text(0.25, 0.5, "no finite values", transform=ax.transAxes)
                    ax.set_title(f"{tensor_name} shape={tuple(arr.shape)} no finite values", fontsize=9)
                    continue

                abs_values = np.abs(values)
                abs_max = self._float(np.max(abs_values))
                p99_abs = self._percentile(abs_values, 99)
                robust_limit = self._percentile(abs_values, 99.9)
                ax.hist(values, bins=200)
                if robust_limit is not None and robust_limit > 0:
                    ax.set_xlim(-robust_limit, robust_limit)
                ax.set_title(
                    f"{tensor_name} shape={tuple(arr.shape)}\n"
                    f"abs_max={abs_max:.6g} p99_abs={p99_abs:.6g}",
                    fontsize=9,
                )
                ax.set_xlabel("signed value")
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.2)

            fig.suptitle(
                f"{layer_name} call={recorded_call} inference={inference_count} signed histograms",
                fontsize=10,
            )
            fig.tight_layout()
            png_path = (
                self.output_dir
                / f"{safe_name}_call{recorded_call}_inf{inference_count}_hist.png"
            )
            fig.savefig(png_path, dpi=150)
            return png_path
        finally:
            plt.close(fig)
            
    def _save_hist_full_plot(
        self,
        layer_name,
        recorded_call,
        inference_count,
        x_signed,
        dx_signed,
        w_signed,
        dw_signed,
    ):
        import matplotlib
    
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    
        safe_name = self._safe_filename(layer_name)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        try:
            arrays = (
                ("x", x_signed),
                ("dx", dx_signed),
                ("w", w_signed),
                ("dw", dw_signed),
            )
    
            for ax, (tensor_name, array) in zip(axes.reshape(-1), arrays):
                if array is None or array.size == 0:
                    ax.text(0.25, 0.5, "not available", transform=ax.transAxes)
                    ax.set_title(f"{tensor_name} not available", fontsize=9)
                    continue
    
                arr = np.asarray(array, dtype=np.float32)
                values = arr.reshape(-1)
    
                # 如果你真的想包含所有 sampled elements，就不要下采样。
                # 这里先保留所有 values，只去掉 NaN/Inf。
                values = values[np.isfinite(values)]
    
                if values.size == 0:
                    ax.text(0.25, 0.5, "no finite values", transform=ax.transAxes)
                    ax.set_title(f"{tensor_name} shape={tuple(arr.shape)} no finite values", fontsize=9)
                    continue
    
                abs_values = np.abs(values)
                abs_max = self._float(np.max(abs_values))
                p99_abs = self._percentile(abs_values, 99)
                p999_abs = self._percentile(abs_values, 99.9)
    
                if abs_max is not None and abs_max > 0:
                    ax.hist(values, bins=800, range=(-abs_max, abs_max))
                    ax.set_xlim(-abs_max, abs_max)
                else:
                    ax.hist(values, bins=800)
    
                # 画 p99 / p99.9 的参考线，方便看中心和 outlier 差距
                if p99_abs is not None and p99_abs > 0:
                    ax.axvline(p99_abs, linestyle="--", linewidth=0.8)
                    ax.axvline(-p99_abs, linestyle="--", linewidth=0.8)
                if p999_abs is not None and p999_abs > 0:
                    ax.axvline(p999_abs, linestyle=":", linewidth=0.8)
                    ax.axvline(-p999_abs, linestyle=":", linewidth=0.8)
    
                ax.set_title(
                    f"{tensor_name} shape={tuple(arr.shape)}\n"
                    f"abs_max={abs_max:.6g} p99_abs={p99_abs:.6g} p99.9_abs={p999_abs:.6g}",
                    fontsize=9,
                )
                ax.set_xlabel("signed value")
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.2)
    
            fig.suptitle(
                f"{layer_name} call={recorded_call} inference={inference_count} full-range signed histograms",
                fontsize=10,
            )
            fig.tight_layout()
            png_path = (
                self.output_dir
                / f"{safe_name}_call{recorded_call}_inf{inference_count}_hist_full.png"
            )
            fig.savefig(png_path, dpi=150)
            return png_path
        finally:
            plt.close(fig)

    def _norm_by_median(self, values):
        values = np.asarray(values, dtype=np.float32)
        median = float(np.median(values)) if values.size else 0.0
        if not np.isfinite(median) or median <= 0:
            median = 1e-12
        return values / median

    def _corr_safe(self, a, b):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        n = min(a.size, b.size)
        if n < 2:
            return None
        a = a[:n]
        b = b[:n]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < 2:
            return None
        a = a[mask]
        b = b[mask]
        if float(np.std(a)) <= 0 or float(np.std(b)) <= 0:
            return None
        return self._float(np.corrcoef(a, b)[0, 1])

    def _topk_overlap(self, a, b, k=16):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        n = min(a.size, b.size)
        if k <= 0 or n <= 0:
            return None
        a = a[:n]
        b = b[:n]
        mask = np.isfinite(a) & np.isfinite(b)
        valid_idx = np.nonzero(mask)[0]
        if valid_idx.size == 0:
            return None
        top_k = min(k, valid_idx.size)
        a_order = valid_idx[np.argsort(a[valid_idx])[-top_k:]]
        b_order = valid_idx[np.argsort(b[valid_idx])[-top_k:]]
        overlap = len(set(a_order.tolist()).intersection(b_order.tolist()))
        return self._float(overlap / float(k))

    def _save_w_dw_curve(self, layer_name, recorded_call, inference_count, w_plot, delta_w_plot):
        if w_plot is None or w_plot.size == 0:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        safe_name = self._safe_filename(layer_name)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        try:
            w_col = np.max(np.abs(np.asarray(w_plot, dtype=np.float32)), axis=0)
            w_col_norm = self._norm_by_median(w_col)
            ax.plot(np.arange(w_col_norm.size), w_col_norm, label="w", linewidth=1.3)

            if delta_w_plot is not None and delta_w_plot.size > 0:
                dw_col = np.max(np.abs(np.asarray(delta_w_plot, dtype=np.float32)), axis=0)
                dw_col_norm = self._norm_by_median(dw_col)
                ax.plot(np.arange(dw_col_norm.size), dw_col_norm, label="dw", linewidth=1.3)

            ax.set_xlabel("sampled input channel")
            ax.set_ylabel("column absmax / median")
            ax.set_title(
                f"{layer_name} call={recorded_call} inference={inference_count} w/dw curve",
                fontsize=9,
            )
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            png_path = (
                self.output_dir
                / f"{safe_name}_call{recorded_call}_inf{inference_count}_w_dw_curve.png"
            )
            fig.savefig(png_path, dpi=150)
            return png_path
        finally:
            plt.close(fig)

    def _save_x_dx_curve(self, layer_name, recorded_call, inference_count, x_plot, dx_plot):
        if x_plot is None or dx_plot is None or x_plot.size == 0 or dx_plot.size == 0:
            return None

        x_arr = np.asarray(x_plot, dtype=np.float32)
        dx_arr = np.asarray(dx_plot, dtype=np.float32)
        if x_arr.ndim != 2 or dx_arr.ndim != 2:
            return None
        t = min(x_arr.shape[0], dx_arr.shape[0])
        c = min(x_arr.shape[1], dx_arr.shape[1])
        if t <= 0 or c <= 0:
            return None

        x_arr = np.abs(x_arr[:t, :c])
        dx_arr = np.abs(dx_arr[:t, :c])
        x_ch_norm = self._norm_by_median(np.max(x_arr, axis=0))
        dx_ch_norm = self._norm_by_median(np.max(dx_arr, axis=0))
        x_tok_norm = self._norm_by_median(np.max(x_arr, axis=1))
        dx_tok_norm = self._norm_by_median(np.max(dx_arr, axis=1))

        channel_corr = self._corr_safe(x_ch_norm, dx_ch_norm)
        token_corr = self._corr_safe(x_tok_norm, dx_tok_norm)
        top16_overlap = self._topk_overlap(x_ch_norm, dx_ch_norm, k=16)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        safe_name = self._safe_filename(layer_name)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        try:
            axes[0].plot(np.arange(c), x_ch_norm, label="x", linewidth=1.3)
            axes[0].plot(np.arange(c), dx_ch_norm, label="dx", linewidth=1.3)
            axes[0].set_xlabel("sampled channel")
            axes[0].set_ylabel("channel absmax / median")
            axes[0].set_title("channel curve", fontsize=9)
            axes[0].legend()
            axes[0].grid(True, alpha=0.25)

            axes[1].plot(np.arange(t), x_tok_norm, label="x", linewidth=1.3)
            axes[1].plot(np.arange(t), dx_tok_norm, label="dx", linewidth=1.3)
            axes[1].set_xlabel("token")
            axes[1].set_ylabel("token absmax / median")
            axes[1].set_title("token curve", fontsize=9)
            axes[1].legend()
            axes[1].grid(True, alpha=0.25)

            metrics = []
            if channel_corr is not None:
                metrics.append(f"channel corr={channel_corr:.4f}")
            if token_corr is not None:
                metrics.append(f"token corr={token_corr:.4f}")
            if top16_overlap is not None:
                metrics.append(f"top16 overlap={top16_overlap:.4f}")
            title = "x/dx curve"
            if metrics:
                title += " | " + " | ".join(metrics)
            fig.suptitle(f"{layer_name} call={recorded_call} inference={inference_count} {title}", fontsize=9)
            fig.tight_layout()
            png_path = (
                self.output_dir
                / f"{safe_name}_call{recorded_call}_inf{inference_count}_x_dx_curve.png"
            )
            fig.savefig(png_path, dpi=150)
            return png_path
        finally:
            plt.close(fig)

    def _safe_filename(self, text):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]

    def _percentile(self, values, q):
        if values is None or len(values) == 0:
            return None
        return self._float(np.percentile(values, q))

    def _list_percentile(self, values, q):
        values = [x for x in values if x is not None]
        if not values:
            return None
        return self._float(np.percentile(np.asarray(values, dtype=np.float32), q))

    def _list_max(self, values):
        values = [x for x in values if x is not None]
        if not values:
            return None
        return self._float(np.max(np.asarray(values, dtype=np.float32)))

    def _ratio(self, numerator, denominator):
        if denominator is None or denominator <= 0:
            if numerator is None or numerator == 0:
                return 0.0
            return None
        return self._float(numerator / denominator)

    def _float(self, value):
        value = float(value)
        if np.isfinite(value):
            return value
        return None

    def _safe_number(self, value):
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            try:
                return self._float(value)
            except Exception:
                return str(value)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return self._float(value)
        if isinstance(value, Path):
            return str(value)
        return value


def attach_linear_outlier_profiler(model, profiler):
    diff_classes = _diff_linear_classes()

    for layer_name, module in model.named_modules():
        class_name = module.__class__.__name__
        is_diff_linear = class_name in ("diffLinear", "QdiffLinear")
        if diff_classes and isinstance(module, diff_classes):
            is_diff_linear = True

        if is_diff_linear:
            module.outlier_profiler = profiler
            if not getattr(module, "layer_name", None):
                module.layer_name = layer_name
            continue

        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(_make_linear_hook(layer_name, profiler))
            profiler._handles.append(handle)

    return profiler


def _make_linear_hook(layer_name, profiler):
    def hook(module, inputs, output):
        if not inputs:
            return
        input_tensor = inputs[0]
        profiler.record_forward(layer_name, module, input_tensor)

    return hook


def _diff_linear_classes():
    try:
        from diff_fake_quant_mx import QdiffLinear, diffLinear

        return (diffLinear, QdiffLinear)
    except Exception:
        return ()