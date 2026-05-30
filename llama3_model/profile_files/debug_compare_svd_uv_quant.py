#!/usr/bin/env python3
"""Compare SVD-LoRA compensation magnitude with LOZO low-rank perturbations.

Example:
  python llama3_model/debug_compare_svd_uv_quant.py \
    --checkpoint-path outputs/svd_lora/adapter_model.bin \
    --rank-r 2 \
    --zo-eps 1e-3 \
    --layers q_proj,k_proj \
    --output-dir debug_svd_uv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SVD_ZOO_ROOT = REPO_ROOT / "SVD-ZOO-Quant"
if str(SVD_ZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(SVD_ZOO_ROOT))

torch = None
_quantize_groupwise_fp = None


def quantize_activation_nvfp8(t: torch.Tensor, group_size=16):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, help="PEFT-style SVD-LoRA checkpoint file or directory")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--layers", default="", help="Comma-separated layer-name substrings; empty means all")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--rank-r", type=int, default=2, help="LOZO perturbation rank")
    parser.add_argument("--zo-eps", type=float, default=1e-3, help="Scale applied to the random LOZO perturbation")
    parser.add_argument("--raw-perturb", action="store_true", help="Use P=u@v.T instead of P=zo_eps*u@v.T")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=16, help="NVFP8 group size")
    parser.add_argument("--sample-elements", type=int, default=0, help="Sample elements after dense S/P construction; 0 uses all")
    parser.add_argument("--hist-bins", type=int, default=160, help="Bins for per-layer distribution plots")
    parser.add_argument("--output-dir", default="debug_svd_uv_quant")
    return parser.parse_args()


def load_state_dict(path: str):
    checkpoint_path = Path(path)
    checkpoint_file = checkpoint_path / "adapter_model.bin" if checkpoint_path.is_dir() else checkpoint_path
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    state = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def discover_layers(state_dict, adapter_name: str, filters):
    suffix_a = f".lora_A.{adapter_name}.weight"
    suffix_b = f".lora_B.{adapter_name}.weight"
    layers = []
    for key in sorted(state_dict):
        if not key.endswith(suffix_a):
            continue
        name = key[: -len(suffix_a)]
        if filters and not any(item in name for item in filters):
            continue
        if f"{name}{suffix_b}" in state_dict:
            layers.append(name)
    return layers


def tensor_stats(x: torch.Tensor):
    abs_x = x.abs()
    return float(abs_x.mean().item()), float(abs_x.max().item())


def distribution_stats(x: torch.Tensor):
    flat = x.detach().float().reshape(-1)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp(min=1e-30)
    centered = flat - mean
    skew = (centered.pow(3).mean() / std.pow(3)).item()
    kurt = (centered.pow(4).mean() / std.pow(4)).item()
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "zero": float((flat == 0).float().mean().item()),
        "pos": float((flat > 0).float().mean().item()),
        "neg": float((flat < 0).float().mean().item()),
        "skew": float(skew),
        "kurt": float(kurt),
    }


def safe_filename(name: str):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def maybe_sample(*tensors, max_elements: int, seed: int):
    if max_elements <= 0:
        return tensors
    numel = tensors[0].numel()
    if numel <= max_elements:
        return tensors
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = torch.randperm(numel, generator=generator)[:max_elements]
    return tuple(t.reshape(-1)[index] for t in tensors)


def plot_distribution_pair(name, p_eval, q_delta_eval, args, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib is not available, skipping distribution plot for {name}: {exc}")
        return

    p_flat = p_eval.detach().float().reshape(-1).cpu()
    q_delta_flat = q_delta_eval.detach().float().reshape(-1).cpu()
    p_stats = distribution_stats(p_flat)
    q_stats = distribution_stats(q_delta_flat)

    def add_stats_box(ax, stats):
        text = (
            f"mean {stats['mean']:.3e}\n"
            f"std  {stats['std']:.3e}\n"
            f"zero {stats['zero']:.3f}\n"
            f"pos  {stats['pos']:.3f}\n"
            f"neg  {stats['neg']:.3f}\n"
            f"skew {stats['skew']:.3f}\n"
            f"kurt {stats['kurt']:.3f}"
        )
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Distribution comparison of LOZO perturbations under NVFP8 fake quantization\n"
        f"{name}",
        fontsize=11,
    )
    for ax, data, title, stats in [
        (axes[0], p_flat, "raw P", p_stats),
        (axes[1], q_delta_flat, "Q(S+P)-Q(S)", q_stats),
    ]:
        ax.hist(data.numpy(), bins=args.hist_bins, density=True, color="#4C7FA3", alpha=0.95)
        ax.axvline(0.0, color="#B22222", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        add_stats_box(ax, stats)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"distribution_{safe_filename(name)}.png", dpi=200)
    plt.close(fig)


def analyze_layer(name, lora_a, lora_b, args, layer_index, output_dir: Path):
    # PEFT LoRA checkpoint: A=[svd_rank,in], B=[out,svd_rank].
    s_matrix = lora_b.float().matmul(lora_a.float())
    out_features, in_features = s_matrix.shape

    generator = torch.Generator(device="cpu").manual_seed(args.seed + layer_index)
    u = torch.randn(out_features, args.rank_r, generator=generator)
    v = torch.randn(in_features, args.rank_r, generator=generator)
    p_matrix = u.matmul(v.t())
    if not args.raw_perturb:
        p_matrix = p_matrix * args.zo_eps

    q_s = quantize_activation_nvfp8(s_matrix, group_size=args.group_size)
    q_sp = quantize_activation_nvfp8(s_matrix + p_matrix, group_size=args.group_size)
    q_delta = q_sp - q_s

    s_eval, p_eval, q_s_eval, q_delta_eval = maybe_sample(
        s_matrix,
        p_matrix,
        q_s,
        q_delta,
        max_elements=args.sample_elements,
        seed=args.seed + 100_000 + layer_index,
    )

    s_mean, s_max = tensor_stats(s_eval)
    p_mean, p_max = tensor_stats(p_eval)
    q_s_mean, q_s_max = tensor_stats(q_s_eval)
    q_delta_mean, q_delta_max = tensor_stats(q_delta_eval)
    zero_frac = float((q_delta_eval.abs() <= 1e-12).float().mean().item())
    plot_distribution_pair(name, p_eval, q_delta_eval, args, output_dir)

    eps = 1e-30
    return {
        "layer": name,
        "out_features": out_features,
        "in_features": in_features,
        "svd_lora_rank": lora_a.shape[0],
        "rank_r": args.rank_r,
        "perturb_scale": 1.0 if args.raw_perturb else args.zo_eps,
        "mean_abs_S": s_mean,
        "max_abs_S": s_max,
        "mean_abs_P": p_mean,
        "max_abs_P": p_max,
        "mean_ratio_P_over_S": p_mean / max(s_mean, eps),
        "max_ratio_P_over_S": p_max / max(s_max, eps),
        "mean_abs_QS": q_s_mean,
        "max_abs_QS": q_s_max,
        "mean_abs_Q_delta": q_delta_mean,
        "max_abs_Q_delta": q_delta_max,
        "zero_frac_Q_delta": zero_frac,
        "mean_ratio_Q_delta_over_P": q_delta_mean / max(p_mean, eps),
        "max_ratio_Q_delta_over_P": q_delta_max / max(p_max, eps),
    }


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib is not available, skipping plots: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["layer"].split(".")[-1] + f"#{i}" for i, row in enumerate(rows)]
    x = list(range(len(rows)))

    def bar_plot(keys, title, filename, logy=True):
        width = 0.8 / len(keys)
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.25), 5))
        for offset, key in enumerate(keys):
            values = [row[key] for row in rows]
            positions = [i + (offset - (len(keys) - 1) / 2) * width for i in x]
            ax.bar(positions, values, width=width, label=key)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("absolute magnitude")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    bar_plot(
        ["mean_abs_S", "mean_abs_P", "mean_abs_QS", "mean_abs_Q_delta"],
        "Mean absolute magnitude",
        "mean_abs_magnitudes.png",
    )
    bar_plot(
        ["max_abs_S", "max_abs_P", "max_abs_QS", "max_abs_Q_delta"],
        "Max absolute magnitude",
        "max_abs_magnitudes.png",
    )

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.25), 4))
    ax.bar(x, [row["zero_frac_Q_delta"] for row in rows])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.set_title("Fraction where Q(S+P)-Q(S) is approximately zero")
    fig.tight_layout()
    fig.savefig(output_dir / "zero_fraction_q_delta.png", dpi=200)
    plt.close(fig)


def main():
    global torch, _quantize_groupwise_fp
    args = parse_args()
    import torch as torch_module
    from smoothquant.fake_quant import _quantize_groupwise_fp as quantize_groupwise_fp

    torch = torch_module
    _quantize_groupwise_fp = quantize_groupwise_fp

    state = load_state_dict(args.checkpoint_path)
    filters = [item.strip() for item in args.layers.split(",") if item.strip()]
    layers = discover_layers(state, args.adapter_name, filters)
    if args.max_layers > 0:
        layers = layers[: args.max_layers]
    if not layers:
        raise ValueError("No LoRA/SVD layers matched the requested filters")

    suffix_a = f".lora_A.{args.adapter_name}.weight"
    suffix_b = f".lora_B.{args.adapter_name}.weight"
    output_dir = Path(args.output_dir)
    rows = []
    for idx, name in enumerate(layers):
        row = analyze_layer(
            name,
            state[f"{name}{suffix_a}"],
            state[f"{name}{suffix_b}"],
            args,
            idx,
            output_dir,
        )
        rows.append(row)
        print(
            f"{name}: mean|S|={row['mean_abs_S']:.3e}, mean|P|={row['mean_abs_P']:.3e}, "
            f"mean|Qdelta|={row['mean_abs_Q_delta']:.3e}, zero_frac={row['zero_frac_Q_delta']:.3f}"
        )

    write_csv(rows, output_dir / "svd_uv_quant_stats.csv")
    plot_rows(rows, output_dir)
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
