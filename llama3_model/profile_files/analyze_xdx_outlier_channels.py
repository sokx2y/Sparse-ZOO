import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def layer_prefix_from_npy(path):
    name = Path(path).name
    for suffix in ["_x.npy", "_dx.npy"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def compute_outlier_channel_stats(arr, q=99.9, channel_stride=1, topks=(1, 2, 4, 8, 16, 32)):
    """
    arr: [tokens, sampled_channels]
    Define outliers globally over the entire tensor, then count where they appear by channel.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])

    abs_arr = np.abs(arr)
    flat = abs_arr.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None

    global_median = float(np.median(flat))
    threshold = float(np.percentile(flat, q))

    mask = abs_arr > threshold
    total_outliers = int(mask.sum())

    # per-channel: how many globally-defined outlier elements fall into this channel
    outlier_count = mask.sum(axis=0).astype(np.int64)
    num_tokens = arr.shape[0]

    if total_outliers > 0:
        outlier_share = outlier_count / float(total_outliers)
    else:
        outlier_share = np.zeros_like(outlier_count, dtype=np.float64)

    outlier_rate = outlier_count / float(max(num_tokens, 1))

    ch_absmax = abs_arr.max(axis=0)
    ch_absmean = abs_arr.mean(axis=0)

    # Sort channels by outlier_count/share, not by sum magnitude.
    order = np.argsort(outlier_count)[::-1]

    topk_cover = {}
    for k in topks:
        kk = min(k, len(order))
        if total_outliers > 0:
            topk_cover[k] = float(outlier_count[order[:kk]].sum() / total_outliers)
        else:
            topk_cover[k] = 0.0

    rows = []
    for rank, c in enumerate(order):
        rows.append(
            {
                "rank": rank + 1,
                "sampled_channel_idx": int(c),
                "approx_original_channel_idx": int(c * channel_stride),
                "outlier_count": int(outlier_count[c]),
                "outlier_share": float(outlier_share[c]),
                "outlier_rate": float(outlier_rate[c]),
                "ch_absmax": float(ch_absmax[c]),
                "ch_absmean": float(ch_absmean[c]),
                "ch_absmax_over_global_median": float(ch_absmax[c] / (global_median + 1e-12)),
            }
        )

    summary = {
        "shape": tuple(arr.shape),
        "q": q,
        "threshold": threshold,
        "global_abs_median": global_median,
        "global_abs_p99": float(np.percentile(flat, 99)),
        "global_abs_p999": float(np.percentile(flat, 99.9)),
        "global_abs_max": float(np.max(flat)),
        "total_elements": int(abs_arr.size),
        "total_outliers": total_outliers,
        "outlier_fraction": float(total_outliers / max(abs_arr.size, 1)),
        "topk_cover": topk_cover,
        "outlier_count": outlier_count,
        "outlier_share": outlier_share,
        "outlier_rate": outlier_rate,
        "order": order,
    }
    return summary, pd.DataFrame(rows)


def topk_overlap(order_a, order_b, k=16):
    k = min(k, len(order_a), len(order_b))
    if k <= 0:
        return np.nan
    a = set(order_a[:k].tolist())
    b = set(order_b[:k].tolist())
    return len(a & b) / float(k)


def plot_x_dx_outlier_channels(prefix, x_stats, dx_stats, out_dir, channel_stride=1):
    x_summary, _ = x_stats
    dx_summary, _ = dx_stats

    x_share = x_summary["outlier_share"]
    dx_share = dx_summary["outlier_share"]
    x_order = x_summary["order"]
    dx_order = dx_summary["order"]

    c = min(len(x_share), len(dx_share))
    sampled_channels = np.arange(c)
    approx_orig_channels = sampled_channels * channel_stride

    x_share = x_share[:c]
    dx_share = dx_share[:c]

    # sorted cumulative coverage
    x_sorted = x_summary["outlier_count"][x_order]
    dx_sorted = dx_summary["outlier_count"][dx_order]
    if x_sorted.sum() > 0:
        x_cum = np.cumsum(x_sorted) / x_sorted.sum()
    else:
        x_cum = np.zeros_like(x_sorted, dtype=np.float64)
    if dx_sorted.sum() > 0:
        dx_cum = np.cumsum(dx_sorted) / dx_sorted.sum()
    else:
        dx_cum = np.zeros_like(dx_sorted, dtype=np.float64)

    overlap16 = topk_overlap(x_order, dx_order, k=16)
    overlap32 = topk_overlap(x_order, dx_order, k=32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(approx_orig_channels, x_share, label="x", linewidth=1.2)
    axes[0].plot(approx_orig_channels, dx_share, label="dx", linewidth=1.2)
    axes[0].set_xlabel("approx original channel index")
    axes[0].set_ylabel("share of global outliers")
    axes[0].set_title("Where do global outlier elements fall?")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(np.arange(1, len(x_cum) + 1), x_cum, label="x", linewidth=1.2)
    axes[1].plot(np.arange(1, len(dx_cum) + 1), dx_cum, label="dx", linewidth=1.2)
    axes[1].set_xlabel("top-k channels ranked by outlier count")
    axes[1].set_ylabel("cumulative fraction of global outliers")
    axes[1].set_title("Outlier concentration across channels")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    title = (
        f"{prefix}\n"
        f"x threshold={x_summary['threshold']:.4g}, dx threshold={dx_summary['threshold']:.4g}, "
        f"top16 overlap={overlap16:.3f}, top32 overlap={overlap32:.3f}"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"{safe_name(prefix)}_xdx_outlier_channels.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy_dir",
        type=str,
        required=True,
        help="Directory containing *_x.npy and *_dx.npy files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: npy_dir/../xdx_outlier_channel_analysis",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=99.0,
        help="Global percentile threshold for defining outliers. Example: 99.9 means top 0.1%% abs values.",
    )
    parser.add_argument(
        "--channel_stride",
        type=int,
        default=1,
        help="Channel stride used when saving npy. Used only to approximate original channel index.",
    )
    parser.add_argument(
        "--layer_regex",
        type=str,
        default="",
        help="Optional regex to filter npy prefix/layer name.",
    )
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir) if args.out_dir else npy_dir.parent / "xdx_outlier_channel_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_re = re.compile(args.layer_regex) if args.layer_regex else None

    x_files = sorted(npy_dir.glob("*_x.npy"))
    all_summary_rows = []
    all_channel_rows = []

    for x_path in x_files:
        prefix = layer_prefix_from_npy(x_path)
        if prefix is None:
            continue
        if layer_re is not None and layer_re.search(prefix) is None:
            continue

        dx_path = npy_dir / f"{prefix}_dx.npy"
        if not dx_path.exists():
            print(f"[SKIP] missing dx for {prefix}")
            continue

        print(f"[INFO] processing {prefix}")
        x_arr = np.load(x_path)
        dx_arr = np.load(dx_path)

        x_stats = compute_outlier_channel_stats(
            x_arr, q=args.q, channel_stride=args.channel_stride
        )
        dx_stats = compute_outlier_channel_stats(
            dx_arr, q=args.q, channel_stride=args.channel_stride
        )
        if x_stats is None or dx_stats is None:
            print(f"[SKIP] empty stats for {prefix}")
            continue

        x_summary, x_df = x_stats
        dx_summary, dx_df = dx_stats

        png_path = plot_x_dx_outlier_channels(
            prefix,
            x_stats,
            dx_stats,
            out_dir,
            channel_stride=args.channel_stride,
        )

        for tensor_name, summary, df in [
            ("x", x_summary, x_df),
            ("dx", dx_summary, dx_df),
        ]:
            row = {
                "layer": prefix,
                "tensor": tensor_name,
                "q": args.q,
                "shape": str(summary["shape"]),
                "threshold": summary["threshold"],
                "global_abs_median": summary["global_abs_median"],
                "global_abs_p99": summary["global_abs_p99"],
                "global_abs_p999": summary["global_abs_p999"],
                "global_abs_max": summary["global_abs_max"],
                "total_elements": summary["total_elements"],
                "total_outliers": summary["total_outliers"],
                "outlier_fraction": summary["outlier_fraction"],
                "top1_cover": summary["topk_cover"].get(1),
                "top2_cover": summary["topk_cover"].get(2),
                "top4_cover": summary["topk_cover"].get(4),
                "top8_cover": summary["topk_cover"].get(8),
                "top16_cover": summary["topk_cover"].get(16),
                "top32_cover": summary["topk_cover"].get(32),
                "plot_path": str(png_path),
            }
            all_summary_rows.append(row)

            df = df.copy()
            df.insert(0, "tensor", tensor_name)
            df.insert(0, "layer", prefix)
            all_channel_rows.append(df)

        # Save per-layer top channel table.
        merged = pd.concat(
            [
                x_df.head(64).assign(tensor="x"),
                dx_df.head(64).assign(tensor="dx"),
            ],
            ignore_index=True,
        )
        merged.to_csv(out_dir / f"{safe_name(prefix)}_top_outlier_channels.csv", index=False)

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(out_dir / "summary.csv", index=False)

    if all_channel_rows:
        pd.concat(all_channel_rows, ignore_index=True).to_csv(
            out_dir / "all_channel_outlier_distribution.csv", index=False
        )

    print(f"[DONE] output dir: {out_dir}")


if __name__ == "__main__":
    main()