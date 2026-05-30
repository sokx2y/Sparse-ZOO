import json
import os
from pathlib import Path

import torch


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int_set(name):
    value = os.environ.get(name, "")
    result = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            continue
    return result


def _profile_dir():
    return Path(os.environ.get("LOZO_FD_PROFILE_DIR", "debug_forward_delta_profile"))


def forward_delta_profile_enabled(call_id):
    if not _env_flag("LOZO_FD_PROFILE"):
        return False
    sampled_calls = _env_int_set("LOZO_FD_PROFILE_CALLS")
    if sampled_calls:
        return call_id in sampled_calls
    return call_id <= _env_int("LOZO_FD_PROFILE_MAX_CALLS", 30)


def svd_term_profile_enabled(call_id):
    if not _env_flag("LOZO_SVD_TERM_PROFILE"):
        return False
    sampled_calls = _env_int_set("LOZO_FD_PROFILE_CALLS")
    if sampled_calls:
        return call_id in sampled_calls
    return call_id <= _env_int("LOZO_FD_PROFILE_MAX_CALLS", 30)


def _rank_suffix():
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    return f"rank{rank}"


def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _stats(tensor):
    if tensor is None:
        return None
    with torch.no_grad():
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": 0,
            }
        abs_flat = flat.abs()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(flat.numel()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean_abs": float(abs_flat.mean().item()),
            "max_abs": float(abs_flat.max().item()),
        }


def _sample(tensor):
    n = _env_int("LOZO_FD_PROFILE_SAMPLE_VALUES", 0)
    if tensor is None or n <= 0:
        return None
    with torch.no_grad():
        flat = tensor.detach().float().reshape(-1)
        return flat[: min(n, flat.numel())].cpu().tolist()


def _write_record(filename, record):
    output_dir = _profile_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _maybe_save_full_tensor(kind, layer_name, call_id, output=None, diff_output=None):
    if not _env_flag("LOZO_FD_PROFILE_FULL_TENSORS"):
        return None
    output_dir = _profile_dir() / "full_tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{kind}_{_safe_name(layer_name)}_call{call_id}_{_rank_suffix()}.pt"
    torch.save(
        {
            "output": None if output is None else output.detach().cpu(),
            "diff_output": None if diff_output is None else diff_output.detach().cpu(),
        },
        path,
    )
    return str(path)


def record_forward_delta_profile(kind, layer_name, module, call_id, output, diff_output, extra=None):
    if not forward_delta_profile_enabled(call_id):
        return
    record = {
        "record_type": "forward_delta",
        "kind": kind,
        "layer": layer_name,
        "class": module.__class__.__name__,
        "call_id": int(call_id),
        "output": _stats(output),
        "diff_output": _stats(diff_output),
    }
    output_sample = _sample(output)
    diff_sample = _sample(diff_output)
    if output_sample is not None:
        record["output_sample"] = output_sample
    if diff_sample is not None:
        record["diff_output_sample"] = diff_sample
    full_path = _maybe_save_full_tensor(kind, layer_name, call_id, output, diff_output)
    if full_path is not None:
        record["full_tensor_path"] = full_path
    if extra:
        record.update(extra)
    _write_record(f"fd_profile_{_rank_suffix()}.jsonl", record)


def record_svd_terms_profile(layer_name, module, call_id, terms, full_svd_diff, approximations=None):
    if not svd_term_profile_enabled(call_id):
        return
    full_stats = _stats(full_svd_diff)
    eps = 1e-30
    full_mean = (full_stats or {}).get("mean_abs", 0.0)
    full_max = (full_stats or {}).get("max_abs", 0.0)
    term_records = {}
    for name, tensor in terms.items():
        stats = _stats(tensor)
        if stats is not None:
            stats["ratio_mean_abs_to_full"] = stats.get("mean_abs", 0.0) / max(full_mean, eps)
            stats["ratio_max_abs_to_full"] = stats.get("max_abs", 0.0) / max(full_max, eps)
        term_records[name] = stats

    approx_records = {}
    for name, approx in (approximations or {}).items():
        error = full_svd_diff - approx
        error_stats = _stats(error)
        approx_records[name] = {
            "approx": _stats(approx),
            "error": error_stats,
            "relative_mean_error": error_stats.get("mean_abs", 0.0) / max(full_mean, eps),
            "relative_max_error": error_stats.get("max_abs", 0.0) / max(full_max, eps),
        }

    record = {
        "record_type": "svd_lora_terms",
        "kind": "svd",
        "layer": layer_name,
        "class": module.__class__.__name__,
        "call_id": int(call_id),
        "full_svd_diff": full_stats,
        "terms": term_records,
        "approximations": approx_records,
    }
    _write_record(f"svd_terms_{_rank_suffix()}.jsonl", record)
