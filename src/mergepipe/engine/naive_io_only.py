#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IO-only naive baseline:
- DO NOT load weights into torch tensors
- DO NOT do TIES/merge compute
- Only measure disk IO: read base+experts weight files fully (streaming),
  and write output by copying base weight files (simulate naive merged output size).
- Export CSV with columns aligned to MergePipe exec log schema.

Usage:
  python3 engine/naive_io_only.py \
    --base /path/to/base \
    --experts /path/to/e1 /path/to/e2 ... \
    --out /path/to/out \
    --model-id Llama-3.1-8B-naive-io \
    --results-dir results/naive_Llama-3.1-8B \
    --also-write-latest
"""

import argparse
import csv
import hashlib
import os
import shutil
import time
import json
import resource
from pathlib import Path
from typing import List, Tuple, Dict

WEIGHT_PATTERNS = [
    "*.safetensors",
    "pytorch_model*.bin",
    "*.bin",
]
# Some dirs contain duplicated non-weight files; we focus on likely weight files by patterns above.
# If you want to be stricter, you can exclude tokenizer/merges etc explicitly.


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ru_maxrss_mb() -> float:
    # Linux: ru_maxrss is KB
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0


def _list_weight_files(model_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pat in WEIGHT_PATTERNS:
        files.extend(model_dir.rglob(pat))
    # Filter: keep regular files, and exclude anything under "optimizer", "scheduler" etc if present
    out = []
    for f in files:
        if not f.is_file():
            continue
        name = f.name.lower()
        # heuristics: avoid tokenizer/config duplicates if they end up matching "*.bin"
        if "tokenizer" in name:
            continue
        if "training_args" in name:
            continue
        if "optimizer" in name:
            continue
        if "scheduler" in name:
            continue
        out.append(f)

    # De-dup + stable order
    out = sorted(list(set(out)))
    return out


def _stream_read_bytes(path: Path, chunk_mb: int = 64) -> int:
    bs = chunk_mb * 1024 * 1024
    total = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            total += len(b)
    return total


def _stream_copy_file(src: Path, dst: Path, chunk_mb: int = 64) -> int:
    bs = chunk_mb * 1024 * 1024
    copied = 0
    _ensure_dir(dst.parent)
    with open(src, "rb", buffering=0) as fin, open(dst, "wb", buffering=0) as fout:
        while True:
            b = fin.read(bs)
            if not b:
                break
            fout.write(b)
            copied += len(b)
    try:
        shutil.copystat(src, dst)
    except Exception:
        pass
    return copied


def _hash_plan(base: str, experts: List[str]) -> str:
    h = hashlib.sha1()
    payload = {"base": base, "experts": experts}
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def run_io_only(
    base_dir: Path,
    expert_dirs: List[Path],
    out_dir: Path,
    model_id: str,
    results_dir: Path,
    also_write_latest: bool,
) -> Dict:
    t0 = time.time()
    peak0 = _ru_maxrss_mb()

    plan_hash = _hash_plan(str(base_dir), [str(e) for e in expert_dirs])

    # -----------------------
    # 1) Read base weights
    # -----------------------
    base_files = _list_weight_files(base_dir)
    base_read_bytes = 0
    base_read_t0 = time.time()
    for f in base_files:
        base_read_bytes += _stream_read_bytes(f)
    base_read_sec = time.time() - base_read_t0

    # -----------------------
    # 2) Read expert weights
    # -----------------------
    expert_read_bytes = 0
    expert_read_t0 = time.time()
    for ed in expert_dirs:
        efiles = _list_weight_files(ed)
        for f in efiles:
            expert_read_bytes += _stream_read_bytes(f)
    expert_read_sec = time.time() - expert_read_t0

    # -----------------------
    # 3) Write out (simulate naive merged output)
    #    We copy base weight files into out_dir, keeping relative paths.
    # -----------------------
    out_write_bytes = 0
    out_write_t0 = time.time()
    _ensure_dir(out_dir)
    for f in base_files:
        rel = f.relative_to(base_dir)
        dst = out_dir / rel
        out_write_bytes += _stream_copy_file(f, dst)
    out_write_sec = time.time() - out_write_t0

    # also copy minimal metadata for a “loadable” folder if needed
    # (optional, but usually helpful)
    for meta_name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
                      "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"]:
        src = base_dir / meta_name
        if src.exists() and src.is_file():
            _ensure_dir((out_dir / meta_name).parent)
            try:
                shutil.copy2(src, out_dir / meta_name)
            except Exception:
                pass

    wall_sec = time.time() - t0
    peak1 = _ru_maxrss_mb()
    peak_mem_mb = max(peak0, peak1)

    io_mb = _bytes_to_mb(base_read_bytes + expert_read_bytes + out_write_bytes)

    stats = {
        "base_model_id": model_id,
        "n_experts": len(expert_dirs),
        "policy": "naive_io_only",
        "plan_hash": plan_hash,
        "io_budget_mb": "",          # N/A for naive
        "top_p": "",                 # N/A
        "ties_thr": "",              # N/A
        "dare_scale": "",            # N/A
        "scoring": "",               # N/A
        "block_size": "",            # N/A

        "expert_io_before_mb": "",   # N/A
        "expert_io_after_mb": "",    # N/A
        "io_budget_scale_applied": "",

        "io_mb": f"{io_mb:.3f}",
        "wall_sec": f"{wall_sec:.3f}",
        "peak_mem_mb": f"{peak_mem_mb:.3f}",

        "io_base_read_mb": f"{_bytes_to_mb(base_read_bytes):.3f}",
        "io_expert_read_mb": f"{_bytes_to_mb(expert_read_bytes):.3f}",
        "io_out_write_mb": f"{_bytes_to_mb(out_write_bytes):.3f}",

        "budget_used_mb": f"{io_mb:.3f}",   # for naive, just equal total
        "skipped_reads": "0",
        "touched_ratio": "1.0",             # naive reads everything

        "planner_sec": "0.0",
        "engine_sec": f"{wall_sec:.3f}",
        "flush_sec": f"{out_write_sec:.3f}",
        "commit_sec": "0.0",

        "db_size_mb": "0.0",
        "manifest_kb": "0.0",
    }

    # Write CSV
    _ensure_dir(results_dir)
    csv_path = results_dir / "naive_io_only.csv"
    cols = [
        "base_model_id","n_experts","policy","plan_hash","io_budget_mb","top_p","ties_thr","dare_scale",
        "scoring","block_size","expert_io_before_mb","expert_io_after_mb","io_budget_scale_applied",
        "io_mb","wall_sec","peak_mem_mb","io_base_read_mb","io_expert_read_mb","io_out_write_mb",
        "budget_used_mb","skipped_reads","touched_ratio","planner_sec","engine_sec","flush_sec",
        "commit_sec","db_size_mb","manifest_kb"
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow({k: stats.get(k, "") for k in cols})

    # also write latest
    if also_write_latest:
        latest = results_dir / "naive_exec_logs_latest.csv"
        with open(latest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({k: stats.get(k, "") for k in cols})

    # Write JSON (debug-friendly)
    json_path = results_dir / "naive_io_only.json"
    payload = {
        "time": _now(),
        "base": str(base_dir),
        "experts": [str(e) for e in expert_dirs],
        "out": str(out_dir),
        "stats": stats,
        "breakdown_sec": {
            "base_read_sec": base_read_sec,
            "expert_read_sec": expert_read_sec,
            "out_write_sec": out_write_sec,
        },
        "breakdown_bytes": {
            "base_read": base_read_bytes,
            "expert_read": expert_read_bytes,
            "out_write": out_write_bytes,
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[naive-io-only] done")
    print(f"  base_read_mb   = {_bytes_to_mb(base_read_bytes):.3f}  ({base_read_sec:.3f}s)")
    print(f"  expert_read_mb = {_bytes_to_mb(expert_read_bytes):.3f}  ({expert_read_sec:.3f}s)")
    print(f"  out_write_mb   = {_bytes_to_mb(out_write_bytes):.3f}  ({out_write_sec:.3f}s)")
    print(f"  total_io_mb    = {io_mb:.3f}")
    print(f"  wall_sec       = {wall_sec:.3f}")
    print(f"  peak_mem_mb    = {peak_mem_mb:.3f}")
    print(f"  csv            = {csv_path}")
    print(f"  json           = {json_path}")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--results-dir", default="results/naive_io_only")
    ap.add_argument("--also-write-latest", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base)
    expert_dirs = [Path(x) for x in args.experts]
    out_dir = Path(args.out)
    results_dir = Path(args.results_dir)

    run_io_only(
        base_dir=base_dir,
        expert_dirs=expert_dirs,
        out_dir=out_dir,
        model_id=args.model_id,
        results_dir=results_dir,
        also_write_latest=args.also_write_latest,
    )


if __name__ == "__main__":
    main()
