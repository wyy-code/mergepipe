#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import time
import argparse
import hashlib
import datetime
import resource
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Import your "standard" TIES implementation helpers (must match your provided files) ----
# You said your reference files are:
#   - TIES.py (calls get_task_vector / ties_merging / vector_to_state_dict)
#   - ties_merging_utils.py
#   - utils.py
#
# Here we import those exact functions from the same directory or repo import path.
# If your repo structure is /zju_0038/wyy/mergepipe/merging_methods/... then adjust sys.path accordingly.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # allow importing utils.py & ties_merging_utils.py sitting with this file

from ties_merging_utils import ties_merging  # noqa
from utils import get_task_vector, vector_to_state_dict  # noqa


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _bytes_to_mb(x: float) -> float:
    return float(x) / (1024.0 * 1024.0)


def _ru_maxrss_mb() -> float:
    # Linux: ru_maxrss is KB; macOS: bytes. Your env is Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / 1024.0


def _sum_model_files_bytes(model_dir: str) -> int:
    """
    Estimate disk I/O as sum of weight files and relevant small metadata files.
    This is good enough for system-level "I/O reduction" comparison.
    """
    d = Path(model_dir)
    if not d.exists():
        return 0

    patterns = [
        "*.safetensors",
        "*.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    total = 0
    seen = set()
    for pat in patterns:
        for p in d.glob(pat):
            try:
                rp = str(p.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                total += p.stat().st_size
            except Exception:
                pass
    return total


def _sum_dir_bytes(dir_path: str) -> int:
    d = Path(dir_path)
    if not d.exists():
        return 0
    total = 0
    for p in d.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            pass
    return total


def _write_csv_row(csv_path: str, row: Dict[str, Any], header: List[str]) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in header})


def _copy_as_latest(src: str, latest_path: str) -> None:
    Path(latest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fr, open(latest_path, "wb") as fw:
        fw.write(fr.read())


def load_hf_causal_lm(model_path: str, dtype: str = "float16"):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    # CPU-only load
    try:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    except ValueError:
        print(model_path)
        print("loading error...")
    return model

def _check_head_and_embed_shapes(base_model, expert_model, expert_path):
    bsd = base_model.state_dict()
    esd = expert_model.state_dict()

    critical = ["model.embed_tokens.weight", "lm_head.weight"]
    for k in critical:
        if k not in esd:
            raise RuntimeError(f"expert missing key: {k} ({expert_path})")
        if bsd[k].shape != esd[k].shape:
            raise RuntimeError(
                f"shape mismatch for {k}: base={tuple(bsd[k].shape)} vs "
                f"expert={tuple(esd[k].shape)} ({expert_path})"
            )
        
def run_ties_merge(
    base_path: str,
    expert_paths: List[str],
    out_path: str,
    scaling_coef: float,
    K: float,
    merge_func: str,
    dtype: str,
) -> None:
    """
    "Real TIES" aligned with your reference:
      - task_vectors = [get_task_vector(ft_model, base_model) ...]
      - merged_tv = scaling_coef * ties_merging(stack(task_vectors), reset_thresh=K, merge_func=merge_func)
      - merged_model = vector_to_state_dict(merged_tv, base_model)
    """
    base_model = load_hf_causal_lm(base_path, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    ft_models = [(p, load_hf_causal_lm(p, dtype=dtype)) for p in expert_paths]

    # task_vectors = [get_task_vector(ftm, base_model) for ftm in ft_models]
    task_vectors = []
    for p, ftm in ft_models:
        try:
            _check_head_and_embed_shapes(base_model, ftm, p)
            task_vector = get_task_vector(ftm, base_model)
            task_vectors.append(task_vector)
        except Exception as e:
            print(f"[BAD EXPERT] path={p}")
            print(f"Exception: {type(e).__name__}: {e}")
            raise
            
    merged_tv = scaling_coef * ties_merging(
        torch.stack(task_vectors),
        reset_thresh=K,
        merge_func=merge_func,
    )
    merged_model = vector_to_state_dict(merged_tv, base_model)

    Path(out_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model directory")
    ap.add_argument("--experts", required=True, nargs="+", help="Expert model directories")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument("--model-id", default="naive_merged", help="Merged model id for logs")

    # TIES params (aligned to your reference)
    ap.add_argument("--scaling-coef", type=float, default=1.0, help="TIES scaling_coef")
    ap.add_argument("--ties-thr", type=float, default=20.0, help="TIES reset_thresh K (same name as your CSV)")
    ap.add_argument("--merge-func", type=str, default="sum",
                    choices=["mean", "sum", "max"],
                    help="Merge func used inside ties_merging_utils.py")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    # metrics export
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--csv-name", type=str, default=None, help="Optional csv filename, else auto timestamp.")
    ap.add_argument("--also-write-latest", action="store_true")

    args = ap.parse_args()

    base_path = args.base
    expert_paths = args.experts
    out_path = args.out
    model_id = args.model_id

    # --- metrics: IO size estimates ---
    base_bytes = _sum_model_files_bytes(base_path)
    expert_bytes = sum(_sum_model_files_bytes(p) for p in expert_paths)

    # --- run ---
    t0 = time.time()
    peak0 = _ru_maxrss_mb()

    # stage: merge
    run_ties_merge(
        base_path=base_path,
        expert_paths=expert_paths,
        out_path=out_path,
        scaling_coef=args.scaling_coef,
        K=args.ties_thr,
        merge_func=args.merge_func,
        dtype=args.dtype,
    )

    wall = time.time() - t0
    peak1 = _ru_maxrss_mb()
    peak_mem_mb = max(peak0, peak1)

    out_bytes = _sum_dir_bytes(out_path)

    # --- build a row aligned with MergePipe exec_logs schema ---
    # Use same column set you listed.
    header = [
        "base_model_id", "n_experts", "policy", "plan_hash", "io_budget_mb",
        "top_p", "ties_thr", "dare_scale", "scoring", "block_size",
        "expert_io_before_mb", "expert_io_after_mb", "io_budget_scale_applied",
        "io_mb", "wall_sec", "peak_mem_mb",
        "io_base_read_mb", "io_expert_read_mb", "io_out_write_mb",
        "budget_used_mb", "skipped_reads", "touched_ratio",
        "planner_sec", "engine_sec", "flush_sec", "commit_sec",
        "db_size_mb", "manifest_kb"
    ]

    # For naive baseline:
    # - policy fixed: naive_ties
    # - no DB / planner / manifest / block planning => keep NA or 0
    # - expert_io_before/after: "before" == after == full read (baseline reads full experts)
    # - touched_ratio = 1.0 (touch everything)
    # - skipped_reads = 0
    # - budget_used_mb: equals io_mb (baseline just uses whatever it uses)
    # - plan_hash: hash config+paths to keep deterministic id
    plan_str = json.dumps({
        "method": "naive_ties",
        "base": str(Path(base_path).resolve()),
        "experts": [str(Path(p).resolve()) for p in expert_paths],
        "scaling_coef": args.scaling_coef,
        "ties_thr": args.ties_thr,
        "merge_func": args.merge_func,
        "dtype": args.dtype,
    }, sort_keys=True)
    plan_hash = _sha1_str(plan_str)[:16]

    base_model_id = "base_" + _sha1_str(str(Path(base_path).resolve()))[:8]

    io_base_read_mb = _bytes_to_mb(base_bytes)
    io_expert_read_mb = _bytes_to_mb(expert_bytes)
    io_out_write_mb = _bytes_to_mb(out_bytes)
    io_mb = io_base_read_mb + io_expert_read_mb + io_out_write_mb

    row = {
        "base_model_id": base_model_id,
        "n_experts": len(expert_paths),
        "policy": "naive_ties",
        "plan_hash": plan_hash,
        "io_budget_mb": None,          # NA for naive
        "top_p": None,                 # NA
        "ties_thr": args.ties_thr,
        "dare_scale": None,            # NA
        "scoring": None,               # NA
        "block_size": None,            # NA

        "expert_io_before_mb": io_expert_read_mb,
        "expert_io_after_mb": io_expert_read_mb,  # naive reads all
        "io_budget_scale_applied": None,

        "io_mb": io_mb,
        "wall_sec": wall,
        "peak_mem_mb": peak_mem_mb,

        "io_base_read_mb": io_base_read_mb,
        "io_expert_read_mb": io_expert_read_mb,
        "io_out_write_mb": io_out_write_mb,

        "budget_used_mb": io_mb,
        "skipped_reads": 0,
        "touched_ratio": 1.0,

        "planner_sec": 0.0,
        "engine_sec": wall,
        "flush_sec": 0.0,
        "commit_sec": 0.0,

        "db_size_mb": None,
        "manifest_kb": None,
    }

    # --- export ---
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_name is None:
        csv_path = results_dir / f"naive_exec_logs_{_now_tag()}.csv"
    else:
        csv_path = results_dir / args.csv_name

    _write_csv_row(str(csv_path), row, header)

    if args.also_write_latest:
        _copy_as_latest(str(csv_path), str(results_dir / "naive_exec_logs_latest.csv"))

    print(f"[OK] naive TIES merged model saved to: {out_path}")
    print(f"[OK] exported 1 row to: {csv_path}")
    if args.also_write_latest:
        print(f"[OK] exported latest to: {results_dir / 'naive_exec_logs_latest.csv'}")


if __name__ == "__main__":
    main()
