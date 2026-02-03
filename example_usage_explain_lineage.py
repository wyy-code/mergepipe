#!/usr/bin/env python3
# example_usage_explain_lineage.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any

import numpy as np

from mergepipe.analyzer.indexer import analyze_model
from mergepipe.catalog.db import (
    connect,
    init_schema,
    register_diff,
    register_model,
    save_block_hashes,
    save_exec_log,
    save_explain_entries,
    save_lineage_op,
    save_plan,
    save_tensor_hashes,
    save_touchmap_bitmap,
    set_model_root_hash,
)
from mergepipe.engine.merge_engine import run_merge_plan
from mergepipe.storage.transaction import MergeTransaction
from mergepipe.catalog.manifest import build_manifest_from_engine, write_manifest
from mergepipe.planner.planner import plan_auto
from mergepipe.storage.io import (
    _TORCH_OK,
    DeltaStorage,
    DirectDeltaStorageProviderNP,
    DirectDeltaStorageProviderPT,
    DiskNPZStorage,
    LoRAProviderNP,
    LoRAProviderPT,
    SafeTensorsStorage,
    ShardedSafeTensorsStorage,
    StorageBase,
    WeightsAsDeltaProviderNP,
    WeightsAsDeltaProviderPT,
    load_lora_from_dir,
)

HF_SHARD_RE = re.compile(r"model-\d+-of-\d+\.safetensors$")


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def detect_env() -> dict[str, Any]:
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mergepipe": "0.1.0",
    }
    try:
        import torch  # type: ignore

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_capability"] = str(torch.cuda.get_device_capability(0))
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    return env


def _has_analysis(con: sqlite3.Connection, model_id: str, block_size: int) -> bool:
    """
    判断给定 model_id 在 DB 里是否已经存在与 `block_size` 匹配的分析结果。
    - 标准 schema：blocks 有 start/end，用 (end-start) 作为块大小。
    - 兼容老 schema：若没有 start/end 列，则只要存在 blocks 记录就认为已分析（可能会保守地触发一次重建）。
    - 注意：embedding 可能使用超大块（如 1,048,576），此处按是否存在任何与目标 block_size 相等的块来判断是否“可复用”。
    """
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM blocks WHERE model_id = ? AND (end_idx - start_idx) = ? LIMIT 1",
            (model_id, block_size),
        )
        row = cur.fetchone()
        if row is not None:
            return True
        return False
    except sqlite3.OperationalError:
        cur.execute("SELECT 1 FROM blocks WHERE model_id = ? LIMIT 1", (model_id,))
        return cur.fetchone() is not None


def is_dir_model(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return True
    shards = glob.glob(os.path.join(path, "model-*-of-*.safetensors"))
    return len(shards) > 0


def is_file_model(path: str) -> bool:
    return os.path.isfile(path) and (path.endswith(".safetensors") or path.endswith(".npz"))


def is_lora_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if glob.glob(os.path.join(path, "A_*.npy")) and glob.glob(os.path.join(path, "B_*.npy")):
        return True
    try:
        from safetensors import safe_open as st_safe_open
    except Exception:
        return False

    for sp in glob.glob(os.path.join(path, "*.safetensors")):
        try:
            with st_safe_open(sp, framework="numpy") as f:
                keys = list(f.keys())
                if any(k.startswith("lora_A.") for k in keys) or any(k.endswith(".lora_A") for k in keys):
                    return True
        except Exception:
            continue
    return False


def storage_from_base(
    path: str, writable: bool = False, backend: str = "np", device: str | None = None
) -> StorageBase:
    if os.path.isdir(path) and is_dir_model(path):
        return ShardedSafeTensorsStorage(path, writable=writable, backend=backend, device=device)
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return SafeTensorsStorage(path, writable=writable, backend=backend, device=device)
    if os.path.isfile(path) and path.endswith(".npz"):
        if backend != "np":
            raise ValueError("NPZ only supports backend=np")
        return DiskNPZStorage(path, writable=writable, backend="np")
    raise ValueError(f"Unsupported base path: {path}")


def delta_storage_from_path(
    path: str, backend: str = "np", device: str | None = None
) -> StorageBase:
    if os.path.isdir(path) and is_dir_model(path):
        return DeltaStorage(path, writable=False, backend=backend, device=device)
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return SafeTensorsStorage(path, writable=False, backend=backend, device=device)
    if os.path.isfile(path) and path.endswith(".npz"):
        if backend != "np":
            raise ValueError("NPZ only supports backend=np")
        return DiskNPZStorage(path, writable=False, backend="np")
    raise ValueError(f"Expert delta path not recognized: {path}")


def compute_model_root_hash(storage: StorageBase) -> str:
    import numpy as _np

    h = hashlib.sha256()
    for name, arr in storage.iter_tensors():
        if hasattr(arr, "detach"):  # torch
            v = (
                arr.detach()
                .to(dtype=None)
                .to("cpu")
                .numpy()
                .astype(_np.float32, copy=False)
                .tobytes()
            )
        else:
            v = _np.asarray(arr, dtype=_np.float32).tobytes()
        th = hashlib.sha256(v).digest()
        h.update(th)
        h.update(name.encode())
    return h.hexdigest()


def _model_summary_from_index(index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    n_tensors = 0
    n_params = 0
    for _, meta in index.items():
        dt = str(meta.get("dtype"))
        dtypes[dt] = dtypes.get(dt, 0) + 1
        shape = meta.get("shape") or []
        n = 1
        for d in shape:
            n *= int(d)
        n_params += n
        n_tensors += 1
    return {"dtype_hist": dtypes, "n_tensors": n_tensors, "n_params_est": n_params}


def register_from_storage(
    con: sqlite3.Connection,
    model_id: str,
    uri: str,
    arch: str,
    storage: StorageBase,
    parent_id: str | None = None,
    stage: str = "dev",
) -> None:
    """
    优先用存储的 _index 只读元数据，避免 iter_tensors() 把整张量读进来。
    同时落 models.summary_json（dtype 直方图、近似参数量），便于后续统计。
    """
    tensors_rows = []
    n_params = 0

    index = getattr(storage, "_index", None)
    summary = {}
    if isinstance(index, dict) and index:
        summary = _model_summary_from_index(index)
        try:
            shards = set(meta["shard"] for meta in index.values() if "shard" in meta)
            summary["n_shards"] = len(shards) if shards else 1
        except Exception:
            summary["n_shards"] = 1

        for name, meta in index.items():
            shape = list(meta["shape"])
            dtype = str(meta["dtype"])
            n_elem = 1
            for d in shape:
                n_elem *= int(d)
            n_params += n_elem
            tensors_rows.append((name, shape, dtype, n_elem))
    else:
        for name, arr in storage.iter_tensors():
            shape = list(arr.shape)
            dtype = str(arr.dtype)
            n_elem = int(arr.size) if hasattr(arr, "size") else int(np.prod(shape))
            n_params += n_elem
            tensors_rows.append((name, shape, dtype, n_elem))
        summary = {
            "dtype_hist": {},
            "n_tensors": len(tensors_rows),
            "n_params_est": n_params,
            "n_shards": 1,
        }

    register_model(
        con,
        model_id=model_id,
        uri=uri,
        arch=arch,
        dtype="mixed",
        tensors=tensors_rows,
        n_params=n_params,
        created_at=_now_str(),
        parent_id=parent_id,
        stage=stage,
        summary=summary,
    )


def flush_output(storage: StorageBase, out_target: str, base_path: str) -> str:
    if isinstance(storage, ShardedSafeTensorsStorage):
        out_dir = out_target or (base_path.rstrip("/\\") + "-merged")
        storage.flush(out_dir)
        return out_dir
    else:
        if not out_target:
            base_noext, ext = os.path.splitext(base_path)
            out_target = base_noext + "-merged" + ext
        storage.flush(out_target)
        return out_target


def auto_kind(path: str) -> str:
    if is_lora_dir(path):
        return "lora"
    if is_dir_model(path) or is_file_model(path):
        return "weights"
    return "lora"


def _normalize_exec_stats(exec_stats: dict[str, Any], fallback_wall_sec: float) -> dict[str, Any]:
    """
    Ensure io_mb / wall_sec exist for logging & printing.
    Also leave original fields intact.
    """
    out = dict(exec_stats or {})

    # wall_sec normalization
    wall = out.get("wall_sec", None)
    if wall is None:
        wall = out.get("wall_time_sec", None)
    if wall is None:
        wall = out.get("elapsed_sec", None)
    if wall is None:
        wall = fallback_wall_sec
    out["wall_sec"] = float(wall)

    # io_mb normalization
    io_mb = out.get("io_mb", None)
    if io_mb is None:
        io_bytes = out.get("io_bytes", None)
        if io_bytes is None:
            rb = float(out.get("io_bytes_read", 0.0) or 0.0)
            wb = float(out.get("io_bytes_written", 0.0) or 0.0)
            io_bytes = rb + wb if (rb or wb) else None
        if io_bytes is None:
            io_mb = 0.0
        else:
            io_mb = float(io_bytes) / (1024.0 * 1024.0)
    out["io_mb"] = float(io_mb)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MergePipe example: auto-detect sharded safetensors & providers with np/pt backends"
    )
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--base", type=str, required=True, help="Base: dir (HF shards) or .safetensors/.npz")
    parser.add_argument("--experts", type=str, nargs="+", required=True,
                        help="Experts: dirs/files for weights/delta, or LoRA dirs")
    parser.add_argument("--experts-kind", type=str, choices=["auto", "weights", "delta", "lora"], default="auto")
    parser.add_argument("--backend", type=str, choices=["np", "pt"], default="np")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device when --backend pt (e.g., cuda:0 or cpu)")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--arch", type=str, default="llm-arch")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=0.35)
    parser.add_argument("--ties-thr", type=float, default=0.02)
    parser.add_argument("--dare-scale", type=float, default=0.8)
    parser.add_argument("--io-budget-mb", type=int, default=8192)
    parser.add_argument("--analyze-all", action="store_true")
    parser.add_argument("--scoring", type=str, choices=["auto", "l2cos", "abs"], default="auto")
    parser.add_argument("--stop-after", choices=["register", "analyze", "none"], default="none",
                        help="Convenience switch to stop early for debugging/perf checks.")
    parser.add_argument("--analyze-experts",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Analyze expert models to enable conflict-aware planning (cos_sim). "
                             "Only applies to --experts-kind=weights.")
    parser.add_argument("--force-analyze",
                        action="store_true",
                        help="Ignore cached analysis in DB and re-run ANALYZE for base/experts.")
    parser.add_argument("--model-id",
                        required=True,
                        help="Logical id (name) of the merged model; used for transaction, snapshot dir, and lineage.")
    parser.add_argument("--disable-budget",
                        action="store_true",
                        help="Disable IO budget enforcement in planner (ablation). Keeps top_p unchanged even if --io-budget-mb is set.",)
    
    args = parser.parse_args()

    if args.backend == "pt" and not _TORCH_OK:
        print("[ERROR] --backend pt requires torch. Please `pip install torch`.", file=sys.stderr)
        sys.exit(2)

    con = connect(args.db)
    init_schema(con)

    t0_total = time.time()

    # base storages
    base_storage_ro = storage_from_base(args.base, writable=False, backend=args.backend, device=args.device)
    out_storage = storage_from_base(args.base, writable=True, backend=args.backend, device=args.device)

    # register base
    base_id = f"base_{hashlib.sha256(args.base.encode()).hexdigest()[:8]}"
    print("[MergePipe] REGISTER base ...")
    register_from_storage(con, model_id=base_id, uri=args.base, arch=args.arch, storage=base_storage_ro)
    print("[MergePipe] REGISTER base done.")

    # providers
    expert_ids: list[str] = []
    delta_providers: list = []
    expert_stores: list[StorageBase] = []

    kind = args.experts_kind
    if kind == "auto":
        inferred = [auto_kind(p) for p in args.experts]
        if len(set(inferred)) > 1:
            print(f"[WARN] Mixed expert kinds detected {set(inferred)}; "
                  f"please set --experts-kind explicitly.", file=sys.stderr)
        kind = inferred[0]

    if args.backend == "pt":
        if kind == "weights":
            expert_stores = [storage_from_base(p, writable=False, backend="pt", device=args.device) for p in args.experts]
            for p, es in zip(args.experts, expert_stores, strict=False):
                eid = f"exp_{hashlib.sha256(p.encode()).hexdigest()[:8]}"
                register_from_storage(con, model_id=eid, uri=p, arch=args.arch, storage=es)
                expert_ids.append(eid)
                delta_providers.append(WeightsAsDeltaProviderPT(es, device=args.device))

        elif kind == "delta":
            from storage.io import compute_delta_coverage

            for p in args.experts:
                ds = delta_storage_from_path(p, backend="pt", device=args.device)
                eid = f"delta_{hashlib.sha256(p.encode()).hexdigest()[:8]}"
                register_from_storage(con, model_id=eid, uri=p, arch=args.arch, storage=ds)

                cov = compute_delta_coverage(
                    provider=DirectDeltaStorageProviderPT(ds, device=args.device),
                    base_storage=base_storage_ro,
                    block_size=args.block_size,
                    backend="pt",
                    device=args.device,
                    only_tensors=None,
                )
                _ = register_diff(
                    con,
                    base_model_id=base_id,
                    derived_model_id=eid,
                    uri=p,
                    granularity=1,
                    coverage_json=cov.coverage_json,
                )
                for tname, bm in cov.touchmaps.items():
                    save_touchmap_bitmap(con, eid, tname, bm)

                expert_ids.append(eid)
                delta_providers.append(DirectDeltaStorageProviderPT(ds, device=args.device))

        elif kind == "lora":
            from storage.io import compute_delta_coverage

            lora_map: dict[str, dict[str, Any]] = {}
            for p in args.experts:
                part = load_lora_from_dir(p, alpha=0.8)
                lora_map.update(part)
            lora_provider = LoRAProviderPT(lora_map, device=args.device)
            delta_providers.append(lora_provider)
            expert_ids.append("lora_mix")

            cov = compute_delta_coverage(
                provider=lora_provider,
                base_storage=base_storage_ro,
                block_size=args.block_size,
                backend="pt",
                device=args.device,
                only_tensors=None,
            )
            register_diff(
                con,
                base_model_id=base_id,
                derived_model_id="lora_mix",
                uri=",".join(args.experts),
                granularity=1,
                coverage_json=cov.coverage_json,
            )
            for tname, bm in cov.touchmaps.items():
                save_touchmap_bitmap(con, "lora_mix", tname, bm)
        else:
            raise ValueError(f"Unsupported experts-kind: {args.experts_kind}")

    else:
        if kind == "weights":
            expert_stores = [storage_from_base(p, writable=False, backend="np") for p in args.experts]
            for p, es in zip(args.experts, expert_stores, strict=False):
                eid = f"exp_{hashlib.sha256(p.encode()).hexdigest()[:8]}"
                register_from_storage(con, model_id=eid, uri=p, arch=args.arch, storage=es)
                expert_ids.append(eid)
                delta_providers.append(WeightsAsDeltaProviderNP(es))

        elif kind == "delta":
            from storage.io import compute_delta_coverage

            for p in args.experts:
                ds = delta_storage_from_path(p, backend="np")
                eid = f"delta_{hashlib.sha256(p.encode()).hexdigest()[:8]}"
                register_from_storage(con, model_id=eid, uri=p, arch=args.arch, storage=ds)

                cov = compute_delta_coverage(
                    provider=DirectDeltaStorageProviderNP(ds),
                    base_storage=base_storage_ro,
                    block_size=args.block_size,
                    backend="np",
                    device=None,
                    only_tensors=None,
                )
                _ = register_diff(
                    con,
                    base_model_id=base_id,
                    derived_model_id=eid,
                    uri=p,
                    granularity=1,
                    coverage_json=cov.coverage_json,
                )
                for tname, bm in cov.touchmaps.items():
                    save_touchmap_bitmap(con, eid, tname, bm)

                expert_ids.append(eid)
                delta_providers.append(DirectDeltaStorageProviderNP(ds))

        elif kind == "lora":
            lora_map: dict[str, dict[str, Any]] = {}
            for p in args.experts:
                part = load_lora_from_dir(p, alpha=0.8)
                lora_map.update(part)
            delta_providers.append(LoRAProviderNP(lora_map))
            expert_ids.append("lora_mix")
        else:
            raise ValueError(f"Unsupported experts-kind: {args.experts_kind}")

    # analyze base (reuse cached)
    if args.force_analyze or not _has_analysis(con, base_id, args.block_size):
        print("[MergePipe] ANALYZE ...")
        analyze_model(
            con,
            model_id=base_id,
            storage=base_storage_ro,
            block_size=args.block_size,
            only_key_tensors=not args.analyze_all,
        )
        print("[MergePipe] ANALYZE done.")
    else:
        print("[MergePipe] ANALYZE base skipped (cached).")

    # analyze experts (weights only), reuse cached analysis
    if args.analyze_experts and kind == "weights" and expert_stores:
        need_list = []
        for eid in expert_ids:
            need = args.force_analyze or not _has_analysis(con, eid, args.block_size)
            need_list.append(need)

        if any(need_list):
            print(f"[MergePipe] ANALYZE experts ({sum(1 for x in need_list if x)}) ...")
            for (eid, es, need) in zip(expert_ids, expert_stores, need_list, strict=False):
                if not need:
                    continue
                analyze_model(
                    con,
                    model_id=eid,
                    storage=es,
                    block_size=args.block_size,
                    only_key_tensors=not args.analyze_all,
                )
            print("[MergePipe] ANALYZE experts done.")
        else:
            print("[MergePipe] ANALYZE experts skipped (cached).")
    elif args.analyze_experts and kind != "weights":
        print("[Hint] --analyze-experts only applies to --experts-kind weights; skipped.", file=sys.stderr)

    if args.stop_after in ("register", "analyze"):
        if args.stop_after == "register":
            print("[MergePipe] Stop after REGISTER as requested.")
        else:
            print("[MergePipe] Stop after ANALYZE as requested.")
        return

    save_lineage_op(
        con,
        "ANALYZE",
        inputs={
            "model_id": base_id,
            "block_size": args.block_size,
            "only_key_tensors": not args.analyze_all,
            "analyze_experts": bool(args.analyze_experts),
            "experts_kind": kind,
        },
        outputs={"indexed": True},
        params={},
        env=detect_env(),
    )

    # plan (stable planner keys)
    options = {
        "io_budget_mb": args.io_budget_mb,
        "top_p": args.top_p,
        "ties_thr": args.ties_thr,
        "dare_scale": args.dare_scale,
        "scoring": args.scoring,
        "policy": "conflict_aware",
        "block_size": args.block_size,
        "disable_budget": bool(args.disable_budget),
    }
    t0_planner = time.time()
    plan = plan_auto(con, base_model_id=base_id, experts=expert_ids, options=options)
    t1_planner = time.time()
    plan["type"] = "merge_plan_v1"


    # planner summary
    p_opts = plan.get("options", {}) or {}
    pol = plan.get("policy") or p_opts.get("policy")
    print("[Planner] policy:", pol)

    n_cos = 0
    for t in plan.get("tensors", []) or []:
        if isinstance(t, dict) and isinstance(t.get("reason"), dict) and ("cos_sim_mean" in t["reason"]):
            n_cos += 1
    print(f"[Planner] tensors with cos_sim_mean: {n_cos}/{len(plan.get('tensors', []))}")

    # exec (transaction-aware)
    outputs_root = args.out
    tx = MergeTransaction(
        db_path=args.db,
        outputs_root=str(outputs_root),
        model_id=args.model_id,
        parents=[base_id] + expert_ids,
    )

    t0_merge_total = time.time()
    tx.begin()
    try:
        # 1) engine merge (do not flush final output yet)
        t0_engine = time.time()
        exec_stats = run_merge_plan(
            plan,
            base_storage=base_storage_ro,
            delta_providers=delta_providers,
            out_storage=out_storage,
            block_size=args.block_size,
            scoring=args.scoring,
            backend=args.backend,
            device=args.device if args.backend == "pt" else None,
            do_flush=False,
        )
        t1_engine = time.time()

        # 2) flush to staging dir
        t0_flush = time.time()
        out_storage.flush(str(tx.staging_dir))
        t1_flush = time.time()

        # 3) compute root hash on staged snapshot (before publish)
        merged_ro_stage = storage_from_base(
            str(tx.staging_dir),
            writable=False,
            backend=args.backend,
            device=args.device,
        )
        try:
            root_hash = compute_model_root_hash(merged_ro_stage)
        except Exception as e:
            print(f"[WARN] failed to compute root hash: {e}", file=sys.stderr)
            root_hash = None

        # 4) build + write manifest
        try:
            tensor_names = []
            for x in plan.get("tensors", []):
                if isinstance(x, dict):
                    n = x.get("tensor_name") or x.get("name")  # backward compat
                    if n:
                        tensor_names.append(n)
                else:
                    tensor_names.append(str(x))
            tensor_names = [n for n in tensor_names if n]

            tensor_shapes = {n: list(base_storage_ro.tensor_shape(n)) for n in tensor_names}

            tensor_dtypes = {}
            for n in tensor_names:
                if hasattr(base_storage_ro, "dtype"):
                    tensor_dtypes[n] = str(base_storage_ro.dtype(n))
                elif hasattr(base_storage_ro, "tensor_dtype"):
                    tensor_dtypes[n] = str(base_storage_ro.tensor_dtype(n))
                else:
                    tensor_dtypes[n] = "float32"

            engine_result = dict(exec_stats)
            engine_result["root_hash"] = root_hash

            manifest = build_manifest_from_engine(
                commit_id=tx.commit_id,
                model_id=args.model_id,
                parents=[base_id] + expert_ids,
                plan_type=str(plan.get("type") or "merge_plan_v1"),
                plan_id=str(plan.get("plan_id") or ""),
                plan_hash=str(plan.get("plan_hash") or ""),
                engine_result=engine_result,
                tensor_shapes=tensor_shapes,
                tensor_dtypes=tensor_dtypes,
            )

            manifest_path = tx.staging_dir / "manifest.json"
            write_manifest(manifest_path, manifest)
            tx.attach_manifest_obj(manifest, manifest_path=manifest_path)
        except Exception as e:
            tx.mark_error(f"manifest build failed: {e}")
            raise

        # 5) publish staging -> outputs/<model_id> atomically
        t0_commit = time.time()
        tx.commit()
        t1_commit = time.time()

        out_path = str(tx.final_dir)

    except Exception as e:
        tx.mark_error(str(e))
        tx.abort()
        raise

    t1_merge_total = time.time()

    # normalize exec_stats for later logging/printing
    exec_stats = _normalize_exec_stats(exec_stats, fallback_wall_sec=(t1_engine - t0_engine))

    # open merged snapshot (final)
    merged_ro = storage_from_base(
        out_path,
        writable=False,
        backend=args.backend,
        device=args.device,
    )

    # register merged
    merged_id = args.model_id
    register_from_storage(
        con,
        model_id=merged_id,
        uri=str(out_path),
        arch=args.arch,
        storage=merged_ro,
        parent_id=base_id,
    )
    if root_hash:
        set_model_root_hash(con, merged_id, root_hash)

    # plan/explain/lineage
    env = detect_env()
    save_plan(con, plan)
    entries = exec_stats.get("explain_entries") or []
    if entries:
        save_explain_entries(con, plan_id=plan["plan_id"], entries=entries)

    save_lineage_op(
        con,
        "MERGE",
        inputs={"base_model": plan["base_model"], "experts": plan.get("experts", [])},
        outputs={"output_model_id": merged_id, "plan_id": plan["plan_id"], "out": str(out_path)},
        params={"plan_hash": plan.get("plan_hash"), "options": plan.get("options", {})},
        env=env,
    )

    # --- Route B: derive exec breakdown metrics ---
    def _b2mb(x: Any) -> float | None:
        if x is None:
            return None
        try:
            return float(x) / (1024.0 * 1024.0)
        except Exception:
            return None

    io_base_read_mb = _b2mb(exec_stats.get("io_bytes_base_read"))
    io_expert_read_mb = _b2mb(exec_stats.get("io_bytes_expert_read"))
    io_out_write_mb = _b2mb(exec_stats.get("io_bytes_out_write"))

    budget_used_mb = None
    skipped_reads = None
    io_budget = exec_stats.get("io_budget")
    if isinstance(io_budget, dict):
        if io_budget.get("used_mb") is not None:
            budget_used_mb = float(io_budget["used_mb"])
        if io_budget.get("skipped_reads") is not None:
            skipped_reads = int(io_budget["skipped_reads"])

    def _popcount_bytes(b: bytes) -> int:
        # python3.8+ int.bit_count available
        return sum(int(x).bit_count() for x in b)

    touched_ratio = None
    try:
        tm = exec_stats.get("touchmaps") or {}
        if isinstance(tm, dict) and tm:
            # Use tensor shapes from base storage to compute block count
            total_blocks = 0
            touched_blocks = 0
            for tname, bm in tm.items():
                if not bm:
                    continue
                try:
                    shape = list(base_storage_ro.tensor_shape(tname))
                    numel = 1
                    for d in shape:
                        numel *= int(d)
                    nb = (numel + args.block_size - 1) // args.block_size
                    total_blocks += int(nb)
                    touched_blocks += int(_popcount_bytes(bytes(bm)))
                except Exception:
                    continue
            if total_blocks > 0:
                touched_ratio = float(touched_blocks) / float(total_blocks)
    except Exception:
        touched_ratio = None

    db_size_mb = None
    manifest_kb = None
    try:
        if os.path.isfile(args.db):
            db_size_mb = os.path.getsize(args.db) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        # manifest_path exists in staging; after commit it also exists in final dir
        mp = os.path.join(out_path, "manifest.json")
        if os.path.isfile(mp):
            manifest_kb = os.path.getsize(mp) / 1024.0
    except Exception:
        pass

    # exec logs
    try:
        save_exec_log(
            con,
            plan_id=plan["plan_id"],
            model_id=merged_id,
            stage="MERGE",
            io_mb=float(exec_stats.get("io_mb", 0.0)),
            wall_sec=float(exec_stats.get("wall_sec", 0.0)),
            peak_mem_mb=None,
            io_base_read_mb=io_base_read_mb,
            io_expert_read_mb=io_expert_read_mb,
            io_out_write_mb=io_out_write_mb,
            budget_used_mb=budget_used_mb,
            skipped_reads=skipped_reads,
            touched_ratio=touched_ratio,
            planner_sec=(t1_planner - t0_planner) if "t0_planner" in locals() else None,
            engine_sec=(t1_engine - t0_engine),
            flush_sec=(t1_flush - t0_flush),
            commit_sec=(t1_commit - t0_commit),
            db_size_mb=db_size_mb,
            manifest_kb=manifest_kb,
        )

    except Exception as e:
        print(f"[WARN] save_exec_log failed: {e}", file=sys.stderr)

    # hashes + touchmaps
    try:
        bh_items = exec_stats.get("block_hashes", []) or []
        th_items = exec_stats.get("tensor_hashes", []) or []
        if bh_items:
            save_block_hashes(con, model_id=merged_id, items=bh_items)
        if th_items:
            save_tensor_hashes(con, model_id=merged_id, items=th_items)
        tm = exec_stats.get("touchmaps") or {}
        for tname, bitmap in tm.items():
            save_touchmap_bitmap(con, merged_id, tname, bitmap)
    except Exception as e:
        print(f"[WARN] saving hashes/touchmaps failed: {e}", file=sys.stderr)

    # summary
    print("\n=== Merge Summary ===")
    print(f"DB:              {os.path.abspath(args.db)}")
    print(f"Backend:         {args.backend}  (device={args.device})")
    print(f"Base model id:   {base_id}")
    print(f"Expert kind:     {kind}")
    print(f"Expert ids:      {plan.get('experts', [])}")
    print(f"Plan id:         {plan['plan_id']}")
    print(f"Policy:          {pol}")

    if "io_budget_scale_applied" in p_opts:
        print(f"Budget scale:    x{float(p_opts['io_budget_scale_applied']):.3f}")

    print(f"Merged model id: {merged_id}")
    print(f"Output:          {out_path}")
    if root_hash:
        print(f"Root hash:       {root_hash}")

    # Print exec stats + planner controllable IO side-by-side
    if ("expert_io_after_mb" in p_opts) and ("expert_io_before_mb" in p_opts):
        print(
            f"Exec stats:      io_mb={float(exec_stats.get('io_mb', 0.0)):.2f}, wall_sec={float(exec_stats.get('wall_sec', 0.0)):.2f}, "
            f"expert_io_mb={float(p_opts['expert_io_after_mb']):.2f} (before={float(p_opts['expert_io_before_mb']):.2f})"
        )
    elif "expert_io_after_mb" in p_opts:
        print(
            f"Exec stats:      io_mb={float(exec_stats.get('io_mb', 0.0)):.2f}, wall_sec={float(exec_stats.get('wall_sec', 0.0)):.2f}, "
            f"expert_io_mb={float(p_opts['expert_io_after_mb']):.2f}"
        )
    else:
        print(f"Exec stats:      io_mb={float(exec_stats.get('io_mb', 0.0)):.2f}, wall_sec={float(exec_stats.get('wall_sec', 0.0)):.2f}")

    # Optional: if merge_engine returns breakdowns, print them.
    io_bd = exec_stats.get("io_breakdown", None)
    if isinstance(io_bd, dict) and io_bd:
        # allow bytes or mb fields
        bd = dict(io_bd)
        if "read_mb" not in bd and "read_bytes" in bd:
            bd["read_mb"] = float(bd.get("read_bytes", 0.0)) / (1024.0 * 1024.0)
        if "write_mb" not in bd and "write_bytes" in bd:
            bd["write_mb"] = float(bd.get("write_bytes", 0.0)) / (1024.0 * 1024.0)
        msg = []
        if "read_mb" in bd:
            msg.append(f"read_mb={float(bd['read_mb']):.2f}")
        if "write_mb" in bd:
            msg.append(f"write_mb={float(bd['write_mb']):.2f}")
        for k in ("base_read_mb", "expert_read_mb", "out_write_mb"):
            if k in bd:
                msg.append(f"{k}={float(bd[k]):.2f}")
        if msg:
            print("IO breakdown:   " + ", ".join(msg))

    # Print local timing breakdown for transaction pipeline
    print(
        "Timing local:    "
        f"engine_sec={t1_engine - t0_engine:.3f}, flush_sec={t1_flush - t0_flush:.3f}, commit_sec={t1_commit - t0_commit:.3f}, "
        f"merge_total_sec={t1_merge_total - t0_merge_total:.3f}"
    )

    print("Explain entries saved to DB.")
    print("Plan saved with tensors & policy in `plans` table.")
    print("Hashes & touchmaps saved to DB.")
    print(f"The totall time cost is {time.time() - t0_total}s.")


if __name__ == "__main__":
    main()
