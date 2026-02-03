# mergepipe/planner/planner.py
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import Any

import numpy as np

from mergepipe.catalog.db import get_touchmap_bitmap, touchmap_ratio_from_bitmap


# -----------------------------
# Helpers: DB access
# -----------------------------

def _dtype_nbytes(dtype_str: str) -> int:
    ds = dtype_str.lower().replace("torch.", "").replace("numpy.", "")
    if "float32" in ds or ds == "fp32":
        return 4
    if "float16" in ds or "fp16" in ds or "half" in ds:
        return 2
    if "bfloat16" in ds or "bf16" in ds:
        return 2
    if "int8" in ds or "uint8" in ds or "qint8" in ds:
        return 1
    if "int16" in ds or "uint16" in ds:
        return 2
    if "int32" in ds or "uint32" in ds:
        return 4
    if "int64" in ds or "uint64" in ds:
        return 8
    return 4


def _fetch_tensors(con: sqlite3.Connection, model_id: str) -> list[tuple[str, tuple[int, ...], str]]:
    cur = con.cursor()
    cur.execute("SELECT name, shape, dtype FROM tensors WHERE model_id=?", (model_id,))
    rows = cur.fetchall()
    out: list[tuple[str, tuple[int, ...], str]] = []
    for name, shape_json, dtype_str in rows:
        try:
            shape = tuple(json.loads(shape_json))
        except Exception:
            shape = tuple(eval(shape_json))
        out.append((name, shape, str(dtype_str or "float32")))
    return out


def _fetch_block_sketches(con: sqlite3.Connection, model_id: str, tensor_name: str) -> dict[int, bytes]:
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT block_id, sketch FROM blocks WHERE model_id=? AND tensor_name=?",
            (model_id, tensor_name),
        )
    except sqlite3.OperationalError:
        return {}
    return {
        int(bid): (sk if isinstance(sk, (bytes, bytearray)) else bytes(sk))
        for (bid, sk) in cur.fetchall()
    }


def _fetch_block_counts(con: sqlite3.Connection, model_id: str, tensor_name: str) -> int:
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT COUNT(1) FROM blocks WHERE model_id=? AND tensor_name=?",
            (model_id, tensor_name),
        )
        x = cur.fetchone()
        return int(x[0]) if x and x[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def _fetch_touch_ratio(con: sqlite3.Connection, expert_id: str, tensor_name: str) -> float | None:
    bm = get_touchmap_bitmap(con, expert_id, tensor_name)
    return touchmap_ratio_from_bitmap(bm)


# -----------------------------
# Sketch cosine (int8 vector)
# -----------------------------

def _sketch_to_vec(sk: bytes) -> np.ndarray | None:
    if not sk:
        return None
    arr = np.frombuffer(sk, dtype=np.int8)
    return arr.astype(np.float32, copy=False)


def _cos_from_sketch(a: bytes, b: bytes) -> float | None:
    va = _sketch_to_vec(a)
    vb = _sketch_to_vec(b)
    if va is None or vb is None or va.shape != vb.shape:
        return None
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float((va @ vb) / (na * nb))


def _mean_cos_from_blocks(base_sk: dict[int, bytes], exp_sk: dict[int, bytes]) -> float | None:
    if not base_sk or not exp_sk:
        return None
    inter = sorted(set(base_sk.keys()) & set(exp_sk.keys()))
    if not inter:
        return None
    cos_vals = []
    for bid in inter:
        c = _cos_from_sketch(base_sk[bid], exp_sk[bid])
        if c is not None:
            cos_vals.append(c)
    if not cos_vals:
        return None
    c_vals = np.array(cos_vals, dtype=np.float32)
    return float(np.mean(c_vals)) if c_vals.size else None


# -----------------------------
# Heuristics & IO estimation
# -----------------------------

_KEEP_HINTS = ("embed", "norm", "layernorm", ".ln", "rmsnorm")


def _pick_op_for_tensor(tname: str, cos_mean: float | None, tau: float, default_keep: bool = True) -> str:
    low = tname.lower()
    if any(h in low for h in _KEEP_HINTS) and default_keep:
        return "KEEP"
    if cos_mean is None:
        return "TIES"
    return "AVG" if cos_mean >= tau else "TIES"


def _num_blocks_from_shape(shape: tuple[int, ...], block_size: int) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(math.ceil(n / float(block_size)))


def _estimate_blocks_for_tensor(con: sqlite3.Connection, model_id: str, tname: str, shape: tuple[int, ...], block_size: int) -> int:
    cnt = _fetch_block_counts(con, model_id, tname)
    if cnt > 0:
        return cnt
    return _num_blocks_from_shape(shape, block_size)


def _estimate_tensor_base_rw_mb(nb_blocks: int, block_size: int, dtype_nbytes: int) -> float:
    """Unavoidable I/O in current merge pipeline: read base + write out."""
    if nb_blocks <= 0:
        return 0.0
    per_block = block_size * dtype_nbytes
    # base read + out write
    return (nb_blocks * per_block * 2) / (1024.0 * 1024.0)


def _estimate_tensor_expert_io_mb(nb_blocks: int, block_size: int, dtype_nbytes: int, n_experts: int, top_p: float) -> float:
    """Budget-controllable **expert extra I/O** (MB).

    This is what should respond to `io_budget_mb`.
    """
    if nb_blocks <= 0 or n_experts <= 0:
        return 0.0
    nb_sel = int(max(1, round(nb_blocks * float(max(0.0, min(1.0, top_p))))))
    per_block = block_size * dtype_nbytes
    return (nb_sel * per_block * int(n_experts)) / (1024.0 * 1024.0)


def _estimate_tensor_total_io_mb(nb_blocks: int, block_size: int, dtype_nbytes: int, n_experts: int, top_p: float) -> float:
    return _estimate_tensor_base_rw_mb(nb_blocks, block_size, dtype_nbytes) + _estimate_tensor_expert_io_mb(
        nb_blocks, block_size, dtype_nbytes, n_experts, top_p
    )


# -----------------------------
# Plan hashing (v1 core fields)
# -----------------------------

def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_plan_hash_v1(core: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(core)).hexdigest()


# -----------------------------
# Main: plan_auto
# -----------------------------

def plan_auto(
    con: sqlite3.Connection,
    base_model_id: str,
    experts: list[str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return merge_plan_v1:
      {
        "type": "merge_plan_v1",
        "plan_hash": "...",
        "plan_id": "...",   # stable derived from plan_hash[:16]
        "base_model": "...",
        "experts": [...],
        "policy": "...",
        "options": {...},   # execution-relevant only
        "tensors": [
            {"tensor_name": "...", "op": "KEEP|AVG|TIES|DARE", "top_p":..., "ties_thr":..., "dare_scale":...,
             "reason": {...} }  # reason NOT in hash core
        ]
      }
    """
    options = dict(options or {})

    # ----- normalize + defaults (stable keys) -----
    block_size = int(options.get("block_size", 4096))
    top_p = float(options.get("top_p", 0.35))
    ties_thr = float(options.get("ties_thr", 0.5))
    dare_scale = float(options.get("dare_scale", 0.8))

    io_budget_raw = options.get("io_budget_mb", None)
    io_budget_mb: float | None
    if io_budget_raw is None:
        io_budget_mb = None
    else:
        io_budget_mb = float(io_budget_raw)

    scoring = str(options.get("scoring", "auto"))
    tau = float(options.get("tau", 0.5))
    policy = str(options.get("policy", "conflict_aware"))
    disable_budget = bool(options.get("disable_budget", False))

    # ---- semantic options that affect execution ----
    # (do NOT add debug-only fields here)
    sem_options: dict[str, Any] = {
        "block_size": block_size,
        "top_p": top_p,
        "ties_thr": ties_thr,
        "dare_scale": dare_scale,
        "io_budget_mb": io_budget_mb,
        "scoring": scoring,
        "tau": tau,
        "policy": policy,
        "disable_budget": disable_budget,
        # executor hints (safe defaults; engine can ignore if not used)
        "sparse_output": bool(options.get("sparse_output", True)),
        "base_id": str(options.get("base_id", base_model_id)),
    }

    # ----- build per-tensor entries -----
    tensors_meta = _fetch_tensors(con, base_model_id)
    experts_sorted = list(experts)  # preserve input order
    n_experts = max(1, len(experts_sorted))

    tensor_entries: list[dict[str, Any]] = []
    total_before_mb = 0.0
    expert_before_mb = 0.0

    # (optional) print planner diagnostics in caller; keep planner pure here

    for (tname, shape, dtype_str) in tensors_meta:
        nb = _estimate_blocks_for_tensor(con, base_model_id, tname, shape, block_size)
        dt_bytes = _dtype_nbytes(dtype_str)

        io_total_mb = _estimate_tensor_total_io_mb(nb, block_size, dt_bytes, n_experts, top_p)
        io_expert_mb = _estimate_tensor_expert_io_mb(nb, block_size, dt_bytes, n_experts, top_p)

        # try sketch-cos
        base_sk = _fetch_block_sketches(con, base_model_id, tname)
        cos_vals = []
        for eid in experts_sorted:
            exp_sk = _fetch_block_sketches(con, eid, tname)
            cm = _mean_cos_from_blocks(base_sk, exp_sk)
            if cm is not None:
                cos_vals.append(cm)
        cos_mean = float(np.mean(np.array(cos_vals, dtype=np.float32))) if cos_vals else None

        op = _pick_op_for_tensor(tname, cos_mean, tau=tau, default_keep=True)

        reason: dict[str, Any] = {
            "policy": policy,
            "io_mb_est": float(io_total_mb),
            "expert_io_mb_est": float(io_expert_mb),
        }
        if cos_mean is not None:
            reason["cos_sim_mean"] = float(cos_mean)

        tensor_entries.append(
            {
                "tensor_name": tname,
                "op": op,
                "top_p": float(top_p),
                "ties_thr": float(ties_thr),
                "dare_scale": float(dare_scale),
                "dtype_bytes": int(dt_bytes),
                "reason": reason,
            }
        )

        total_before_mb += float(io_total_mb)
        expert_before_mb += float(io_expert_mb)

    # ----- enforce IO budget by scaling top_p on **expert extra IO** only -----
    # NOTE: With current merge pipeline, base read + out write is mostly unavoidable.
    #       Budget should constrain the additional expert reads, which is what affects
    #       quality/runtime trade-off.
    scale = 1.0
    if (not disable_budget) and (io_budget_mb is not None) and (io_budget_mb > 0) \
        and (expert_before_mb > io_budget_mb) and (expert_before_mb > 0):
        scale = max(0.0, min(1.0, float(io_budget_mb / expert_before_mb)))

        for te in tensor_entries:
            te["top_p"] = float(max(0.0, min(1.0, te["top_p"] * scale)))
            te["reason"]["io_budget_hit"] = True
            te["reason"]["top_p_scaled_by"] = float(scale)

        sem_options["top_p"] = float(max(0.0, min(1.0, sem_options["top_p"] * scale)))
        sem_options["io_budget_hit"] = True
        sem_options["top_p_scaled_by"] = float(scale)

    # After-scaling estimates (for logging / lineage)
    total_after_mb = 0.0
    expert_after_mb = 0.0

    # Build quick lookup for dtype/shape
    meta_map: dict[str, tuple[tuple[int, ...], str]] = {n: (sh, dt) for (n, sh, dt) in tensors_meta}

    for te in tensor_entries:
        tname = str(te.get("tensor_name"))
        meta = meta_map.get(tname, None)
        if meta is None:
            continue
        shape, dtype_str = meta
        nb = _estimate_blocks_for_tensor(con, base_model_id, tname, shape, block_size)
        dt_bytes = _dtype_nbytes(dtype_str)
        tp = float(te.get("top_p", top_p))

        total_after_mb += float(_estimate_tensor_total_io_mb(nb, block_size, dt_bytes, n_experts, tp))
        expert_after_mb += float(_estimate_tensor_expert_io_mb(nb, block_size, dt_bytes, n_experts, tp))

    sem_options["io_total_before_mb"] = float(total_before_mb)
    sem_options["io_total_after_mb"] = float(total_after_mb)
    sem_options["expert_io_before_mb"] = float(expert_before_mb)
    sem_options["expert_io_after_mb"] = float(expert_after_mb)
    sem_options["expert_io_scale"] = float(scale)
    if io_budget_mb is not None:
        sem_options["io_budget_scale_applied"] = float(scale)

    # ----- finalize plan -----
    plan: dict[str, Any] = {
        "type": "merge_plan_v1",
        "base_model": base_model_id,
        "experts": experts_sorted,
        "policy": policy,          # IMPORTANT: top-level for logging/lineage
        "options": sem_options,
        "tensors": tensor_entries,
    }

    # core fields for hash (exclude reason; keep stable)
    core = {
        "type": plan["type"],
        "base_model": plan["base_model"],
        "experts": plan["experts"],
        "options": plan["options"],
        "tensors": [
            {
                "tensor_name": t["tensor_name"],
                "op": t["op"],
                "top_p": t["top_p"],
                "ties_thr": t["ties_thr"],
                "dare_scale": t["dare_scale"],
            }
            for t in plan["tensors"]
        ],
    }
    plan_hash = _compute_plan_hash_v1(core)
    plan["plan_hash"] = plan_hash
    plan["plan_id"] = plan_hash[:16]

    return plan
