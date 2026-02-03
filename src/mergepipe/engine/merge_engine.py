# mergepipe/engine/merge_engine.py
from __future__ import annotations

import hashlib
import time
from typing import Any, List

import numpy as np

from mergepipe.storage.io import (
    _TORCH_OK,
    DeltaProvider,
    StorageBase,
    torch as _torch,
    # NEW: budget helpers (you added them into storage/io.py in previous step)
    IOBudgetGate,
    load_block_budgeted,
    io_budget_summary,
)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_update_str(h: "hashlib._Hash", s: str) -> None:
    h.update(s.encode("utf-8"))


def _tensor_hash_from_block_hashes(block_hashes_for_tensor: List[str]) -> str:
    """
    Rolling / merkle-style tensor hash:
      tensor_hash = sha256(concat(block_hash_i))
    Avoid re-reading full tensor after writing blocks.
    """
    h = hashlib.sha256()
    for bh in block_hashes_for_tensor:
        _sha256_update_str(h, bh)
    return h.hexdigest()


def _normalize_keep_ratio(x: float) -> float:
    v = float(x)
    if v > 1.0:
        v = v / 100.0
    return float(max(0.0, min(1.0, v)))


def _base_ref_block_hash(base_id: str, tensor_name: str, block_id: int) -> str:
    """
    Stable pseudo-hash for blocks that are NOT materialized (overlay reference to base).
    This prevents "plan_hash/manifest_hash instability" when we skip reading bytes.

    IMPORTANT:
      - This is NOT the content hash of the base block bytes.
      - It encodes a stable logical reference:
          ("BASE_REF", base_id, tensor_name, block_id)
      - That is enough for MergePipe lineage / snapshot identity.
    """
    s = f"BASE_REF|{base_id}|{tensor_name}|{int(block_id)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# TIES (numpy)
# -----------------------------

def _ties_topk_mask_np(deltas: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, float]:
    """
    deltas: [E, B]
    keep_ratio: [0,1]
    return: masked_deltas, active_ratio
    """
    E, B = deltas.shape
    k = int(max(1, round(B * keep_ratio)))
    absd = np.abs(deltas)
    kth = np.partition(absd, B - k, axis=1)[:, B - k]  # [E]
    mask = absd >= kth[:, None]
    masked = deltas * mask
    active_ratio = float(mask.sum()) / float(E * B)
    return masked, active_ratio


def _ties_resolve_sign_np(masked: np.ndarray) -> np.ndarray:
    sums = masked.sum(axis=0)  # [B]
    sign = np.sign(sums).astype(np.float32)
    zeros = sign == 0
    if np.any(zeros):
        maj = np.sign(np.sum(np.sign(masked), axis=0)).astype(np.float32)
        maj[maj == 0] = 1.0
        sign[zeros] = maj[zeros]
    sign[sign == 0] = 1.0
    return sign


def _ties_disjoint_mean_np(masked: np.ndarray, sign: np.ndarray) -> np.ndarray:
    agree = np.sign(masked) == sign[None, :]
    disjoint = masked * agree
    denom = np.count_nonzero(disjoint, axis=0).astype(np.float32)
    denom = np.maximum(denom, 1.0)
    return disjoint.sum(axis=0).astype(np.float32) / denom


# -----------------------------
# TIES (torch)
# -----------------------------

def _ties_topk_mask_pt(deltas: "_torch.Tensor", keep_ratio: float) -> tuple["_torch.Tensor", float]:
    E, B = deltas.shape
    k = int(max(1, round(B * keep_ratio)))
    absd = _torch.abs(deltas)
    kth, _ = _torch.kthvalue(absd, k=B - k + 1, dim=1)
    mask = absd >= kth.unsqueeze(1)
    masked = deltas * mask
    active_ratio = float(mask.sum().item()) / float(E * B)
    return masked, active_ratio


def _ties_resolve_sign_pt(masked: "_torch.Tensor") -> "_torch.Tensor":
    sums = masked.sum(dim=0)
    sign = _torch.sign(sums)
    zeros = sign == 0
    if zeros.any():
        maj = _torch.sign(_torch.sum(_torch.sign(masked), dim=0))
        maj[maj == 0] = 1
        sign[zeros] = maj[zeros]
    sign[sign == 0] = 1
    return sign.to(dtype=_torch.float32)


def _ties_disjoint_mean_pt(masked: "_torch.Tensor", sign: "_torch.Tensor") -> "_torch.Tensor":
    agree = _torch.sign(masked) == sign.unsqueeze(0)
    disjoint = masked * agree
    denom = _torch.count_nonzero(disjoint, dim=0).to(dtype=_torch.float32).clamp(min=1.0)
    return disjoint.sum(dim=0).to(dtype=_torch.float32) / denom


# -----------------------------
# Plan schema validation
# -----------------------------

def _validate_merge_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dict")
    if plan.get("type") != "merge_plan_v1":
        raise ValueError(f"Unsupported plan type: {plan.get('type')!r} (expected 'merge_plan_v1')")

    for k in ("plan_hash", "base_model", "experts", "tensors", "options"):
        if k not in plan:
            raise KeyError(f"plan missing required field: {k}")

    if not isinstance(plan["tensors"], list):
        raise TypeError("plan['tensors'] must be a list")

    for i, t in enumerate(plan["tensors"]):
        if not isinstance(t, dict):
            raise TypeError(f"plan['tensors'][{i}] must be a dict")
        if "tensor_name" not in t or "op" not in t:
            raise KeyError(f"plan['tensors'][{i}] missing 'tensor_name'/'op'")


# -----------------------------
# Executor (transaction-aware: no publish/rename here)
# -----------------------------

def run_merge_plan(
    plan: dict[str, Any],
    base_storage: StorageBase,
    delta_providers: list[DeltaProvider],
    out_storage: StorageBase,
    block_size: int = 4096,
    scoring: str = "auto",
    backend: str = "np",
    device: str | None = None,
    do_flush: bool = True,
    **_compat_kwargs: Any,  # swallow legacy args like atomic_out_dir
) -> dict[str, Any]:
    """
    Transaction-aware executor:
      - Reads base from base_storage
      - Computes deltas via delta_providers.delta_block(...)
      - Writes merged blocks to out_storage (staging / overlay)
      - Does NOT publish/rename final dir (handled by transaction manager)

    NEW: IO budget is truly enforced by enabling sparse / overlay output:
      - For KEEP/top_p-skip/budget-skip blocks:
          * do NOT read base bytes
          * do NOT write out
          * record a stable "base_ref_block_hash"
      - Only touched blocks are materialized (read + delta + write)

    Returns:
      {
        wall_sec,
        io_bytes,
        io_bytes_base_read,
        io_bytes_expert_read,
        io_bytes_out_write,
        io_budget: {...},
        tensor_hashes: [(tensor_name, tensor_hash)],
        block_hashes: [(tensor_name, block_id, block_hash)],
        touchmaps: {tensor_name: bitmap_bytes},
        explain: [...]
      }
    """
    _validate_merge_plan(plan)

    if backend == "pt":
        if not _TORCH_OK:
            raise RuntimeError("backend='pt' requires torch")
        _t = _torch

    t0 = time.time()

    # I/O accounting breakdown (logical bytes, consistent with budget estimator)
    base_read_bytes = 0
    expert_read_bytes = 0
    out_write_bytes = 0

    # --- budget gate ---
    opts = dict(plan.get("options", {}))
    io_budget_mb = float(opts.get("io_budget_mb", 0.0) or 0.0)
    gate: IOBudgetGate | None = None
    if io_budget_mb > 0:
        gate = IOBudgetGate(int(io_budget_mb * 1024 * 1024))

    # --- crucial switch: sparse/overlay output ---
    sparse_output = bool(opts.get("sparse_output", True))

    # base id for stable base-ref hashing
    base_id = str(opts.get("base_id") or plan.get("base_model") or getattr(base_storage, "model_id", "base"))

    block_hashes: list[tuple[str, int, str]] = []
    tensor_hashes: list[tuple[str, str]] = []
    explain: list[dict[str, Any]] = []
    touchmaps: dict[str, bytes] = {}

    block_size = int(block_size)

    def _block_keep_by_top_p(tensor_name: str, block_id: int, top_p: float) -> bool:
        """Deterministic sampling (no IO needed). True => KEEP(base ref)."""
        if top_p >= 1.0:
            return False
        h = hashlib.sha256(f"{tensor_name}:{block_id}".encode("utf-8")).digest()
        r = int.from_bytes(h[:4], "little") / 2**32
        return bool(r > top_p)

    for te in plan["tensors"]:
        name = te["tensor_name"]
        op = str(te["op"]).upper()

        # per-tensor params (stable defaults)
        top_p = float(te.get("top_p", opts.get("top_p", 1.0)))
        ties_thr = float(te.get("ties_thr", opts.get("ties_thr", 0.5)))
        dare_scale = float(te.get("dare_scale", opts.get("dare_scale", 0.8)))

        dtype_bytes = int(te.get("dtype_bytes", 4))
        if dtype_bytes <= 0:
            dtype_bytes = 4

        # metadata
        shape = base_storage.tensor_shape(name)
        numel = 1
        for d in shape:
            numel *= int(d)

        n_blocks = int((numel + block_size - 1) // block_size)
        bitmap = bytearray((n_blocks + 7) // 8)

        tensor_block_hashes: list[str] = []
        ties_active_acc: list[float] = []

        for bid in range(n_blocks):
            s = bid * block_size
            e = min(s + block_size, numel)

            # --- skip decisions that do NOT require reading base ---
            top_p_keep = _block_keep_by_top_p(name, bid, top_p)
            op_keep = (op == "KEEP")

            if sparse_output and (top_p_keep or op_keep):
                # logical base reference, no IO, no write
                bh = _base_ref_block_hash(base_id, name, bid)
                block_hashes.append((name, bid, bh))
                tensor_block_hashes.append(bh)
                continue

            # --- read base block (budgeted) ---
            est_base_bytes = (e - s) * dtype_bytes
            if gate is not None and not gate.try_charge(est_base_bytes):
                if sparse_output:
                    gate.mark_skipped()
                    bh = _base_ref_block_hash(base_id, name, bid)
                    block_hashes.append((name, bid, bh))
                    tensor_block_hashes.append(bh)
                    continue
                # fallback: still read (but this breaks strict budgeting; last resort)
                # NOTE: we intentionally do NOT charge again to avoid negative budget.

            base_blk = base_storage.load_block(name, s, e)
            base_read_bytes += est_base_bytes

            # --- compute deltas (budget-aware) ---
            deltas = []
            if op not in ("KEEP",) and delta_providers:
                per_expert_est = (e - s) * dtype_bytes

                if gate is not None and not gate.try_charge(per_expert_est):
                    gate.mark_skipped()
                    if sparse_output:
                        bh = _base_ref_block_hash(base_id, name, bid)
                        block_hashes.append((name, bid, bh))
                        tensor_block_hashes.append(bh)
                        continue

                    # materialize-full fallback: write base
                    out_storage.write_block(name, s, e, base_blk)
                    out_write_bytes += (e - s) * dtype_bytes

                    if backend == "pt":
                        bh = _hash_bytes(
                            base_blk.detach().to("cpu", dtype=_t.float32).contiguous().numpy().tobytes()
                        )
                    else:
                        bh = _hash_bytes(np.asarray(base_blk, dtype=np.float32).tobytes())
                    block_hashes.append((name, bid, bh))
                    tensor_block_hashes.append(bh)
                    continue
                else:
                    for j, dp in enumerate(delta_providers):
                        if gate is not None and j > 0:
                            if not gate.try_charge(per_expert_est):
                                gate.mark_skipped()
                                break
                        d = dp.delta_block(name, s, e, base_blk)
                        deltas.append(d)
                        expert_read_bytes += per_expert_est  # logical accounting

            # --- merge ---
            used = op
            ties_active = None

            if op == "KEEP" or not deltas:
                merged = base_blk
                used = "KEEP" if op == "KEEP" else "BASE"
            elif op == "AVG":
                if backend == "pt":
                    merged = base_blk + _t.stack(deltas, dim=0).mean(dim=0)
                else:
                    merged = (
                        np.asarray(base_blk, dtype=np.float32)
                        + np.stack([np.asarray(x, dtype=np.float32) for x in deltas], axis=0).mean(axis=0)
                    )
            elif op == "DARE":
                if backend == "pt":
                    merged = base_blk + _t.stack(deltas, dim=0).mean(dim=0) * float(dare_scale)
                else:
                    merged = (
                        np.asarray(base_blk, dtype=np.float32)
                        + np.stack([np.asarray(x, dtype=np.float32) for x in deltas], axis=0).mean(axis=0)
                        * float(dare_scale)
                    )
            elif op == "TIES":
                keep_ratio = _normalize_keep_ratio(float(ties_thr))
                if backend == "pt":
                    D = _t.stack(deltas, dim=0)
                    masked, active = _ties_topk_mask_pt(D, keep_ratio)
                    sign = _ties_resolve_sign_pt(masked)
                    md = _ties_disjoint_mean_pt(masked, sign)
                    merged = base_blk + md
                    ties_active = float(active)
                else:
                    D = np.stack([np.asarray(x, dtype=np.float32) for x in deltas], axis=0)
                    masked, active = _ties_topk_mask_np(D, keep_ratio)
                    sign = _ties_resolve_sign_np(masked)
                    md = _ties_disjoint_mean_np(masked, sign)
                    merged = np.asarray(base_blk, dtype=np.float32) + md
                    ties_active = float(active)
                ties_active_acc.append(float(ties_active))
            else:
                raise ValueError(f"Unknown op: {op}")

            # touched only if we truly applied non-KEEP and had deltas
            touched = (used not in ("KEEP", "BASE"))
            if touched:
                bitmap[bid // 8] |= 1 << (bid % 8)

            # If sparse_output and NOT touched => keep base reference and avoid writing
            if sparse_output and not touched:
                bh = _base_ref_block_hash(base_id, name, bid)
                block_hashes.append((name, bid, bh))
                tensor_block_hashes.append(bh)
                continue

            # --- write materialized block ---
            out_storage.write_block(name, s, e, merged)
            out_write_bytes += (e - s) * dtype_bytes

            # block hash (content hash for materialized blocks)
            if backend == "pt":
                bh = _hash_bytes(merged.detach().to("cpu", dtype=_t.float32).contiguous().numpy().tobytes())
            else:
                bh = _hash_bytes(np.asarray(merged, dtype=np.float32).tobytes())

            block_hashes.append((name, bid, bh))
            tensor_block_hashes.append(bh)

        tensor_hashes.append((name, _tensor_hash_from_block_hashes(tensor_block_hashes)))
        touchmaps[name] = bytes(bitmap)

        explain.append(
            {
                "tensor_name": name,
                "op": op,
                "policy": "conflict_aware" if scoring in ("auto", "l2cos") else "mvp_rule",
                "ties_active_mean": float(np.mean(np.array(ties_active_acc, dtype=np.float32))) if ties_active_acc else None,
                "top_p": float(top_p),
                "ties_thr": float(ties_thr),
                "dare_scale": float(dare_scale),
                "sparse_output": bool(sparse_output),
                "io_budget_mb": float(io_budget_mb),
            }
        )

    if do_flush:
        # out_storage in overlay mode may still need flush to persist written blocks
        out_storage.flush(None)

    io_budget = io_budget_summary(gate)

    io_bytes = int(base_read_bytes + expert_read_bytes + out_write_bytes)

    return {
        "wall_sec": float(time.time() - t0),
        "io_bytes": int(io_bytes),
        "io_bytes_base_read": int(base_read_bytes),
        "io_bytes_expert_read": int(expert_read_bytes),
        "io_bytes_out_write": int(out_write_bytes),
        "io_budget": io_budget,
        "tensor_hashes": tensor_hashes,
        "block_hashes": block_hashes,
        "touchmaps": touchmaps,
        "explain": explain,
    }
