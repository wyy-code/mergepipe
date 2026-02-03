"""mergepipe CLI

Drop-in replacement for `src/mergepipe/cli/app.py`.

This version fixes the concrete failures you reported:
- ImportError: missing `copy_tokenizer_files`.
- TypeError: plan_auto() unexpected kwargs (uses correct signature).
- TypeError: build_manifest_from_engine() unexpected kwargs (uses correct signature).
- TypeError: bytes not JSON serializable (encodes bytes when dumping debug JSON).
- CLI inconsistency: supports both `mergepipe merge ...` and `mergepipe ...`.

Assumptions based on the files you provided:
- DB helpers live in `mergepipe.catalog.db` with `connect()` and `get_model_uri()`.
- Planner exposes `plan_auto(con, base_model_id, experts, options)`.
- Engine exposes `run_merge_plan(...)` returning a dict (including `touchmaps` bytes).
- Storage primitives live in `mergepipe.storage.io`.
- Manifest builder is `mergepipe.catalog.manifest.build_manifest_from_engine(con, plan, run_dir, base_storage, expert_storages, out_storage, explain, stats)`.

If your package paths differ slightly, adjust the imports at the top.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# ===== project imports (match your attached files) =====
from mergepipe.catalog.db import connect, get_model_uri
from mergepipe.planner.planner import plan_auto
from mergepipe.engine.merge_engine import run_merge_plan
from mergepipe.catalog.manifest import build_manifest_from_engine, write_manifest
from mergepipe.storage.io import ShardedSafeTensorsStorage, WeightsAsDeltaProviderNP, _open_storage_from_uri


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
    help="MergePipe: budget-aware checkpoint merging",
)


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_base_model_files(base_dir: Path, out_dir: Path) -> None:
    """Copy the base model directory skeleton into out_dir.

    We keep this conservative and copy *files only* (no giant git caches etc.).
    ShardedSafeTensorsStorage requires shard files to exist in out_dir.
    """
    _ensure_dir(out_dir)

    # Copy top-level files
    for item in base_dir.iterdir():
        if item.is_file():
            dst = out_dir / item.name
            if not dst.exists():
                shutil.copy2(item, dst)

    # Copy tokenizer-related directories if present (some HF repos use subdirs)
    for sub in ["tokenizer", "spm", "sentencepiece", "assets"]:
        src_sub = base_dir / sub
        if src_sub.exists() and src_sub.is_dir():
            dst_sub = out_dir / sub
            if not dst_sub.exists():
                shutil.copytree(src_sub, dst_sub)


def _json_sanitize(obj: Any) -> Any:
    """Convert non-JSON-serializable types (notably bytes) into JSON-safe forms."""
    if isinstance(obj, bytes):
        # Use base64 to keep compact and reversible
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


# -----------------------------
# CLI entry
# -----------------------------

@app.callback()
def main(
    ctx: typer.Context,
    maybe_cmd: Optional[str] = typer.Argument(
        None,
        help="Optional command word. Accepts 'merge' for compatibility.",
    ),
    # ---- required args ----
    db: str = typer.Option(..., "--db", help="Path to mergedb sqlite file"),
    base: str = typer.Option(..., "--base", help="Base model id in DB"),
    experts: List[str] = typer.Option(
        ..., "--experts", help="Expert model id(s) in DB", show_default=False
    ),
    out: str = typer.Option(..., "--out", help="Output directory (HF repo)"),
    model_id: str = typer.Option(..., "--model-id", help="Logical model id for bookkeeping"),
    # ---- merge knobs (match script defaults) ----
    strategy: str = typer.Option("ties", "--strategy", help="Merge strategy: ties / dare / avg"),
    scoring: str = typer.Option("l2cos", "--scoring", help="Planner scoring function"),
    top_p: float = typer.Option(0.35, "--top-p", help="Top-p for sparse output"),
    ties_thr: float = typer.Option(0.6, "--ties-thr", help="TIES threshold"),
    dare_scale: float = typer.Option(0.0, "--dare-scale", help="DARE scale"),
    io_budget_mb: float = typer.Option(906.0, "--io-budget-mb", help="Expert I/O budget in MB"),
    block_size: int = typer.Option(640000, "--block-size", help="Block size in elements"),
    backend: str = typer.Option("pt", "--backend", help="Storage backend (pt)"),
    device: str = typer.Option("cpu", "--device", help="Execution device"),
    experts_kind: str = typer.Option(
        "weights",
        "--experts-kind",
        help="Expert kind: weights (delta provider reads full weights)",
    ),
    sparse_output: bool = typer.Option(
        True,
        "--sparse-output/--dense-output",
        help="Whether to write sparse output tensors (top-p masking)",
    ),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Run artifacts directory"),
    dump_debug: bool = typer.Option(
        True,
        "--dump-debug/--no-dump-debug",
        help="Write plan/engine debug JSON into runs/<rid>/",
    ),
) -> None:
    """Perform one merge.

    Compatibility note:
    - If your installed entrypoint wraps the CLI as a *single command*,
      calling `mergepipe merge --db ...` will pass `merge` as an extra argument.
      We accept that argument via `maybe_cmd`.
    """

    # Allow `mergepipe merge ...` when the entrypoint is single-command
    if maybe_cmd is not None and maybe_cmd != "merge":
        raise typer.BadParameter(
            f"Unexpected argument '{maybe_cmd}'. If you meant to run a merge, use 'merge' or omit it."
        )

    out_dir = Path(out)
    rid = Path(runs_dir) / f"{_timestamp()}_{model_id}"
    _ensure_dir(rid)

    # 1) Open DB and resolve model URIs
    con = connect(db)
    base_uri = get_model_uri(con, base)
    expert_uris = [get_model_uri(con, e) for e in experts]

    # 2) Ensure out dir has base shards / configs
    _copy_base_model_files(Path(base_uri), out_dir)

    # 3) Build storages
    base_storage = ShardedSafeTensorsStorage(base_uri, backend=backend, device=device, writable=False)
    expert_storages = [
        ShardedSafeTensorsStorage(u, backend=backend, device=device, writable=False)
        for u in expert_uris
    ]
    out_storage = ShardedSafeTensorsStorage(str(out_dir), backend=backend, device=device, writable=True)

    # 4) Plan
    opts = {
        "strategy": strategy,
        "top_p": top_p,
        "ties_thr": ties_thr,
        "dare_scale": dare_scale,
        "io_budget_mb": io_budget_mb,
        "block_size": block_size,
        "scoring": scoring,
        "experts_kind": experts_kind,
        "backend": backend,
        "device": device,
        "sparse_output": sparse_output,
    }

    t_plan0 = time.time()
    plan = plan_auto(con, base, experts, opts)
    t_plan1 = time.time()

    if dump_debug:
        _dump_json(rid / "plan.json", plan)

    # 5) Execute
    if experts_kind != "weights":
        raise typer.BadParameter(
            f"Unsupported --experts-kind={experts_kind!r} in this CLI. Expected 'weights'."
        )

    delta_providers = [WeightsAsDeltaProviderNP(s) for s in expert_storages]

    t_exec0 = time.time()
    engine_result = run_merge_plan(
        plan=plan,
        base_storage=base_storage,
        delta_providers=delta_providers,
        out_storage=out_storage,
        strategy=strategy,
        top_p=top_p,
        ties_thr=ties_thr,
        dare_scale=dare_scale,
        io_budget_mb=io_budget_mb,
        block_size=block_size,
        sparse_output=sparse_output,
    )
    t_exec1 = time.time()

    if dump_debug:
        _dump_json(rid / "engine_result.json", engine_result)

    # 6) Manifest (use the *actual* signature)
    explain = engine_result.get("explain", [])
    stats = engine_result

    # ---- Manifest (align with catalog/manifest.py signature) ----

    # 1) 给 manifest 一个 io_mb（可选但推荐）
    if "io_mb" not in engine_result and "io_bytes" in engine_result:
        engine_result["io_mb"] = float(engine_result["io_bytes"]) / (1024.0 * 1024.0)

    # 2) 只为 engine 输出涉及到的 tensor 补齐 shape/dtype
    tensor_names = [str(n) for (n, _) in engine_result.get("tensor_hashes", [])]
    tensor_shapes = {n: list(base_storage.tensor_shape(n)) for n in tensor_names}
    tensor_dtypes = {n: str(base_storage.dtype(n)) for n in tensor_names}

    # 3) CLI 模式下 commit/parents 做 best-effort（不依赖 DB commit 体系也能跑通）
    commit_id = rid.name                 # 例如：20260203_103459_qwen3-0.6B
    parents = [base] + list(experts)     # 这里先用 model_id 列表占位

    manifest_obj = build_manifest_from_engine(
        commit_id=commit_id,
        model_id=model_id,
        parents=parents,
        plan_type=str(plan.get("type", "merge_plan_v1")),
        plan_id=str(plan.get("plan_id", "")),
        plan_hash=str(plan.get("plan_hash", "")),
        engine_result=engine_result,
        tensor_shapes=tensor_shapes,
        tensor_dtypes=tensor_dtypes,
    )

    # 4) 用官方写入函数，确保 manifest_hash 规则正确（且不写入 json 内部）
    write_manifest(rid / "manifest.json", manifest_obj)



    # 7) Persist run artifacts
    # Keep manifest small and JSON-safe.
    # _dump_json(rid / "manifest.json", manifest_obj)
    _dump_json(
        rid / "timing.json",
        {
            "plan_wall_sec": t_plan1 - t_plan0,
            "exec_wall_sec": t_exec1 - t_exec0,
            "total_wall_sec": (t_plan1 - t_plan0) + (t_exec1 - t_exec0),
        },
    )

    # A small stdout summary (script redirects full logs anyway)
    typer.echo(
        f"[OK] merged base={base} experts={experts} -> {out_dir} | run={rid} | "
        f"plan={t_plan1 - t_plan0:.3f}s exec={t_exec1 - t_exec0:.3f}s"
    )


if __name__ == "__main__":
    # Allows: python -m mergepipe.cli.app [merge] --db ...
    app()
