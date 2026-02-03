# cli/explain.py
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple


def _popcount_bytes(b: bytes) -> int:
    return sum(bin(x).count("1") for x in b)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_plan(cur: sqlite3.Cursor, plan_id: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT base_model_id, experts_json, options_json, policy, tensors_json, created_at
        FROM plans
        WHERE plan_id=?
        """,
        (plan_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    base_id, experts_json, options_json, policy, tensors_json, created_at = row
    experts = json.loads(experts_json or "[]")
    options = json.loads(options_json or "{}")
    tensors = json.loads(tensors_json or "[]")
    return {
        "base": base_id,
        "experts": experts,
        "options": options,
        "policy": policy,
        "tensors": tensors,
        "created_at": created_at,
    }


def _print_header(plan_id: str, plan: dict[str, Any]) -> None:
    print(f"== EXPLAIN PLAN {plan_id} ==")
    print(f"base: {plan['base']}")
    print(f"experts: {plan['experts']}")
    print(f"policy: {plan.get('policy', '?')}")
    print(f"created_at: {plan.get('created_at', '?')}")
    print("")


def _format_reason(d: dict[str, Any]) -> str:
    # 将 plan.tensors[i].reason 里的关键信息拍平展示
    keys = [
        "io_before_mb",
        "io_after_mb",
        "budget_hit",
        "guard",
        "reason",
        "io_est_mb_tensor",
        "selected_ratio",
        "ties_active_ratio_mean",
    ]
    parts: List[str] = []
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, (int, float)):
                if k.startswith("io"):
                    parts.append(f"{k}={float(v):.2f}")
                elif "ratio" in k:
                    parts.append(f"{k}={float(v):.2%}")
                else:
                    parts.append(f"{k}={float(v):.3f}")
            else:
                parts.append(f"{k}={v}")
    return " | ".join(parts)


def _format_tensor_meta(t: dict[str, Any]) -> str:
    meta: List[str] = []
    if "cos_sim_mean" in t:
        meta.append(f"cos={_safe_float(t['cos_sim_mean']):.3f}")
    if "top_p" in t:
        meta.append(f"top_p={_safe_float(t['top_p']):.2f}")
    if "ties_thr" in t:
        meta.append(f"thr={_safe_float(t['ties_thr']):.3f}")
    if "scale_max" in t:
        meta.append(f"scale={_safe_float(t['scale_max']):.2f}")
    # 从 reason 里抓 io 估计（若 planner/engine 已写）
    r = t.get("reason", {})
    if isinstance(r, dict):
        if "io_before_mb" in r:
            meta.append(f"io_before={_safe_float(r['io_before_mb']):.2f}MB")
        if "io_after_mb" in r:
            meta.append(f"io_after={_safe_float(r['io_after_mb']):.2f}MB")
        if "budget_hit" in r:
            meta.append(f"budget_hit={bool(r['budget_hit'])}")
    return " | ".join(meta)


def _print_tensors(plan: dict[str, Any], show: int, show_reason: bool) -> None:
    tensors: List[dict[str, Any]] = list(plan.get("tensors", []))
    n_all = len(tensors)
    n_show = min(max(show, 0), n_all)
    print(f"-- tensors ({n_show}/{n_all}) --")
    for t in tensors[:n_show]:
        op = t.get("op", "?")
        name = t.get("name", "?")
        meta = _format_tensor_meta(t)
        print(f"{op:>5}  {name}   {meta}")
        if show_reason and isinstance(t.get("reason"), dict):
            reason_text = _format_reason(t["reason"])
            if reason_text:
                print(f"       reason: {reason_text}")
    print("")


def _fetch_explain_entries(
    cur: sqlite3.Cursor, plan_id: str
) -> List[Tuple[str, str, dict, dict]]:
    try:
        cur.execute(
            """
            SELECT tensor_name, op, reason_json, incremental_json
            FROM explain_entries
            WHERE plan_id=?
            """,
            (plan_id,),
        )
    except sqlite3.OperationalError:
        return []
    rows = cur.fetchall()
    out: List[Tuple[str, str, dict, dict]] = []
    for tname, op, rj, ij in rows:
        try:
            r = json.loads(rj or "{}")
        except Exception:
            r = {}
        try:
            inc = json.loads(ij or "{}")
        except Exception:
            inc = {}
        out.append((tname or "?", op or "?", r, inc))
    return out


def _merge_view_aggregate(
    entries: Iterable[Tuple[str, str, dict, dict]]
) -> List[Tuple[str, Dict[str, float], Dict[str, int]]]:
    """
    聚合 explain_entries：
      sel: 平均 selected_ratio（触达率）
      ties: 平均 ties_active_ratio_mean
      io: 平均 io_est_mb_tensor
      ops_count: 每个张量各 op 的样本数（用于展示 DARE/AVG/TIES 的样本条目数）
    """
    agg: Dict[str, Dict[str, float]] = {}
    cnt: Dict[str, int] = {}
    ops: Dict[str, Dict[str, int]] = {}

    for tname, op, _r, inc in entries:
        name = tname
        sel = _safe_float(inc.get("selected_ratio"))
        ties = _safe_float(inc.get("ties_active_ratio_mean"))
        io_mb = _safe_float(inc.get("io_est_mb_tensor"))
        a = agg.setdefault(name, {"sel": 0.0, "ties": 0.0, "io": 0.0})
        a["sel"] += sel
        a["ties"] += ties
        a["io"] += io_mb
        cnt[name] = cnt.get(name, 0) + 1
        ops.setdefault(name, {}).setdefault(op, 0)
        ops[name][op] += 1

    result: List[Tuple[str, Dict[str, float], Dict[str, int]]] = []
    for name, sums in agg.items():
        c = max(1, cnt.get(name, 1))
        avg = {"sel": sums["sel"] / c, "ties": sums["ties"] / c, "io": sums["io"] / c}
        result.append((name, avg, ops.get(name, {})))
    result.sort(key=lambda x: x[1].get("io", 0.0), reverse=True)
    return result


def _print_merge_view(cur: sqlite3.Cursor, plan_id: str, show: int, sample: int) -> None:
    entries = _fetch_explain_entries(cur, plan_id)
    if not entries:
        print("[Merge-View] no explain_entries for this plan (engine may not have written them).")
        print("")
        return

    agg = _merge_view_aggregate(entries)
    n_all = len(agg)
    n_show = min(max(show, 0), n_all)

    print(f"== Merge View (aggregated by tensor) ({n_show}/{n_all}) ==")
    print("metric: sel≈selected_ratio | ties≈ties_active_ratio_mean | io≈io_est_mb_tensor")
    for name, m, opcnt in agg[:n_show]:
        parts = [f"sel≈{m['sel']:.2%}", f"ties≈{m['ties']:.2%}", f"io≈{m['io']:.2f}MB"]
        if opcnt:
            parts.append("ops=" + ", ".join(f"{k}:{v}" for k, v in sorted(opcnt.items())))
        print(f"{name}   " + " | ".join(parts))
    print("")

    if sample and sample > 0:
        random.shuffle(entries)
        keep = entries[:max(1, sample)]
        print(f"== Samples (n={len(keep)}) ==")
        for tname, op, _r, inc in keep:
            meta: List[str] = []
            if "top_p" in inc:
                meta.append(f"top_p={_safe_float(inc['top_p']):.2f}")
            if "selected_ratio" in inc:
                meta.append(f"sel={_safe_float(inc['selected_ratio']):.2%}")
            if "ties_active_ratio_mean" in inc:
                meta.append(f"ties≈{_safe_float(inc['ties_active_ratio_mean']):.2%}")
            if "io_est_mb_tensor" in inc:
                meta.append(f"io≈{_safe_float(inc['io_est_mb_tensor']):.2f}MB")
            print(f"{op:>5}  {tname}   {' | '.join(meta)}")
        print("")


def _dump_json(plan_id: str, plan: dict[str, Any], cur: sqlite3.Cursor, merge_view: bool, show: int) -> None:
    out: Dict[str, Any] = {
        "plan_id": plan_id,
        "header": {
            "base": plan.get("base"),
            "experts": plan.get("experts", []),
            "policy": plan.get("policy"),
            "created_at": plan.get("created_at"),
            "options": plan.get("options", {}),
        },
        "tensors": plan.get("tensors", []),
    }
    if merge_view:
        entries = _fetch_explain_entries(cur, plan_id)
        agg = _merge_view_aggregate(entries)
        n_show = min(max(show, 0), len(agg))
        out["merge_view"] = [
            {"tensor": name, "avg": avg, "ops_count": opcnt} for name, avg, opcnt in agg[:n_show]
        ]
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain a saved merge plan; show Plan/Merge views; support JSON export."
    )
    parser.add_argument("--db", required=True, help="Path to mergepipe sqlite file")
    parser.add_argument("--plan", required=True, help="Plan id, e.g. plan_xxx")
    parser.add_argument("--show", type=int, default=60, help="How many tensors to show")
    parser.add_argument("--show-reason", action="store_true", help="Print per-tensor reason")
    parser.add_argument(
        "--merge-view",
        action="store_true",
        help="Show merge-view aggregated by tensor (touch/ratio/io from explain_entries)",
    )
    parser.add_argument("--sample", type=int, default=0, help="Sample N explain entries")
    parser.add_argument("--json", action="store_true", help="Export as JSON (to stdout)")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    try:
        cur = con.cursor()
        plan = _load_plan(cur, args.plan)
        if not plan:
            print(f"[ERR] plan {args.plan} not found")
            return

        if args.json:
            _dump_json(args.plan, plan, cur, merge_view=args.merge_view, show=args.show)
            return

        _print_header(args.plan, plan)
        _print_tensors(plan, show=args.show, show_reason=args.show_reason)

        if args.merge_view:
            _print_merge_view(cur, args.plan, show=args.show, sample=args.sample)
    finally:
        con.close()


if __name__ == "__main__":
    main()
