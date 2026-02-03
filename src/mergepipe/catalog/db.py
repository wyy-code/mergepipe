# mergepipe/catalog/db.py
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable
from typing import Any

# ---------- DDL ----------
DDL = [
    # models & adapters
    """
    CREATE TABLE IF NOT EXISTS models(
      model_id TEXT PRIMARY KEY,
      uri TEXT, arch TEXT, dtype TEXT,
      n_params INTEGER, created_at TEXT,
      parent_id TEXT, stage TEXT,
      root_hash TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS adapters(
      adapter_id TEXT PRIMARY KEY,
      base_model_id TEXT, type TEXT,
      rank INTEGER, uri TEXT, created_at TEXT
    );
    """,
    # tensors & blocks
    """
    CREATE TABLE IF NOT EXISTS tensors(
      model_id TEXT, name TEXT, shape TEXT, dtype TEXT,
      shard_id INTEGER DEFAULT 0, n_elem INTEGER,
      PRIMARY KEY (model_id, name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS blocks(
      model_id TEXT, tensor_name TEXT, block_id INTEGER,
      start_idx INTEGER, end_idx INTEGER,
      l2 REAL, max_abs REAL, sparsity REAL,
      sketch BLOB,
      PRIMARY KEY (model_id, tensor_name, block_id)
    );
    """,
    # 注意：touchmap 这里是“每张量的位图”，若未来需要 per-block 记录，可新增表而不是改这个表
    """
    CREATE TABLE IF NOT EXISTS touchmap(
      model_id TEXT, tensor_name TEXT, bitmap BLOB,
      PRIMARY KEY (model_id, tensor_name)
    );
    """,
    # diffs & plans
    """
    CREATE TABLE IF NOT EXISTS diffs(
      diff_id TEXT PRIMARY KEY,
      base_model_id TEXT, derived_model_id TEXT,
      granularity INTEGER, coverage REAL, uri TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS plans(
      plan_id TEXT PRIMARY KEY,
      base_model_id TEXT,
      experts_json TEXT,
      options_json TEXT,
      created_at TEXT, planner_version TEXT,
      plan_hash TEXT
    );
    """,
    # lineage / eval / exec logs / explain
    """
    CREATE TABLE IF NOT EXISTS lineage(
      node_id TEXT PRIMARY KEY, op TEXT,
      inputs_json TEXT, outputs_json TEXT,
      params_json TEXT, env_json TEXT, created_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS evals(
      model_id TEXT, task TEXT, metric TEXT,
      value REAL, dataset_hash TEXT, cfg_json TEXT, ts TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS exec_logs(
    plan_id TEXT, model_id TEXT, stage TEXT,
    io_mb REAL, wall_sec REAL, peak_mem_mb REAL, ts TEXT,

    -- I/O breakdown
    io_base_read_mb REAL,
    io_expert_read_mb REAL,
    io_out_write_mb REAL,

    -- budget behavior
    budget_used_mb REAL,
    skipped_reads INTEGER,

    -- quality proxy
    touched_ratio REAL,

    -- time breakdown
    planner_sec REAL,
    engine_sec REAL,
    flush_sec REAL,
    commit_sec REAL,

    -- metadata overhead
    db_size_mb REAL,
    manifest_kb REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS explain_entries(
      plan_id TEXT,
      tensor_name TEXT,
      op TEXT,
      reason_json TEXT,        -- {"policy": "...", "cos_sim_mean": ..., "tau_cos": ...}
      incremental_json TEXT,   -- {"touched_blocks": N, "top_p": 0.35, "io_mb_est": ...}
      created_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tensor_hashes(
      model_id TEXT,
      tensor_name TEXT,
      tensor_hash TEXT,
      PRIMARY KEY (model_id, tensor_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tensor_aliases(
      model_id TEXT,
      alias_name TEXT,
      target_name TEXT,
      reason TEXT,
      PRIMARY KEY (model_id, alias_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS block_hashes(
      model_id TEXT,
      tensor_name TEXT,
      block_id INTEGER,
      block_hash TEXT,
      PRIMARY KEY (model_id, tensor_name, block_id)
    );
    """,
]


# ---------- Connection & Schema ----------


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_schema(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    con.commit()
    _migrate_schema(con)


def _table_info(con: sqlite3.Connection, table: str) -> list[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def _col_absent(cols: list[str], name: str) -> bool:
    return name not in cols


def _migrate_schema(con: sqlite3.Connection) -> None:
    """
    按需“无损升级”：
      - plans: + policy TEXT, + tensors_json TEXT
      - diffs: + coverage_json TEXT（原 coverage REAL 继续保留，作为均值/占位）
      - models: + summary_json TEXT（便于存档 dtype/分片摘要，暂不强制使用）
    """
    cur = con.cursor()

    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS tensor_hashes(
        model_id TEXT,
        tensor_name TEXT,
        tensor_hash TEXT,
        PRIMARY KEY (model_id, tensor_name)
      );
    """
    )
    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS block_hashes(
        model_id TEXT,
        tensor_name TEXT,
        block_id INTEGER,
        block_hash TEXT,
        PRIMARY KEY (model_id, tensor_name, block_id)
      );
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tensor_aliases(
          model_id TEXT,
          alias_name TEXT,
          target_name TEXT,
          reason TEXT,
          PRIMARY KEY (model_id, alias_name)
        );
        """
    )

    # plans
    cols = _table_info(con, "plans")
    if _col_absent(cols, "policy"):
        cur.execute("ALTER TABLE plans ADD COLUMN policy TEXT")
    if _col_absent(cols, "tensors_json"):
        cur.execute("ALTER TABLE plans ADD COLUMN tensors_json TEXT")

    # diffs
    cols = _table_info(con, "diffs")
    if _col_absent(cols, "coverage_json"):
        cur.execute("ALTER TABLE diffs ADD COLUMN coverage_json TEXT")

    # models
    cols = _table_info(con, "models")
    if _col_absent(cols, "summary_json"):
        cur.execute("ALTER TABLE models ADD COLUMN summary_json TEXT")

    # exec_logs (Route B fields)
    cols = _table_info(con, "exec_logs")
    # I/O breakdown
    if _col_absent(cols, "io_base_read_mb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN io_base_read_mb REAL")
    if _col_absent(cols, "io_expert_read_mb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN io_expert_read_mb REAL")
    if _col_absent(cols, "io_out_write_mb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN io_out_write_mb REAL")

    # budget behavior
    if _col_absent(cols, "budget_used_mb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN budget_used_mb REAL")
    if _col_absent(cols, "skipped_reads"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN skipped_reads INTEGER")

    # quality proxy
    if _col_absent(cols, "touched_ratio"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN touched_ratio REAL")

    # time breakdown
    if _col_absent(cols, "planner_sec"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN planner_sec REAL")
    if _col_absent(cols, "engine_sec"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN engine_sec REAL")
    if _col_absent(cols, "flush_sec"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN flush_sec REAL")
    if _col_absent(cols, "commit_sec"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN commit_sec REAL")

    # metadata overhead
    if _col_absent(cols, "db_size_mb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN db_size_mb REAL")
    if _col_absent(cols, "manifest_kb"):
        cur.execute("ALTER TABLE exec_logs ADD COLUMN manifest_kb REAL")


    con.commit()


# ---------- REGISTER ----------


def register_model(
    con: sqlite3.Connection,
    model_id: str,
    uri: str,
    arch: str,
    dtype: str,
    tensors: Iterable[tuple[str, tuple[int, ...], str, int]],  # (name, shape, dtype, n_elem)
    n_params: int,
    created_at: str,
    parent_id: str | None = None,
    stage: str = "dev",
    summary: dict[str, Any] | None = None,
) -> None:
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO models(model_id, uri, arch, dtype, n_params, created_at, parent_id, stage, summary_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            model_id,
            uri,
            arch,
            dtype,
            n_params,
            created_at,
            parent_id,
            stage,
            json.dumps(summary or {}, ensure_ascii=False),
        ),
    )
    for name, shape, tdtype, n_elem in tensors:
        cur.execute(
            "INSERT OR REPLACE INTO tensors(model_id, name, shape, dtype, shard_id, n_elem) VALUES (?,?,?,?,?,?)",
            (model_id, name, json.dumps(shape), tdtype, 0, n_elem),
        )
    con.commit()


def register_adapter(
    con: sqlite3.Connection,
    adapter_id: str,
    base_model_id: str,
    a_type: str,
    rank: int,
    uri: str,
    created_at: str,
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO adapters(adapter_id, base_model_id, type, rank, uri, created_at) VALUES (?,?,?,?,?,?)",
        (adapter_id, base_model_id, a_type, rank, uri, created_at),
    )
    con.commit()


# ---------- DIFFS / TOUCHMAP ----------


def register_diff(
    con: sqlite3.Connection,
    base_model_id: str,
    derived_model_id: str,
    uri: str,
    granularity: int = 1,  # 0: tensor, 1: block, 2: element（保留）
    coverage_json: dict[str, Any] | None = None,
    diff_id: str | None = None,
) -> str:
    """
    将 base→derived 的 ΔW 容器注册到 diffs。
    - coverage_json: 可写 {"tensor_name": {"ratio": 0.42}} 或其他摘要
    - granularity: 建议 1（block）
    """
    if diff_id is None:
        diff_id = (
            "diff_"
            + hashlib.sha256(f"{base_model_id}->{derived_model_id}:{uri}".encode()).hexdigest()[:12]
        )

    # 兼容老列 coverage REAL：写入 coverage_json 的全局均值（若能算出 ratio 均值）
    coverage_mean: float | None = None
    if isinstance(coverage_json, dict) and coverage_json:
        vals = []
        for v in coverage_json.values():
            if isinstance(v, dict) and "ratio" in v:
                try:
                    vals.append(float(v["ratio"]))
                except Exception:
                    pass
        if vals:
            coverage_mean = float(sum(vals) / len(vals))

    con.execute(
        "INSERT OR REPLACE INTO diffs(diff_id, base_model_id, derived_model_id, granularity, coverage, uri, coverage_json) VALUES (?,?,?,?,?,?,?)",
        (
            diff_id,
            base_model_id,
            derived_model_id,
            int(granularity),
            coverage_mean if coverage_mean is not None else None,
            uri,
            json.dumps(coverage_json or {}, ensure_ascii=False),
        ),
    )
    con.commit()
    return diff_id


def save_touchmap_bitmap(
    con: sqlite3.Connection, model_or_adapter_id: str, tensor_name: str, bitmap: bytes
) -> None:
    """
    将触达位图（bit-per-block 或 byte-per-block）写入 touchmap。
    注意：当前表 key 为 (model_id, tensor_name)，这里沿用字段名 model_id。
    """
    con.execute(
        "INSERT OR REPLACE INTO touchmap(model_id, tensor_name, bitmap) VALUES (?, ?, ?)",
        (model_or_adapter_id, tensor_name, sqlite3.Binary(bitmap)),
    )
    con.commit()


# ---------- PLANS / EXPLAIN / LINEAGE / EXEC LOGS ----------


def save_plan(con: sqlite3.Connection, plan: dict[str, Any]) -> None:
    """
    将 planner 生成的 plan 落库到 plans：
      - experts_json：专家列表
      - options_json：planner 选项（含 io 预算缩放信息等）
      - policy：planner 采用的策略（conflict_aware/mvp_rule）
      - tensors_json：每张量的算子与理由（用于 EXPLAIN PLAN）
    """
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO plans(plan_id, base_model_id, experts_json, options_json, created_at, planner_version, plan_hash, policy, tensors_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            plan["plan_id"],
            plan["base_model"],
            json.dumps(plan.get("experts", []), ensure_ascii=False),
            json.dumps(plan.get("options", {}), ensure_ascii=False),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            plan.get("planner_version", "v0"),
            plan.get(
                "plan_hash", hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            ),
            plan.get("policy", "unknown"),
            json.dumps(plan.get("tensors", []), ensure_ascii=False),
        ),
    )
    con.commit()


def save_explain_entries(
    con: sqlite3.Connection, plan_id: str, entries: Iterable[dict[str, Any]]
) -> None:
    """
    entries: 迭代器，每条形如：
      {
        "tensor_name": "...",
        "op": "TIES",
        "reason": {...},        # dict → JSON
        "incremental": {...},   # dict → JSON
      }
    """
    cur = con.cursor()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for e in entries:
        cur.execute(
            "INSERT INTO explain_entries(plan_id, tensor_name, op, reason_json, incremental_json, created_at) VALUES (?,?,?,?,?,?)",
            (
                plan_id,
                e.get("tensor_name"),
                e.get("op"),
                json.dumps(e.get("reason", {}), ensure_ascii=False),
                json.dumps(e.get("incremental", {}), ensure_ascii=False),
                ts,
            ),
        )
    con.commit()


def save_lineage_op(
    con: sqlite3.Connection,
    op: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    params: dict[str, Any],
    env: dict[str, Any],
) -> str:
    """
    通用 LINEAGE 写入：返回 node_id。
    - op: "REGISTER" | "ANALYZE" | "MERGE" | "CALIB" | "EVAL" ...
    """
    node_id = f"node_{hashlib.sha256((op + str(time.time()) + json.dumps(inputs, sort_keys=True)).encode()).hexdigest()[:12]}"
    con.execute(
        "INSERT INTO lineage(node_id, op, inputs_json, outputs_json, params_json, env_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            node_id,
            op,
            json.dumps(inputs, ensure_ascii=False),
            json.dumps(outputs, ensure_ascii=False),
            json.dumps(params, ensure_ascii=False),
            json.dumps(env, ensure_ascii=False),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    con.commit()
    return node_id


def save_exec_log(
    con: sqlite3.Connection,
    plan_id: str,
    model_id: str,
    stage: str,
    io_mb: float,
    wall_sec: float,
    peak_mem_mb: float | None = None,
    *,
    io_base_read_mb: float | None = None,
    io_expert_read_mb: float | None = None,
    io_out_write_mb: float | None = None,
    budget_used_mb: float | None = None,
    skipped_reads: int | None = None,
    touched_ratio: float | None = None,
    planner_sec: float | None = None,
    engine_sec: float | None = None,
    flush_sec: float | None = None,
    commit_sec: float | None = None,
    db_size_mb: float | None = None,
    manifest_kb: float | None = None,
) -> None:
    """
    Route B: exec_logs 一次到位落库。旧调用不传新增字段也可用。
    """
    con.execute(
        """
        INSERT INTO exec_logs(
          plan_id, model_id, stage,
          io_mb, wall_sec, peak_mem_mb, ts,
          io_base_read_mb, io_expert_read_mb, io_out_write_mb,
          budget_used_mb, skipped_reads,
          touched_ratio,
          planner_sec, engine_sec, flush_sec, commit_sec,
          db_size_mb, manifest_kb
        ) VALUES (?,?,?,?,?,?,?,
                  ?,?,?, ?,?,
                  ?, ?,?,?,?,
                  ?,?)
        """,
        (
            plan_id,
            model_id,
            stage,
            float(io_mb),
            float(wall_sec),
            float(peak_mem_mb) if peak_mem_mb is not None else None,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            float(io_base_read_mb) if io_base_read_mb is not None else None,
            float(io_expert_read_mb) if io_expert_read_mb is not None else None,
            float(io_out_write_mb) if io_out_write_mb is not None else None,
            float(budget_used_mb) if budget_used_mb is not None else None,
            int(skipped_reads) if skipped_reads is not None else None,
            float(touched_ratio) if touched_ratio is not None else None,
            float(planner_sec) if planner_sec is not None else None,
            float(engine_sec) if engine_sec is not None else None,
            float(flush_sec) if flush_sec is not None else None,
            float(commit_sec) if commit_sec is not None else None,
            float(db_size_mb) if db_size_mb is not None else None,
            float(manifest_kb) if manifest_kb is not None else None,
        ),
    )
    con.commit()



def set_model_root_hash(con: sqlite3.Connection, model_id: str, root_hash: str) -> None:
    con.execute("UPDATE models SET root_hash=? WHERE model_id=?", (root_hash, model_id))
    con.commit()


def save_tensor_hashes(
    con: sqlite3.Connection, model_id: str, items: list[tuple[str, str]]
) -> None:
    """
    items: [(tensor_name, tensor_hash)]
    """
    cur = con.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO tensor_hashes(model_id, tensor_name, tensor_hash) VALUES (?,?,?)",
        [(model_id, tname, th) for (tname, th) in items],
    )
    con.commit()


def save_block_hashes(
    con: sqlite3.Connection, model_id: str, items: list[tuple[str, int, str]]
) -> None:
    """
    items: [(tensor_name, block_id, block_hash)]
    """
    cur = con.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO block_hashes(model_id, tensor_name, block_id, block_hash) VALUES (?,?,?,?)",
        [(model_id, tname, int(bid), bh) for (tname, bid, bh) in items],
    )
    con.commit()


# === Touchmap helpers (read) ===


def get_touchmap_bitmap(
    con: sqlite3.Connection, model_or_adapter_id: str, tensor_name: str
) -> bytes | None:
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT bitmap FROM touchmap WHERE model_id=? AND tensor_name=?",
            (model_or_adapter_id, tensor_name),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            # row[0] 可能是 memoryview，需要转 bytes
            return bytes(row[0])
        return None
    except sqlite3.OperationalError:
        return None


def touchmap_ratio_from_bitmap(bitmap: bytes | None) -> float | None:
    if not bitmap:
        return None
    # 统计置位 bit 数
    ones = 0
    for b in bitmap:
        ones += int(b).bit_count()
    # 位图长度 * 8 是“位数上限”；真实块数 planner 会用 blocks 表估计
    total_bits = len(bitmap) * 8
    if total_bits <= 0:
        return None
    return float(ones) / float(total_bits)

# === Convenience getters (NEW) =================================================

def get_model_uri(con: sqlite3.Connection, model_id: str) -> str | None:
    cur = con.cursor()
    cur.execute("SELECT uri FROM models WHERE model_id=?", (model_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_models_uri(con: sqlite3.Connection, model_ids: list[str]) -> dict[str, str]:
    if not model_ids:
        return {}
    cur = con.cursor()
    q = f"SELECT model_id, uri FROM models WHERE model_id IN ({','.join('?'*len(model_ids))})"
    cur.execute(q, model_ids)
    return {mid: uri for (mid, uri) in cur.fetchall() if uri}


def get_adapter(con: sqlite3.Connection, adapter_id: str) -> dict[str, Any] | None:
    cur = con.cursor()
    cur.execute(
        "SELECT adapter_id, base_model_id, type, rank, uri, created_at FROM adapters WHERE adapter_id=?",
        (adapter_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "adapter_id": row[0],
        "base_model_id": row[1],
        "type": row[2],
        "rank": row[3],
        "uri": row[4],
        "created_at": row[5],
    }


def get_adapters(con: sqlite3.Connection, adapter_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not adapter_ids:
        return {}
    cur = con.cursor()
    q = f"""
        SELECT adapter_id, base_model_id, type, rank, uri, created_at
        FROM adapters
        WHERE adapter_id IN ({','.join('?'*len(adapter_ids))})
    """
    cur.execute(q, adapter_ids)
    out: dict[str, dict[str, Any]] = {}
    for row in cur.fetchall():
        out[row[0]] = {
            "adapter_id": row[0],
            "base_model_id": row[1],
            "type": row[2],
            "rank": row[3],
            "uri": row[4],
            "created_at": row[5],
        }
    return out


def get_tensor_meta_map(con: sqlite3.Connection, model_id: str) -> dict[str, dict[str, Any]]:
    """
    返回 {tensor_name: {"shape": tuple[int,...], "dtype": str, "n_elem": int}}
    用于 I/O 估算或 Planner 兜底。
    """
    cur = con.cursor()
    cur.execute(
        "SELECT name, shape, dtype, n_elem FROM tensors WHERE model_id=?",
        (model_id,),
    )
    mp: dict[str, dict[str, Any]] = {}
    for name, shape_json, dtype, n_elem in cur.fetchall():
        try:
            shape = tuple(int(x) for x in json.loads(shape_json))
        except Exception:
            shape = ()
        mp[name] = {"shape": shape, "dtype": dtype, "n_elem": int(n_elem or 0)}
    return mp

def upsert_tensor_alias(
    con: sqlite3.Connection,
    model_id: str,
    alias_name: str,
    target_name: str,
    reason: str = "tied_weights",
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO tensor_aliases(model_id, alias_name, target_name, reason) VALUES (?,?,?,?)",
        (model_id, alias_name, target_name, reason),
    )
    con.commit()


def get_tensor_alias_map(con: sqlite3.Connection, model_id: str) -> dict[str, str]:
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT alias_name, target_name FROM tensor_aliases WHERE model_id=?",
            (model_id,),
        )
        return {a: t for (a, t) in cur.fetchall()}
    except sqlite3.OperationalError:
        return {}


def resolve_tensor_name(con: sqlite3.Connection, model_id: str, name: str) -> str:
    mp = get_tensor_alias_map(con, model_id)
    return mp.get(name, name)

def get_tensor_meta(con: sqlite3.Connection, model_id: str):
    cur = con.cursor()
    cur.execute("SELECT name, shape, dtype FROM tensors WHERE model_id=?", (model_id,))
    tshapes = {}
    tdtypes = {}
    for name, shape_json, dtype in cur.fetchall():
        try:
            tshapes[str(name)] = list(json.loads(shape_json))
        except Exception:
            tshapes[str(name)] = []
        tdtypes[str(name)] = str(dtype)
    return tshapes, tdtypes
