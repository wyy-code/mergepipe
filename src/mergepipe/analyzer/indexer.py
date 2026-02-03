# mergepipe/analyzer/indexer.py
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable

import numpy as np

try:
    import torch  # 可选，用于 backend="pt"

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

from mergepipe.storage.io import StorageBase
from mergepipe.catalog import db as catalog_db

# 覆盖注意力/MLP/Embedding/Norm/Proj + 输出头（lm_head / classifier 等）
KEY_TENSOR_HINTS = ("attn", "mlp", "embed", "norm", "proj", "lm_head", "head")
KEY_TENSOR_EXACT = {
    "lm_head.weight",
    "output.weight",
    "classifier.weight",
    "score.weight",
}

def is_key_tensor(name: str) -> bool:
    n = name.lower()

    # 1) 关键张量：输出头（很多模型不包含 proj/norm 关键词）
    if n in KEY_TENSOR_EXACT:
        return True
    if n.endswith("lm_head.weight") or "lm_head" in n:
        return True
    if n.endswith("output.weight") or ".output" in n:
        return True
    if "classifier" in n or n.endswith("score.weight"):
        return True

    # 2) 常规规则
    return any(h in n for h in KEY_TENSOR_HINTS)


def make_blocks(shape: tuple[int, ...], block_size: int) -> Iterable[tuple[int, int, int]]:
    n_elem = int(np.prod(shape))
    block_id = 0
    for start in range(0, n_elem, block_size):
        end = min(start + block_size, n_elem)
        yield block_id, start, end
        block_id += 1


# -------- 轻量随机投影 sketch（CountSketch 风格）--------


def _countsketch_numpy(vec: np.ndarray, dim: int = 16, seed: int = 1234) -> bytes:
    n = vec.size
    if n == 0:
        return b""
    rng = np.random.default_rng(seed + n)
    idx = rng.integers(0, dim, size=n, endpoint=False)
    sgn = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    acc = np.zeros(dim, dtype=np.float32)
    for b in range(dim):
        mask = idx == b
        if np.any(mask):
            acc[b] = np.sum(vec[mask].astype(np.float32, copy=False) * sgn[mask])
    std = float(np.std(acc)) + 1e-6
    q = np.clip(np.round(acc / std * 10.0), -128, 127).astype(np.int8)
    return q.tobytes()


def _countsketch_torch(vec: torch.Tensor, dim: int = 16, seed: int = 1234) -> bytes:
    n = int(vec.numel())
    if n == 0:
        return b""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + n)
    idx = torch.randint(low=0, high=dim, size=(n,), generator=g, device="cpu")
    sgn = torch.where(
        torch.rand(n, generator=g) < 0.5,
        torch.tensor(-1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    v = vec.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    acc = torch.zeros(dim, dtype=torch.float32)
    for b in range(dim):
        m = idx == b
        if torch.any(m):
            acc[b] = torch.sum(v[m] * sgn[m])
    std = float(torch.std(acc)) + 1e-6
    q = torch.clamp(torch.round(acc / std * 10.0), -128, 127).to(dtype=torch.int8)
    return q.numpy().tobytes()


def randproj_sketch(vec, dim: int = 16, seed: int = 1234) -> bytes:
    if _TORCH_OK and isinstance(vec, torch.Tensor):
        return _countsketch_torch(vec, dim=dim, seed=seed)
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return _countsketch_numpy(v, dim=dim, seed=seed)


# -------- 主入口：分析并落库 --------


def analyze_model(
    con: sqlite3.Connection,
    model_id: str,
    storage: StorageBase,
    block_size: int = 4096,
    only_key_tensors: bool = True,
) -> None:
    """
    - 仅用 tensors 表的 (name, shape) 元数据枚举张量；
    - 对每个块通过 storage.load_block 流式读取（同一张量配合存储层 RO cache 只读一次）；
    - 兼容 numpy / torch 两种返回类型；
    - 写入 l2 / max_abs / sparsity / sketch；
    - 加块级进度日志与批量 commit。
    """
    cur = con.cursor()
    cur.execute("SELECT name, shape FROM tensors WHERE model_id=?", (model_id,))
    rows = cur.fetchall()

    # --- auto alias for tied weights / missing exports ---
    names = {r[0] for r in rows}
    if "lm_head.weight" not in names:
        # 常见：lm_head 和 embed_tokens tied，只导出了 embed_tokens
        # 这里尽量泛化：找任意以 embed_tokens.weight 结尾的张量名
        embed_candidates = [n for n in names if n.endswith("embed_tokens.weight")]
        if embed_candidates:
            target = embed_candidates[0]  # 通常只有一个：model.embed_tokens.weight
            catalog_db.upsert_tensor_alias(con, model_id, "lm_head.weight", target, reason="tied_or_missing")
            # 同时把 tensors 表补一行，让 planner/统计能看到 311/311
            # shape/dtype/n_elem 用 target 的
            cur2 = con.cursor()
            cur2.execute(
                "SELECT shape, dtype, n_elem FROM tensors WHERE model_id=? AND name=?",
                (model_id, target),
            )
            r = cur2.fetchone()
            if r:
                shape_json, dtype, n_elem = r
                cur2.execute(
                    "INSERT OR REPLACE INTO tensors(model_id, name, shape, dtype, shard_id, n_elem) VALUES (?,?,?,?,?,?)",
                    (model_id, "lm_head.weight", shape_json, dtype, 0, n_elem),
                )
                con.commit()
                # rows/names 更新一下，避免本次 analyze 还看不到
                cur.execute("SELECT name, shape FROM tensors WHERE model_id=?", (model_id,))
                rows = cur.fetchall()
                names = {rr[0] for rr in rows}


    if not rows:
        print(f"[ANALYZE] tensors table empty for model_id={model_id}. Did you run REGISTER?")
        return

    t0 = time.time()
    n_tensors = 0
    n_blocks = 0

    print(
        f"[ANALYZE] start: {len(rows)} tensors, block_size={block_size}, only_key_tensors={only_key_tensors}"
    )

    for i, (name, shape_json) in enumerate(rows, 1):
        shape = tuple(json.loads(shape_json))
        if only_key_tensors and not is_key_tensor(name):
            continue

        # 对超大 embedding 提高块大小以减少块数（不影响其他层）
        local_block = block_size
        lowname = name.lower()
        if "embed" in lowname and block_size < (1 << 20):  # 1Mi 元素
            local_block = 1 << 20

        if i == 1 or i % 5 == 0:
            print(f"[ANALYZE] tensor {i}/{len(rows)}: {name} shape={shape} block={local_block}")

        blk_cnt = 0
        for block_id, start, end in make_blocks(shape, local_block):
            # 如果本 tensor 在该模型上是 alias，就用 target 去读
            real_name = catalog_db.resolve_tensor_name(con, model_id, name)
            blk = storage.load_block(real_name, start, end)

            # numpy / torch 统计
            if _TORCH_OK and isinstance(blk, torch.Tensor):
                v = blk.detach()
                l2 = float(torch.linalg.vector_norm(v))
                max_abs = float(torch.max(torch.abs(v))) if v.numel() > 0 else 0.0
                sparsity = (
                    float(torch.mean((torch.abs(v) <= 1e-12).to(torch.float32)))
                    if v.numel() > 0
                    else 0.0
                )
            else:
                v = np.asarray(blk, dtype=np.float32)
                l2 = float(np.linalg.norm(v))
                max_abs = float(np.max(np.abs(v))) if v.size > 0 else 0.0
                sparsity = float(np.mean(np.abs(v) <= 1e-12)) if v.size > 0 else 0.0

            sketch = randproj_sketch(blk, dim=16)

            cur.execute(
                "INSERT OR REPLACE INTO blocks(model_id, tensor_name, block_id, start_idx, end_idx, l2, max_abs, sparsity, sketch) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (model_id, name, block_id, start, end, l2, max_abs, sparsity, sketch),
            )
            n_blocks += 1
            blk_cnt += 1

            # 每 200 块提交一次，同时给点进度
            if n_blocks % 200 == 0:
                con.commit()
            if blk_cnt % 2000 == 0:
                print(f"[ANALYZE]   {name}: processed {blk_cnt} blocks")

        n_tensors += 1

    con.commit()
    print(f"[ANALYZE] done: tensors={n_tensors}, blocks={n_blocks}, elapsed={time.time()-t0:.2f}s")
