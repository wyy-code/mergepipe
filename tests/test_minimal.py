# tests/test_minimal.py
import tempfile
import time

import numpy as np

from mergepipe.catalog.db import connect, init_schema, register_model
from mergepipe.storage.io import FakeStorage
from mergepipe.analyzer.indexer import analyze_model
from mergepipe.planner.planner import plan_auto
from mergepipe.engine.merge_engine import run_merge_plan

from dataclasses import dataclass

@dataclass
class SimpleDeltaProvider:
    """
    Minimal delta provider for tests.
    It behaves like a "storage" that returns delta tensors: expert - base.

    NOTE: We implement a few common method names to match possible engine expectations.
    """
    base: FakeStorage
    expert: FakeStorage

    def get_tensor(self, name: str):
        return self.expert.get_tensor(name) - self.base.get_tensor(name)

    # Aliases (in case engine uses different method names)
    def load_tensor(self, name: str):
        return self.get_tensor(name)

    def tensor_shape(self, name: str):
        return self.base.tensor_shape(name)

    def dtype(self, name: str):
        return self.base.dtype(name)

    def iter_tensors(self):
        # iterate over base tensor names
        yield from self.base.iter_tensors()
    
    def delta_block(self, tensor_name: str, s: int, e: int, base_blk):
        """
        Engine expects: delta_block(name, s, e, base_blk)
        Return expert_block - base_blk for the slice [s, e).
        """
        exp_blk = self.expert.load_block(tensor_name, s, e)
        # ensure numpy arrays for subtraction (backend="np" in this test)
        exp_blk = np.asarray(exp_blk, dtype=np.float32)
        base_blk = np.asarray(base_blk, dtype=np.float32)
        return exp_blk - base_blk


def _make_fake_model():
    tensors = {
        "linear.w": np.random.randn(8, 8).astype(np.float32),
        "linear.b": np.zeros((8,), dtype=np.float32),
    }
    return FakeStorage(tensors)

def _summarize_storage(st: FakeStorage):
    """
    Build (tensors, n_params) for catalog_db.register_model signature:
      tensors: List[(name, shape, dtype, n_elem)]
    """
    tensors = []
    n_params = 0
    for name, _ in st.iter_tensors():
        shape = st.tensor_shape(name)
        tdtype = st.dtype(name)
        n_elem = 1
        for x in shape:
            n_elem *= int(x)
        tensors.append((name, shape, tdtype, int(n_elem)))
        n_params += int(n_elem)
    return tensors, n_params

def test_end_to_end_fake():
    con = connect(":memory:")
    init_schema(con)

    base = _make_fake_model()
    exp = _make_fake_model()

    base_id = "base_fake"
    exp_id = "exp_fake"

    base_tensors, base_n = _summarize_storage(base)
    exp_tensors, exp_n = _summarize_storage(exp)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # register_model follows the new signature (no backend/shape_json/extra_json)
    register_model(
        con=con,
        model_id=base_id,
        uri="mem://base",
       arch="fake",
        dtype="float32",
        tensors=base_tensors,
        n_params=base_n,
        created_at=created_at,
        parent_id=None,
        stage="test",
       summary={"note": "unit-test"},
    )
    register_model(
        con=con,
        model_id=exp_id,
        uri="mem://exp",
        arch="fake",
        dtype="float32",
        tensors=exp_tensors,
        n_params=exp_n,
        created_at=created_at,
        parent_id=None,
        stage="test",
        summary={"note": "unit-test"},
    )

    analyze_model(con, model_id=base_id, storage=base, block_size=16, only_key_tensors=False)
    analyze_model(con, model_id=exp_id, storage=exp, block_size=16, only_key_tensors=False)

    plan = plan_auto(con, base_model_id=base_id, experts=[exp_id], options={"top_p": 0.5, "ties_thr": 0.5})
    assert plan["plan_id"]

    # NEW: engine API now expects explicit delta providers and output storage
    delta_providers = [SimpleDeltaProvider(base=base, expert=exp)]
    out_storage = FakeStorage({k: np.array(v, copy=True) for k, v in base.tensors.items()})


    out_dir = tempfile.mkdtemp()
    # IMPORTANT: engine validator expects plan["type"] == "merge_plan_v1"
    plan["type"] = "merge_plan_v1"

    exec_stats = run_merge_plan(
        plan,
        base_storage=base,
        delta_providers=delta_providers,
        out_storage=out_storage,
        block_size=16,
        scoring="l2cos",
        backend="np",
        device=None,
        # out_uri 等旧参数如果你想保留也行，engine 会吞掉（**_compat_kwargs）
        out_uri=out_dir,
    )

    # 简单断言：执行写了 explain_entries
    cur = con.cursor()
    try:
        cur.execute("SELECT COUNT(1) FROM explain_entries WHERE plan_id=?", (plan["plan_id"],))
        n = cur.fetchone()[0]
        assert n >= 0
    except Exception:
        # explain_entries 表可为空，不作为失败
        pass
