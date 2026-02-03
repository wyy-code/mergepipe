import sqlite3
import numpy as np

from mergepipe.catalog import db as catalog_db
from mergepipe.storage.io import FakeStorage
from mergepipe.analyzer.indexer import analyze_model

def test_db_init_and_analyze_smoke(tmp_path):
    db_path = tmp_path / "t.sqlite"
    con = sqlite3.connect(str(db_path))
    catalog_db.init_schema(con)

    # register minimal fake model
    tensors = [("w", (16,), "float32", 16)]
    catalog_db.register_model(
        con=con,
        model_id="m0",
        uri="fake://m0",
        arch="fake",
        dtype="float32",
        tensors=tensors,
        n_params=16,
        created_at="now",
        parent_id=None,
        stage="test",
        summary={},
    )

    st = FakeStorage({"w": np.arange(16, dtype=np.float32)}, backend="np")
    analyze_model(con, model_id="m0", storage=st, block_size=8, only_key_tensors=False)
    con.close()
