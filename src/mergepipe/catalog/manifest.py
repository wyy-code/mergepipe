# catalog/manifest.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon_json_bytes(obj: Any) -> bytes:
    # canonical JSON (stable bytes)
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return s.encode("utf-8")


@dataclass
class BlockEntry:
    block_id: int
    block_hash: str


@dataclass
class TensorEntry:
    name: str
    shape: List[int]
    dtype: str
    tensor_hash: str
    blocks: List[BlockEntry]


@dataclass
class MergeManifest:
    version: str
    commit_id: str
    model_id: str
    parents: List[str]

    plan_type: str
    plan_id: str
    plan_hash: str

    created_at: str
    tensors: List[TensorEntry]
    io_mb: Optional[float] = None
    wall_sec: Optional[float] = None

    # IMPORTANT:
    # - This field is NOT serialized into manifest.json on disk.
    # - It is computed as sha256(file_bytes_of_manifest.json).
    manifest_hash: Optional[str] = None

    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Dict that will be serialized to manifest.json.

        Strong invariant required by storage/transaction.py:
          manifest.manifest_hash == sha256(canonical_json_bytes(to_canonical_dict()))
        """
        d = asdict(self)
        # Do NOT include manifest_hash in the file.
        d.pop("manifest_hash", None)
        return d


def write_manifest(path: Union[str, Path], manifest: MergeManifest) -> str:
    """
    Write manifest.json marker and set manifest.manifest_hash to sha256(file_bytes).

    NOTE: manifest_hash is deliberately NOT embedded inside the json file,
          to avoid self-referential hashing problems.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    obj = manifest.to_canonical_dict()
    blob = _canon_json_bytes(obj)

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(p)

    mh = _sha256_bytes(blob)
    manifest.manifest_hash = mh
    return mh


def read_manifest(path: Union[str, Path]) -> MergeManifest:
    """
    Read manifest.json. Since manifest_hash is not stored in file,
    caller can compute it from file bytes if needed.
    """
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    m = MergeManifest(**obj)
    m.manifest_hash = _sha256_bytes(_canon_json_bytes(m.to_canonical_dict()))
    return m


def build_manifest_from_engine(
    *,
    commit_id: str,
    model_id: str,
    parents: List[str],
    plan_type: str,
    plan_id: str,
    plan_hash: str,
    engine_result: Dict[str, Any],
    tensor_shapes: Dict[str, List[int]],
    tensor_dtypes: Dict[str, str],
) -> MergeManifest:
    """
    Build manifest from engine_result.

    engine_result expected:
      - "tensor_hashes": [(tensor_name, tensor_hash), ...]
      - "block_hashes":  [(tensor_name, block_id, block_hash), ...]
    """
    blocks_by_tensor: Dict[str, List[BlockEntry]] = {}
    for (tname, bid, bh) in engine_result.get("block_hashes", []):
        tn = str(tname)
        blocks_by_tensor.setdefault(tn, []).append(
            BlockEntry(block_id=int(bid), block_hash=str(bh))
        )
    for tn in blocks_by_tensor:
        blocks_by_tensor[tn].sort(key=lambda x: x.block_id)

    tensor_hash_map = {str(n): str(h) for (n, h) in engine_result.get("tensor_hashes", [])}

    tensors: List[TensorEntry] = []
    for name, th in tensor_hash_map.items():
        tensors.append(
            TensorEntry(
                name=name,
                shape=list(tensor_shapes.get(name, [])),
                dtype=str(tensor_dtypes.get(name, "float32")),
                tensor_hash=th,
                blocks=blocks_by_tensor.get(name, []),
            )
        )
    tensors.sort(key=lambda x: x.name)

    return MergeManifest(
        version="manifest_v1",
        commit_id=commit_id,
        model_id=model_id,
        parents=list(parents),
        plan_type=str(plan_type),
        plan_id=str(plan_id),
        plan_hash=str(plan_hash),
        created_at=_now(),
        tensors=tensors,
        io_mb=float(engine_result.get("io_mb")) if engine_result.get("io_mb") is not None else None,
        wall_sec=float(engine_result.get("wall_sec")) if engine_result.get("wall_sec") is not None else None,
        manifest_hash=None,  # set by write_manifest()
    )
