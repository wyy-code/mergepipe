# storage/transaction.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from catalog.db import connect, init_schema
from catalog.manifest import MergeManifest


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fsync_dir(p: Path) -> None:
    """
    Best-effort directory fsync for durability (POSIX).
    """
    try:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_publish_dir(staging_dir: Path, final_dir: Path) -> None:
    """
    Atomically publish staging_dir -> final_dir.
    If final exists, move it aside then replace, then delete old.
    """
    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    backup_dir: Optional[Path] = None
    if final_dir.exists():
        backup_dir = final_dir.with_name(final_dir.name + f".old.{uuid.uuid4().hex[:8]}")
        final_dir.rename(backup_dir)
        _fsync_dir(final_dir.parent)

    staging_dir.rename(final_dir)
    _fsync_dir(final_dir.parent)

    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _ensure_commits_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS commits (
            commit_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            parents_json TEXT NOT NULL,
            status TEXT NOT NULL,
            staging_dir TEXT NOT NULL,
            final_dir TEXT NOT NULL,
            created_at REAL NOT NULL,
            committed_at REAL,
            manifest_hash TEXT,
            manifest_json TEXT,
            error TEXT
        )
        """
    )


class MergeTransaction:
    """
    Transaction wrapper:
      staging: outputs_root/_staging/<commit_id>/
      final:   outputs_root/<model_id>/

    SQLite catalog records commit lifecycle & embeds manifest payload.
    """

    def __init__(
        self,
        *,
        db_path: str,
        outputs_root: str,
        model_id: str,
        parents: list[str] | None = None,
    ):
        self.db_path = db_path
        self.outputs_root = Path(outputs_root)
        self.model_id = model_id
        self.parents = parents or []

        self.commit_id = uuid.uuid4().hex
        self.staging_dir = self.outputs_root / "_staging" / self.commit_id
        self.final_dir = self.outputs_root / self.model_id

        self._active = False

        con = connect(self.db_path)
        try:
            init_schema(con)
            _ensure_commits_table(con)
            con.commit()
        finally:
            con.close()

    def begin(self) -> str:
        if self._active:
            raise RuntimeError("Transaction already active")

        self.staging_dir.mkdir(parents=True, exist_ok=False)

        con = connect(self.db_path)
        try:
            _ensure_commits_table(con)
            con.execute(
                """
                INSERT INTO commits
                (commit_id, model_id, parents_json, status, staging_dir, final_dir, created_at)
                VALUES (?, ?, ?, 'PENDING', ?, ?, ?)
                """,
                (
                    self.commit_id,
                    self.model_id,
                    json.dumps(self.parents, ensure_ascii=False),
                    str(self.staging_dir),
                    str(self.final_dir),
                    time.time(),
                ),
            )
            con.commit()
        finally:
            con.close()

        self._active = True
        return self.commit_id

    def attach_manifest(self, *, manifest_hash: str, manifest_json: str) -> None:
        if not self._active:
            raise RuntimeError("Transaction not active")
        con = connect(self.db_path)
        try:
            con.execute(
                """
                UPDATE commits
                SET manifest_hash=?, manifest_json=?
                WHERE commit_id=?
                """,
                (manifest_hash, manifest_json, self.commit_id),
            )
            con.commit()
        finally:
            con.close()

    def attach_manifest_obj(
        self, manifest: MergeManifest, *, manifest_path: Optional[Path] = None
    ) -> None:
        """
        Attach manifest both as:
          - a file marker in staging (manifest.json)
          - an embedded DB payload in commits.manifest_json

        Invariant: commits.manifest_hash == sha256(file_bytes_of_manifest_json)
        """
        if not self._active:
            raise RuntimeError("Transaction not active")
        if manifest.manifest_hash is None:
            raise ValueError("manifest.manifest_hash is None; write_manifest() should have set it")

        if manifest_path is None:
            manifest_path = self.staging_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found at {manifest_path}")

        blob = manifest_path.read_bytes()
        file_hash = _sha256_bytes(blob)

        if str(manifest.manifest_hash) != file_hash:
            raise ValueError(
                "manifest_hash mismatch: manifest.manifest_hash != sha256(manifest.json bytes)"
            )

        # Store exactly what is on disk (decoded as UTF-8)
        self.attach_manifest(
            manifest_hash=file_hash,
            manifest_json=blob.decode("utf-8"),
        )

    def mark_error(self, err: str) -> None:
        con = connect(self.db_path)
        try:
            con.execute("UPDATE commits SET error=? WHERE commit_id=?", (err, self.commit_id))
            con.commit()
        finally:
            con.close()

    def _require_manifest_attached(self) -> tuple[str, str]:
        """
        Return (manifest_hash, manifest_json) if attached; otherwise raise.
        Also validate staging marker presence and hash match.
        """
        con = connect(self.db_path)
        try:
            row = con.execute(
                "SELECT manifest_hash, manifest_json FROM commits WHERE commit_id=?",
                (self.commit_id,),
            ).fetchone()
        finally:
            con.close()

        if not row:
            raise RuntimeError("Commit record not found")
        mh, mj = row
        if not mh or not mj:
            raise RuntimeError("Manifest not attached. Call attach_manifest_obj() before commit().")

        manifest_path = self.staging_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"manifest.json marker missing in staging dir: {manifest_path}")

        blob = manifest_path.read_bytes()
        if _sha256_bytes(blob) != str(mh):
            raise RuntimeError("manifest.json sha256 does not match commits.manifest_hash")

        return str(mh), str(mj)

    def commit(self) -> Path:
        if not self._active:
            raise RuntimeError("Transaction not active")

        mh, _mj = self._require_manifest_attached()

        con = connect(self.db_path)
        try:
            con.execute("UPDATE commits SET status='COMMITTING' WHERE commit_id=?", (self.commit_id,))
            con.commit()
        finally:
            con.close()

        _atomic_publish_dir(self.staging_dir, self.final_dir)

        # Post-condition: final snapshot must contain manifest marker and hash match.
        final_manifest = self.final_dir / "manifest.json"
        if not final_manifest.exists():
            self.mark_error(f"manifest.json marker missing in final dir: {final_manifest}")
            raise RuntimeError(f"manifest.json marker missing in final dir: {final_manifest}")

        blob = final_manifest.read_bytes()
        if _sha256_bytes(blob) != mh:
            self.mark_error("final manifest sha256 != commits.manifest_hash")
            raise RuntimeError("final manifest sha256 != commits.manifest_hash")

        con = connect(self.db_path)
        try:
            con.execute(
                """
                UPDATE commits
                SET status='COMMITTED', committed_at=?
                WHERE commit_id=?
                """,
                (time.time(), self.commit_id),
            )
            con.commit()
        finally:
            con.close()

        self._active = False
        return self.final_dir

    def abort(self) -> None:
        if not self._active:
            return

        shutil.rmtree(self.staging_dir, ignore_errors=True)

        con = connect(self.db_path)
        try:
            con.execute("UPDATE commits SET status='ABORTED' WHERE commit_id=?", (self.commit_id,))
            con.commit()
        finally:
            con.close()

        self._active = False

    @staticmethod
    def recover_all(*, db_path: str, outputs_root: str) -> dict[str, Any]:
        """
        Recovery:
          - find commits with status IN ('PENDING','COMMITTING')
          - if final_dir exists -> mark COMMITTED
          - else delete staging dir (safe under outputs_root) and mark ABORTED
        """
        con = connect(db_path)
        try:
            init_schema(con)
            _ensure_commits_table(con)

            rows = con.execute(
                "SELECT commit_id, staging_dir, final_dir, status FROM commits "
                "WHERE status IN ('PENDING','COMMITTING')"
            ).fetchall()

            cleaned = 0
            fixed = 0
            root = Path(outputs_root).resolve()

            for cid, sdir, fdir, status in rows:
                sdir_p = Path(sdir)
                fdir_p = Path(fdir)

                if fdir_p.exists():
                    con.execute(
                        "UPDATE commits SET status='COMMITTED', committed_at=? WHERE commit_id=?",
                        (time.time(), cid),
                    )
                    fixed += 1
                    continue

                # best-effort cleanup staging (only if under outputs_root)
                try:
                    if root in sdir_p.resolve().parents and sdir_p.exists():
                        shutil.rmtree(sdir_p, ignore_errors=True)
                        cleaned += 1
                except Exception:
                    if sdir_p.exists():
                        shutil.rmtree(sdir_p, ignore_errors=True)
                        cleaned += 1

                con.execute("UPDATE commits SET status='ABORTED' WHERE commit_id=?", (cid,))

            con.commit()
            return {"pending_or_committing": len(rows), "cleaned": cleaned, "fixed_to_committed": fixed}
        finally:
            con.close()
