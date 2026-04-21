"""Job queue for the stem-remover feature (ADR-SR-001 — separate SQLite DB).

Parallel to `src/jobs.py` but completely isolated: different DB file, different
schema, different stage vocabulary. Zero shared state with the chord-sheet
flow.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StemsStage = Literal[
    "uploaded",
    "queued",
    "separating",
    "encoding",
    "ready",
    "error",
    "cancelled",
]

TERMINAL_STAGES: tuple[StemsStage, ...] = ("ready", "error", "cancelled")
ACTIVE_STAGES: tuple[StemsStage, ...] = ("separating", "encoding")


@dataclass
class StemsJob:
    id: str
    filename: str
    stage: StemsStage
    progress: float
    remove_mask: tuple[str, ...]
    input_sha256: str
    bitrate: int
    output_path: str | None
    error: str | None
    created_ts: float
    updated_ts: float
    heartbeat_ts: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stems_jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    remove_mask TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    bitrate INTEGER NOT NULL,
    output_path TEXT,
    error TEXT,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    heartbeat_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stems_stage ON stems_jobs(stage);
CREATE INDEX IF NOT EXISTS idx_stems_updated ON stems_jobs(updated_ts DESC);
CREATE INDEX IF NOT EXISTS idx_stems_sha ON stems_jobs(input_sha256);
"""


def _row_to_job(row: sqlite3.Row) -> StemsJob:
    return StemsJob(
        id=row["id"],
        filename=row["filename"],
        stage=row["stage"],
        progress=row["progress"],
        remove_mask=tuple(json.loads(row["remove_mask"])),
        input_sha256=row["input_sha256"],
        bitrate=row["bitrate"],
        output_path=row["output_path"],
        error=row["error"],
        created_ts=row["created_ts"],
        updated_ts=row["updated_ts"],
        heartbeat_ts=row["heartbeat_ts"],
    )


class StemsJobRepo:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA)

    def create(
        self,
        filename: str,
        remove_mask: tuple[str, ...],
        input_sha256: str,
        bitrate: int,
    ) -> StemsJob:
        now = time.time()
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO stems_jobs("
                "id, filename, stage, progress, remove_mask, input_sha256, "
                "bitrate, output_path, error, created_ts, updated_ts, heartbeat_ts"
                ") VALUES (?, ?, 'uploaded', 0.0, ?, ?, ?, NULL, NULL, ?, ?, ?)",
                (
                    job_id,
                    filename,
                    json.dumps(sorted(remove_mask)),
                    input_sha256,
                    int(bitrate),
                    now,
                    now,
                    now,
                ),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> StemsJob | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM stems_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_recent(self, limit: int = 50) -> list[StemsJob]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM stems_jobs ORDER BY updated_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def list_older_than(self, cutoff_ts: float) -> list[StemsJob]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM stems_jobs WHERE updated_ts < ?",
                (cutoff_ts,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def next_queued(self) -> StemsJob | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM stems_jobs WHERE stage = 'queued' "
                "ORDER BY updated_ts ASC LIMIT 1"
            ).fetchone()
        return _row_to_job(row) if row else None

    def advance(
        self,
        job_id: str,
        stage: StemsStage,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            current = self._conn.execute(
                "SELECT progress FROM stems_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            current_progress = current["progress"] if current else 0.0
            new_progress = current_progress if progress is None else progress
            self._conn.execute(
                "UPDATE stems_jobs SET stage=?, progress=?, error=?, "
                "updated_ts=?, heartbeat_ts=? WHERE id=?",
                (stage, new_progress, error, now, now, job_id),
            )

    def heartbeat(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE stems_jobs SET heartbeat_ts=? WHERE id=?",
                (time.time(), job_id),
            )

    def set_output_path(self, job_id: str, output_path: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE stems_jobs SET output_path=?, updated_ts=? WHERE id=?",
                (output_path, now, job_id),
            )

    def cancel(self, job_id: str, reason: str = "cancelled by user") -> bool:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT stage FROM stems_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                return False
            if row["stage"] in TERMINAL_STAGES:
                return False
            self._conn.execute(
                "UPDATE stems_jobs SET stage='cancelled', error=?, "
                "updated_ts=?, heartbeat_ts=? WHERE id=?",
                (reason, now, now, job_id),
            )
        return True

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM stems_jobs WHERE id = ?", (job_id,))

    def reap_stale(self, older_than_s: float = 30 * 60.0) -> int:
        cutoff = time.time() - older_than_s
        placeholders = ",".join("?" for _ in ACTIVE_STAGES)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE stems_jobs SET stage='error', error='interrupted', "
                f"updated_ts=? "
                f"WHERE stage IN ({placeholders}) AND heartbeat_ts < ?",
                (time.time(), *ACTIVE_STAGES, cutoff),
            )
        return cur.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
