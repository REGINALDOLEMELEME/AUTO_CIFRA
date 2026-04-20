from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Stage = Literal[
    "uploaded",
    "queued",
    "separating",
    "transcribing",
    "aligning",
    "chords",
    "structure",
    "rendering",
    "ready_for_review",
    "exported",
    "error",
    "cancelled",
]

TERMINAL_STAGES: tuple[Stage, ...] = ("ready_for_review", "exported", "error", "cancelled")

ACTIVE_STAGES: tuple[Stage, ...] = (
    "separating",
    "transcribing",
    "aligning",
    "chords",
    "structure",
    "rendering",
)


@dataclass
class Job:
    id: str
    filename: str
    stage: Stage
    progress: float
    error: str | None
    transpose_semitones: int
    capo_fret: int
    created_ts: float
    updated_ts: float
    heartbeat_ts: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    transpose_semitones INTEGER NOT NULL DEFAULT 0,
    capo_fret INTEGER NOT NULL DEFAULT 0,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    heartbeat_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_stage ON jobs(stage);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_ts DESC);
"""


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        filename=row["filename"],
        stage=row["stage"],
        progress=row["progress"],
        error=row["error"],
        transpose_semitones=row["transpose_semitones"],
        capo_fret=row["capo_fret"],
        created_ts=row["created_ts"],
        updated_ts=row["updated_ts"],
        heartbeat_ts=row["heartbeat_ts"],
    )


class JobRepo:
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

    def create(self, filename: str) -> Job:
        now = time.time()
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id, filename, stage, progress, created_ts, updated_ts, heartbeat_ts) "
                "VALUES (?, ?, 'uploaded', 0.0, ?, ?, ?)",
                (job_id, filename, now, now, now),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY updated_ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def next_queued(self) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE stage = 'queued' "
                "ORDER BY updated_ts ASC LIMIT 1"
            ).fetchone()
        return _row_to_job(row) if row else None

    def advance(
        self,
        job_id: str,
        stage: Stage,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            current = self._conn.execute(
                "SELECT progress FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            current_progress = current["progress"] if current else 0.0
            new_progress = current_progress if progress is None else progress
            self._conn.execute(
                "UPDATE jobs SET stage=?, progress=?, error=?, updated_ts=?, heartbeat_ts=? "
                "WHERE id=?",
                (stage, new_progress, error, now, now, job_id),
            )

    def heartbeat(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET heartbeat_ts=? WHERE id=?", (time.time(), job_id)
            )

    def cancel(self, job_id: str, reason: str = "cancelled by user") -> bool:
        """Mark a job as cancelled. Returns True if the job existed and was
        in a cancellable state."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT stage FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                return False
            if row["stage"] in TERMINAL_STAGES:
                return False
            self._conn.execute(
                "UPDATE jobs SET stage='cancelled', error=?, updated_ts=?, heartbeat_ts=? "
                "WHERE id=?",
                (reason, now, now, job_id),
            )
        return True

    def update_transpose(
        self, job_id: str, transpose_semitones: int | None, capo_fret: int | None
    ) -> None:
        sets: list[str] = []
        values: list = []
        if transpose_semitones is not None:
            sets.append("transpose_semitones=?")
            values.append(int(transpose_semitones))
        if capo_fret is not None:
            sets.append("capo_fret=?")
            values.append(int(capo_fret))
        if not sets:
            return
        sets.append("updated_ts=?")
        values.append(time.time())
        values.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", tuple(values)
            )

    def reap_stale(self, older_than_s: float = 60.0) -> int:
        """Mark as 'error/interrupted' any job whose worker died mid-stage."""
        cutoff = time.time() - older_than_s
        placeholders = ",".join("?" for _ in ACTIVE_STAGES)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET stage='error', error='interrupted', updated_ts=? "
                f"WHERE stage IN ({placeholders}) AND heartbeat_ts < ?",
                (time.time(), *ACTIVE_STAGES, cutoff),
            )
        return cur.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def stage_progress_fraction(stages: Iterable[Stage], current: Stage) -> float:
    ordered = list(stages)
    if current not in ordered:
        return 0.0
    idx = ordered.index(current)
    return (idx + 1) / len(ordered)
