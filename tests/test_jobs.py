from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.jobs import JobRepo


def test_create_and_get(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    job = repo.create("song.mp3")
    assert job.stage == "uploaded"
    assert job.progress == 0.0
    fetched = repo.get(job.id)
    assert fetched is not None
    assert fetched.filename == "song.mp3"


def test_advance_progress_and_stage(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    j = repo.create("a.mp3")
    repo.advance(j.id, "queued")
    repo.advance(j.id, "separating", progress=0.1)
    got = repo.get(j.id)
    assert got.stage == "separating"
    assert got.progress == pytest.approx(0.1)


def test_next_queued_picks_oldest(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    a = repo.create("a.mp3")
    b = repo.create("b.mp3")
    repo.advance(a.id, "queued")
    time.sleep(0.01)
    repo.advance(b.id, "queued")
    picked = repo.next_queued()
    assert picked is not None
    assert picked.id == a.id


def test_reap_stale_marks_interrupted(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    j = repo.create("a.mp3")
    repo.advance(j.id, "transcribing", progress=0.4)
    # Manually roll back heartbeat beyond cutoff
    repo._conn.execute(
        "UPDATE jobs SET heartbeat_ts=? WHERE id=?", (time.time() - 120.0, j.id)
    )
    reaped = repo.reap_stale(older_than_s=30.0)
    assert reaped == 1
    got = repo.get(j.id)
    assert got.stage == "error"
    assert got.error == "interrupted"


def test_cancel_queued_job(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    j = repo.create("a.mp3")
    repo.advance(j.id, "queued")
    assert repo.cancel(j.id) is True
    got = repo.get(j.id)
    assert got.stage == "cancelled"
    assert got.error == "cancelled by user"
    # Cancelling an already-cancelled job returns False (idempotent guard).
    assert repo.cancel(j.id) is False


def test_cancel_terminal_job_refused(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    j = repo.create("a.mp3")
    repo.advance(j.id, "ready_for_review", progress=1.0)
    assert repo.cancel(j.id) is False
    got = repo.get(j.id)
    assert got.stage == "ready_for_review"


def test_update_transpose_and_capo(tmp_path: Path) -> None:
    repo = JobRepo(tmp_path / "jobs.sqlite")
    j = repo.create("a.mp3")
    repo.update_transpose(j.id, transpose_semitones=3, capo_fret=2)
    got = repo.get(j.id)
    assert got.transpose_semitones == 3
    assert got.capo_fret == 2
    repo.update_transpose(j.id, transpose_semitones=None, capo_fret=5)
    got = repo.get(j.id)
    assert got.transpose_semitones == 3
    assert got.capo_fret == 5
