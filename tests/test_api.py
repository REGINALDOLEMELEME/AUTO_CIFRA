from __future__ import annotations

import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_project: Path):
    # Lazy import to ensure fresh settings after tmp_project chdir.
    from fastapi.testclient import TestClient

    from app.api import create_app

    app = create_app()
    # Wipe lifespan's background task to keep the test synchronous — we only
    # exercise HTTP contracts here, not the real pipeline.
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_upload_invalid_extension(client):
    res = client.post(
        "/upload",
        files={"file": ("evil.exe", io.BytesIO(b"x"), "application/octet-stream")},
    )
    assert res.status_code == 400


def test_upload_then_status(client, tmp_project: Path):
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res = client.post(
        "/upload",
        files={"file": ("a.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert res.status_code == 200
    job = res.json()["job"]
    assert job["stage"] == "uploaded"
    assert job["filename"] == "a.wav"

    # Job status endpoint
    status = client.get(f"/jobs/{job['id']}")
    assert status.status_code == 200
    assert status.json()["id"] == job["id"]


def test_process_requires_existing_job(client):
    res = client.post("/process/does-not-exist")
    assert res.status_code == 404


def test_cancel_queued_job(client, tmp_project):
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    up = client.post(
        "/upload",
        files={"file": ("c.wav", io.BytesIO(wav_bytes), "audio/wav")},
    ).json()
    job_id = up["job"]["id"]
    client.post(f"/process/{job_id}")
    res = client.post(f"/cancel/{job_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["job"]["stage"] == "cancelled"


def test_save_then_export_round_trip(client, sample_aligned, tmp_project: Path):
    # upload → save aligned JSON (simulating review edit) → export DOCX
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    up = client.post(
        "/upload",
        files={"file": ("song.wav", io.BytesIO(wav_bytes), "audio/wav")},
    ).json()
    job_id = up["job"]["id"]

    # Save aligned payload for the job
    res = client.post(
        f"/save/{job_id}",
        json={"aligned": sample_aligned, "transpose_semitones": 0, "capo_fret": 0},
    )
    assert res.status_code == 200

    # Export DOCX
    exp = client.post(f"/export/{job_id}", json={"transpose_semitones": 2, "capo_fret": 0, "prefer_flats": True})
    assert exp.status_code == 200
    assert exp.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert int(exp.headers.get("content-length", "0")) > 1000  # DOCX is never empty


def test_history_lists_created_jobs(client):
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    client.post("/upload", files={"file": ("x.wav", io.BytesIO(wav_bytes), "audio/wav")})
    res = client.get("/history")
    assert res.status_code == 200
    assert len(res.json()["items"]) >= 1
