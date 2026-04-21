"""HTTP routes for the stem-remover feature.

Mounts at ``/stems``. Four endpoints:
    GET  /stems                  → HTML form
    POST /stems                  → multipart upload, validate, enqueue, 303 redirect
    GET  /stems/{job_id}         → HTML status page (or JSON if Accept: application/json)
    GET  /stems/{job_id}/download → FileResponse of the encoded MP3
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.templating import Jinja2Templates

from app.api.stems_schemas import ALLOWED_EXTS, STEMS_ALL, StemsJobOut
from src.separation_stems import hash_file, probe_duration_sec
from src.stems_jobs import StemsJobRepo

router = APIRouter(prefix="/stems")


def _repo(request: Request) -> StemsJobRepo:
    return request.app.state.stems_repo


def _templates(request: Request) -> Jinja2Templates:
    return Jinja2Templates(
        directory=str(request.app.state.settings.frontend_dir)
    )


def _max_bytes(request: Request) -> int:
    return request.app.state.settings.stems.max_bytes


def _max_dur(request: Request) -> int:
    return request.app.state.settings.stems.max_duration_sec


@router.get("", response_class=HTMLResponse)
async def stems_form(request: Request):
    return _templates(request).TemplateResponse(
        "stems.html",
        {"request": request, "job": None, "stems_all": STEMS_ALL},
    )


@router.post("")
async def stems_submit(
    request: Request,
    file: UploadFile = File(...),
    remove_drums: str | None = Form(None),
    remove_bass: str | None = Form(None),
    remove_vocals: str | None = Form(None),
    remove_other: str | None = Form(None),
):
    mask = tuple(
        sorted(
            name
            for name, val in (
                ("drums", remove_drums),
                ("bass", remove_bass),
                ("vocals", remove_vocals),
                ("other", remove_other),
            )
            if val
        )
    )
    if not mask:
        raise HTTPException(
            status_code=422, detail="Pick at least one stem to remove."
        )
    if len(mask) == 4:
        raise HTTPException(
            status_code=422,
            detail="You've asked for silence. Uncheck at least one.",
        )

    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if not safe_name or ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail="Unsupported audio file.")

    settings = request.app.state.settings
    max_bytes = _max_bytes(request)
    pending_dir = Path(settings.input_dir) / "stems" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Stream upload to a temp file inside the pending directory so rename
    # stays on the same filesystem (avoid cross-device EXDEV).
    fd, tmp_name = tempfile.mkstemp(dir=str(pending_dir), suffix=ext)
    import os

    try:
        total = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large — max "
                            f"{max_bytes // (1024 * 1024)} MB."
                        ),
                    )
                out.write(chunk)

        tmp_path = Path(tmp_name)

        dur = probe_duration_sec(tmp_path)
        if dur is None:
            raise HTTPException(
                status_code=415, detail="Unsupported audio file."
            )
        max_dur = _max_dur(request)
        if dur > max_dur:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too long — max "
                    f"{max_dur // 60} min for MVP."
                ),
            )

        sha = hash_file(tmp_path)
        bitrate = settings.stems.mp3_bitrate
        job = _repo(request).create(
            filename=safe_name,
            remove_mask=mask,
            input_sha256=sha,
            bitrate=bitrate,
        )

        # Move file into the job's final dir.
        final_dir = Path(settings.input_dir) / "stems" / job.id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / safe_name
        tmp_path.replace(final_path)

        _repo(request).advance(job.id, "queued", progress=0.1)
        return RedirectResponse(url=f"/stems/{job.id}", status_code=303)
    except HTTPException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        Path(tmp_name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


@router.get("/{job_id}", response_class=HTMLResponse)
async def stems_status(request: Request, job_id: str):
    job = _repo(request).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return JSONResponse(StemsJobOut.from_row(job).model_dump())

    return _templates(request).TemplateResponse(
        "stems.html",
        {"request": request, "job": job, "stems_all": STEMS_ALL},
    )


@router.get("/{job_id}/download")
async def stems_download(request: Request, job_id: str):
    job = _repo(request).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.stage != "ready" or not job.output_path:
        raise HTTPException(
            status_code=409,
            detail=f"job not ready (stage={job.stage})",
        )
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(
            status_code=410, detail="output expired — please resubmit"
        )
    return FileResponse(
        str(path), media_type="audio/mpeg", filename=path.name
    )
