from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.docx_export import export_aligned_chord_docx
from src.jobs import JobRepo
from app.api.schemas import (
    AlignedDoc,
    ExportOptions,
    HistoryItem,
    HistoryResponse,
    JobOut,
    ProcessResponse,
    SaveBody,
    UploadResponse,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _templates(request: Request) -> Jinja2Templates:
    return Jinja2Templates(directory=str(request.app.state.settings.frontend_dir))


def _repo(request: Request) -> JobRepo:
    return request.app.state.repo


def _aligned_path(request: Request, job_id: str) -> Path:
    return request.app.state.settings.tmp_dir / job_id / "aligned.json"


def _read_aligned(request: Request, job_id: str) -> dict:
    p = _aligned_path(request, job_id)
    if not p.exists():
        return {"source_file": "", "lines": [], "warnings": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_aligned(request: Request, job_id: str, aligned: dict) -> None:
    p = _aligned_path(request, job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "reaped_on_start": getattr(request.app.state, "reaped_on_start", 0)}


@router.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if not safe_name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid or missing filename/extension")

    settings = request.app.state.settings
    dest = Path(settings.input_dir) / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    job = _repo(request).create(safe_name)
    return UploadResponse(job=JobOut.from_row(job))


@router.post("/process/{job_id}", response_model=ProcessResponse)
async def process(job_id: str, request: Request) -> ProcessResponse:
    repo = _repo(request)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.stage not in {"uploaded", "error", "ready_for_review", "exported"}:
        return ProcessResponse(ok=False, job=JobOut.from_row(job))
    repo.advance(job_id, "queued", progress=0.0, error=None)
    job = repo.get(job_id)
    return ProcessResponse(ok=True, job=JobOut.from_row(job))  # type: ignore[arg-type]


@router.get("/jobs/{job_id}", response_model=JobOut)
async def job_status(job_id: str, request: Request) -> JobOut:
    job = _repo(request).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut.from_row(job)


@router.get("/history", response_model=HistoryResponse)
async def history(request: Request) -> HistoryResponse:
    settings = request.app.state.settings
    jobs = _repo(request).list_recent(50)
    items: list[HistoryItem] = []
    for j in jobs:
        docx = settings.output_dir / j.id / f"{Path(j.filename).stem}.docx"
        items.append(
            HistoryItem(
                job_id=j.id,
                filename=j.filename,
                stage=j.stage,
                updated_ts=j.updated_ts,
                output_docx=str(docx) if docx.exists() else None,
            )
        )
    return HistoryResponse(items=items)


@router.get("/review/{job_id}", response_class=HTMLResponse)
async def review(job_id: str, request: Request):
    job = _repo(request).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    aligned = _read_aligned(request, job_id)
    templates = _templates(request)
    return templates.TemplateResponse(
        "review.html",
        {"request": request, "job": job, "aligned": aligned, "job_id": job_id},
    )


@router.post("/save/{job_id}")
async def save(job_id: str, body: SaveBody, request: Request) -> JSONResponse:
    repo = _repo(request)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if body.aligned is not None:
        _write_aligned(request, job_id, body.aligned.model_dump())
    if body.transpose_semitones is not None or body.capo_fret is not None:
        repo.update_transpose(job_id, body.transpose_semitones, body.capo_fret)
    return JSONResponse({"ok": True})


@router.post("/export/{job_id}")
async def export(job_id: str, request: Request, opts: ExportOptions | None = None):
    repo = _repo(request)
    settings = request.app.state.settings
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    aligned = _read_aligned(request, job_id)
    if not aligned.get("lines"):
        raise HTTPException(status_code=409, detail="no aligned content to export")

    opts = opts or ExportOptions()
    transpose = opts.transpose_semitones if opts.transpose_semitones is not None else job.transpose_semitones
    capo = opts.capo_fret if opts.capo_fret is not None else job.capo_fret

    out_dir = settings.output_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(job.filename).stem}.docx"
    export_aligned_chord_docx(
        arrangement=aligned,
        output_path=out_path,
        title=Path(job.filename).stem,
        transpose_semitones=transpose,
        capo_fret=capo,
        prefer_flats=opts.prefer_flats,
        body_font=settings.docx.body_font,
        chord_font=settings.docx.chord_font,
        body_size_pt=settings.docx.body_size_pt,
        chord_size_pt=settings.docx.chord_size_pt,
    )
    repo.advance(job_id, "exported", progress=1.0)
    return FileResponse(
        str(out_path),
        filename=out_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    templates = _templates(request)
    return templates.TemplateResponse("index.html", {"request": request})
