from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.jobs import Job, Stage


class JobOut(BaseModel):
    id: str
    filename: str
    stage: Stage
    progress: float
    error: str | None = None
    transpose_semitones: int = 0
    capo_fret: int = 0
    created_ts: float
    updated_ts: float

    @classmethod
    def from_row(cls, job: Job) -> "JobOut":
        return cls(
            id=job.id,
            filename=job.filename,
            stage=job.stage,
            progress=job.progress,
            error=job.error,
            transpose_semitones=job.transpose_semitones,
            capo_fret=job.capo_fret,
            created_ts=job.created_ts,
            updated_ts=job.updated_ts,
        )


class UploadResponse(BaseModel):
    job: JobOut


class ProcessResponse(BaseModel):
    ok: bool = True
    job: JobOut


class HistoryItem(BaseModel):
    job_id: str
    filename: str
    stage: Stage
    updated_ts: float
    output_docx: str | None = None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class AlignedWord(BaseModel):
    text: str
    start: float
    end: float
    chord: str | None = None


class AlignedLine(BaseModel):
    section: str | None = None
    start: float
    end: float
    words: list[AlignedWord] = Field(default_factory=list)
    lyric_line: str = ""
    chord_line: str = ""


class AlignedDoc(BaseModel):
    source_file: str = ""
    transcription_mode: str = "real"
    chord_mode: str = "real"
    warnings: list[str] = Field(default_factory=list)
    lines: list[AlignedLine] = Field(default_factory=list)


class SaveBody(BaseModel):
    aligned: AlignedDoc | None = None
    transpose_semitones: int | None = Field(default=None, ge=-11, le=11)
    capo_fret: int | None = Field(default=None, ge=0, le=11)


class ExportOptions(BaseModel):
    transpose_semitones: int | None = Field(default=None, ge=-11, le=11)
    capo_fret: int | None = Field(default=None, ge=0, le=11)
    prefer_flats: bool = True
