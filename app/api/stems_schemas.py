from __future__ import annotations

from pydantic import BaseModel

from src.stems_jobs import StemsJob, StemsStage

STEMS_ALL: tuple[str, ...] = (
    "drums", "bass", "vocals", "other", "guitar", "piano",
)
ALLOWED_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


class StemsJobOut(BaseModel):
    id: str
    filename: str
    stage: StemsStage
    progress: float
    remove_mask: list[str]
    bitrate: int
    output_path: str | None = None
    error: str | None = None
    created_ts: float
    updated_ts: float

    @classmethod
    def from_row(cls, job: StemsJob) -> "StemsJobOut":
        return cls(
            id=job.id,
            filename=job.filename,
            stage=job.stage,
            progress=job.progress,
            remove_mask=list(job.remove_mask),
            bitrate=job.bitrate,
            output_path=job.output_path,
            error=job.error,
            created_ts=job.created_ts,
            updated_ts=job.updated_ts,
        )
