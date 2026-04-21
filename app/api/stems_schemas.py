from __future__ import annotations

from pydantic import BaseModel

from src.config import QUALITY_PRESETS
from src.stems_jobs import StemsJob, StemsStage

STEMS_ALL: tuple[str, ...] = (
    "drums", "bass", "vocals", "other", "guitar", "piano",
)
ALLOWED_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
OUTPUT_FORMATS: tuple[str, ...] = ("flac", "wav", "mp3")
__all__ = [
    "STEMS_ALL", "ALLOWED_EXTS", "QUALITY_PRESETS", "OUTPUT_FORMATS",
    "StemsJobOut",
]


class StemsJobOut(BaseModel):
    id: str
    filename: str
    stage: StemsStage
    progress: float
    remove_mask: list[str]
    bitrate: int
    quality: str
    output_format: str
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
            quality=getattr(job, "quality", "best"),
            output_format=getattr(job, "output_format", "mp3"),
            output_path=job.output_path,
            error=job.error,
            created_ts=job.created_ts,
            updated_ts=job.updated_ts,
        )
