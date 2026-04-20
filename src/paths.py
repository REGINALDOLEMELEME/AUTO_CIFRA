from __future__ import annotations

from pathlib import Path


def ensure_directories(paths: dict) -> None:
    for key in ("input_dir", "output_dir", "tmp_dir", "models_dir"):
        if key in paths and paths[key]:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
    db_path = paths.get("db_path")
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def job_tmp_dir(tmp_dir: str | Path, job_id: str) -> Path:
    p = Path(tmp_dir) / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_output_dir(output_dir: str | Path, job_id: str) -> Path:
    p = Path(output_dir) / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p
