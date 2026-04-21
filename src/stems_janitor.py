"""TTL-based cleanup for the stem-remover feature (ADR-SR-007).

Removes ``data/input/stems/{id}``, ``data/tmp/stems/{id}``,
``data/output/stems/{id}`` for jobs older than ``stems.cache_ttl_hours``,
and deletes orphan ``data/stems_cache/{sha}`` directories whose contents
were last touched before the cutoff.

Idempotent and defensive: missing directories are not an error.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from .config import Settings
from .stems_jobs import StemsJobRepo

logger = logging.getLogger("auto_cifra.stems_janitor")


def _rm_dir(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def cleanup(repo: StemsJobRepo, settings: Settings) -> dict:
    """Run one pass. Returns a small report dict for logging / tests."""
    stems_cfg = settings.stems
    cutoff = time.time() - stems_cfg.cache_ttl_hours * 3600.0

    root = settings.project_root
    input_root = settings.input_dir / "stems"
    tmp_root = settings.tmp_dir / "stems"
    output_root = root / "data" / "output" / "stems"
    cache_root = root / "data" / "stems_cache"

    # 1. Delete job directories for jobs older than TTL.
    old_jobs = repo.list_older_than(cutoff)
    job_dirs_removed = 0
    for job in old_jobs:
        for base in (input_root, tmp_root, output_root):
            if _rm_dir(base / job.id):
                job_dirs_removed += 1
        repo.delete(job.id)

    # 2. Orphan cache cleanup — keep cache dirs whose mtime is within the TTL
    #    window, even if no job references them right now (they may be reused
    #    by an upcoming resubmission).
    cache_dirs_removed = 0
    if cache_root.exists():
        for entry in cache_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                if _rm_dir(entry):
                    cache_dirs_removed += 1

    report = {
        "old_jobs": len(old_jobs),
        "job_dirs_removed": job_dirs_removed,
        "cache_dirs_removed": cache_dirs_removed,
    }
    if any(report.values()):
        logger.info("stems janitor: %s", report)
    return report
