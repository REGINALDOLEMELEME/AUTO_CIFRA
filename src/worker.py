from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from .config import Settings
from .jobs import JobRepo

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger("auto_cifra.worker")


def _run_job_sync(job_id: str, repo: JobRepo, settings: Settings) -> None:
    from .pipeline import run as run_pipeline, PipelineCancelled

    job = repo.get(job_id)
    if not job:
        return
    try:
        run_pipeline(job=job, repo=repo, settings=settings)
    except PipelineCancelled:
        logger.info("pipeline cancelled for job %s", job_id)
        # stage is already 'cancelled' in the DB (set by the cancel endpoint)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline crashed for job %s", job_id)
        repo.advance(job_id, "error", error=str(exc))


async def worker_loop(repo: JobRepo, settings: Settings) -> None:
    loop = asyncio.get_running_loop()
    logger.info("worker loop started")
    while True:
        try:
            job = repo.next_queued()
            if not job:
                await asyncio.sleep(settings.worker.poll_interval_seconds)
                continue
            logger.info("picked up job %s (%s)", job.id, job.filename)

            def _heartbeat_ticker(stop_evt: threading.Event, jid: str) -> None:
                while not stop_evt.wait(5.0):
                    repo.heartbeat(jid)

            stop = threading.Event()
            ht = threading.Thread(
                target=_heartbeat_ticker, args=(stop, job.id), daemon=True
            )
            ht.start()
            try:
                await loop.run_in_executor(
                    None, _run_job_sync, job.id, repo, settings
                )
            finally:
                stop.set()
                ht.join(timeout=2.0)
        except asyncio.CancelledError:
            logger.info("worker loop cancelled")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("worker loop error")
            await asyncio.sleep(settings.worker.poll_interval_seconds)
