"""Asyncio worker loop for the stem-remover queue (ADR-SR-002).

Mirrors the pattern in ``src/worker.py`` but is bound to ``StemsJobRepo`` and
calls ``separation_stems.process_job``. Runs as a sibling asyncio task of the
existing cifra worker — no shared state.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from .config import Settings
from .separation_stems import StemRemoverError, process_job
from .stems_jobs import StemsJobRepo

logger = logging.getLogger("auto_cifra.stems_worker")


def _run_job_sync(
    job_id: str, repo: StemsJobRepo, settings: Settings
) -> None:
    job = repo.get(job_id)
    if not job:
        return
    try:
        process_job(job, repo, settings)
    except StemRemoverError as exc:
        logger.warning("stems job %s failed: %s", job_id, exc)
        repo.advance(job_id, "error", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("stems job %s crashed", job_id)
        repo.advance(job_id, "error", error=str(exc))


async def stems_worker_loop(
    repo: StemsJobRepo, settings: Settings
) -> None:
    loop = asyncio.get_running_loop()
    logger.info("stems worker loop started")
    while True:
        try:
            job = repo.next_queued()
            if not job:
                await asyncio.sleep(settings.worker.poll_interval_seconds)
                continue
            logger.info(
                "stems worker picked up job %s (%s) remove=%s",
                job.id, job.filename, job.remove_mask,
            )

            stop_evt = threading.Event()

            def _heartbeat(evt: threading.Event = stop_evt, jid: str = job.id) -> None:
                while not evt.wait(5.0):
                    repo.heartbeat(jid)

            ht = threading.Thread(target=_heartbeat, daemon=True)
            ht.start()
            try:
                await loop.run_in_executor(
                    None, _run_job_sync, job.id, repo, settings
                )
            finally:
                stop_evt.set()
                ht.join(timeout=2.0)
        except asyncio.CancelledError:
            logger.info("stems worker loop cancelled")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("stems worker loop error")
            await asyncio.sleep(settings.worker.poll_interval_seconds)
