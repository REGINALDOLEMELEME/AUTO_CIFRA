from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.stems_routes import router as stems_router
from src.config import get_settings
from src.jobs import JobRepo
from src.stems_janitor import cleanup as stems_cleanup
from src.stems_jobs import StemsJobRepo

logger = logging.getLogger("auto_cifra.app")


async def _stems_janitor_tick(
    repo: StemsJobRepo, settings, interval_sec: int
) -> None:
    while True:
        try:
            await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            raise
        try:
            stems_cleanup(repo, settings)
        except Exception:  # noqa: BLE001
            logger.exception("stems janitor tick failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repo = JobRepo(settings.db_path)
    reaped = repo.reap_stale(older_than_s=60.0)
    app.state.settings = settings
    app.state.repo = repo
    app.state.reaped_on_start = reaped

    # --- stems feature wiring ---------------------------------------------
    stems_db_path = settings.project_root / "data" / "stems_jobs.sqlite"
    stems_repo = StemsJobRepo(stems_db_path)
    app.state.stems_repo = stems_repo
    app.state.reaped_stems_on_start = stems_repo.reap_stale(older_than_s=30 * 60.0)
    try:
        stems_cleanup(stems_repo, settings)
    except Exception:  # noqa: BLE001
        logger.exception("initial stems janitor pass failed")

    from src.worker import worker_loop  # deferred: avoids loading torch until the worker really needs it
    from src.stems_worker import stems_worker_loop  # deferred for the same reason

    worker_task = asyncio.create_task(worker_loop(repo=repo, settings=settings))
    stems_task = asyncio.create_task(
        stems_worker_loop(repo=stems_repo, settings=settings)
    )
    janitor_task = asyncio.create_task(
        _stems_janitor_tick(
            stems_repo, settings, settings.stems.janitor_interval_sec
        )
    )
    app.state.worker_task = worker_task
    app.state.stems_worker_task = stems_task
    app.state.stems_janitor_task = janitor_task
    try:
        yield
    finally:
        for t in (worker_task, stems_task, janitor_task):
            t.cancel()
        for t in (worker_task, stems_task, janitor_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        repo.close()
        stems_repo.close()


def create_app() -> FastAPI:
    # Always read settings fresh at factory time so tests that chdir to a
    # temp project directory see the right paths (cache is cleared by fixtures).
    settings = get_settings()
    app = FastAPI(title="AUTO_CIFRA", version="1.0.0", lifespan=lifespan)
    static_path = settings.frontend_dir / "static"
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    app.include_router(router)
    app.include_router(stems_router)
    return app
