from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from src.config import get_settings
from src.jobs import JobRepo


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repo = JobRepo(settings.db_path)
    reaped = repo.reap_stale(older_than_s=60.0)
    app.state.settings = settings
    app.state.repo = repo
    app.state.reaped_on_start = reaped

    from src.worker import worker_loop  # deferred: avoids loading torch until the worker really needs it

    task = asyncio.create_task(worker_loop(repo=repo, settings=settings))
    app.state.worker_task = task
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        repo.close()


def create_app() -> FastAPI:
    # Always read settings fresh at factory time so tests that chdir to a
    # temp project directory see the right paths (cache is cleared by fixtures).
    settings = get_settings()
    app = FastAPI(title="AUTO_CIFRA", version="1.0.0", lifespan=lifespan)
    static_path = settings.frontend_dir / "static"
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    app.include_router(router)
    return app
