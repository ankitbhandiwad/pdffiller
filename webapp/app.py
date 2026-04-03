from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.config import build_paths, build_templates
from webapp.integrations import build_integrations
from webapp.routes.api import router as api_router
from webapp.routes.pages import router as pages_router
from webapp.store import QASessionStore


def create_app() -> FastAPI:
    paths = build_paths()
    app = FastAPI()
    app.state.paths = paths
    app.state.templates = build_templates(paths)
    app.state.integrations = build_integrations()
    app.state.qa_store = QASessionStore()

    app.mount("/static", StaticFiles(directory=paths.static_dir), name="static")
    app.include_router(pages_router)
    app.include_router(api_router)
    return app


app = create_app()
