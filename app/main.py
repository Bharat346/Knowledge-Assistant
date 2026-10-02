from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import router
from app.config.settings import Settings
from app.services.rag_service import RAGService, ServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def create_app(settings: Settings | None = None, rag_service: RAGService | None = None) -> FastAPI:
    active_settings = (settings or Settings.from_env()).validate_chunking()
    application = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")
    application.state.settings = active_settings
    application.state.rag_service = rag_service or RAGService(active_settings)
    application.include_router(router)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled request failure id=%s", getattr(request.state, "request_id", "unknown"))
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred"})

    @application.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @application.get("/style.css", include_in_schema=False)
    async def frontend_css() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @application.get("/app.js", include_in_schema=False)
    async def frontend_js() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

    return application


app = create_app()