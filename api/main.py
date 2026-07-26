"""지식 파이프라인 HTTP API 진입점 (docs/api_design.md).

실행:
    pip install -r api/requirements.txt
    KP_API_KEYS=my-secret uvicorn api.main:app --host 0.0.0.0 --port 8000
    # 문서: http://localhost:8000/v1/docs
"""
from contextlib import asynccontextmanager

import yaml
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, store, worker
from .auth import require_api_key
from .routers import jobs as jobs_router
from .routers import pipeline as pipeline_router
from .routers import projects as projects_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_storage()
    store.init_db()
    worker.start()
    yield
    worker.stop()


app = FastAPI(
    title="Knowledge Pipeline API",
    version="0.1.0",
    description="CPT→SFT→(tool/plan/react/planact)→export 파이프라인의 HTTP API. "
                "설계: docs/api_design.md",
    docs_url="/v1/docs", openapi_url="/v1/openapi.json",
    lifespan=lifespan,
)

_V1 = "/v1"
app.include_router(projects_router.router, prefix=_V1, tags=["projects"])
app.include_router(pipeline_router.router, prefix=_V1, tags=["pipeline"])
app.include_router(jobs_router.router, prefix=_V1, tags=["jobs"])


@app.get("/v1/healthz", tags=["meta"])
def healthz():
    return {"status": "ok"}


@app.get("/v1/readyz", tags=["meta"])
def readyz():
    return {"status": "ready", "gpu_concurrency": config.GPU_CONCURRENCY,
            "cpu_concurrency": config.CPU_CONCURRENCY}


@app.get("/v1/config/schema", tags=["meta"], dependencies=[Depends(require_api_key)])
def config_schema():
    """기본 config.yaml(override 참고용)을 반환."""
    with open(config.TEMPLATE_CONFIG, "r", encoding="utf-8") as f:
        return {"default_config": yaml.safe_load(f)}


# HTTPException(detail={"error":..})은 그대로, 그 외 문자열 detail은 감싼다.
@app.exception_handler(StarletteHTTPException)
async def http_exc(_request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={
        "error": {"code": "http_error", "message": str(detail)}})


# 미처리 예외는 500으로 감싼다.
@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": {"code": "internal", "message": str(exc)}})
