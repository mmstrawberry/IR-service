from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from core.registry import autodiscover_algorithms


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("workdirs/tmp").mkdir(parents=True, exist_ok=True)
    Path("workdirs/inputs").mkdir(parents=True, exist_ok=True)
    Path("workdirs/outputs").mkdir(parents=True, exist_ok=True)
    autodiscover_algorithms("models")
    yield


app = FastAPI(
    title="Image Enhancement Integration Service",
    version="0.1.0",
    lifespan=lifespan,
)

# 将 API 路由注册到 FastAPI 实例上，并设置静态文件目录
app.include_router(api_router, prefix="/api", tags=["algorithms"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse("static/index.html")