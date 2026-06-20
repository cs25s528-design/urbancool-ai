"""FastAPI entrypoint for UrbanCool AI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routers import heat, intervention, report, scenario, weather

PROJECT_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(title="UrbanCool AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(heat.router)
app.include_router(scenario.router)
app.include_router(intervention.router)
app.include_router(report.router)
app.include_router(weather.router)

static_dir = PROJECT_DIR / "data" / "outputs"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

frontend_dir = PROJECT_DIR / "app" / "frontend"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="frontend-assets")


@app.get("/")
def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "UrbanCool AI", "status": "ok"}


@app.get("/health")
def healthcheck():
    return {"status": "ok"}
