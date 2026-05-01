from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.demos.overfitting import router as overfitting_router
from backend.demos.regression import router as regression_router

app = FastAPI()
app.include_router(regression_router, prefix="/api/regression")
app.include_router(overfitting_router, prefix="/api/overfitting")

_frontend = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_frontend, html=True), name="static")
