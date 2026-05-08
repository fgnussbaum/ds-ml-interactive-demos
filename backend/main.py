from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.demos.classification import router as classification_router
from backend.demos.decision_tree import router as decision_tree_router
from backend.demos.gradient_descent import router as gradient_descent_router
from backend.demos.overfitting import router as overfitting_router
from backend.demos.regression import router as regression_router

app = FastAPI()
app.include_router(regression_router, prefix="/api/regression")
app.include_router(overfitting_router, prefix="/api/overfitting")
app.include_router(classification_router, prefix="/api/classification")
app.include_router(decision_tree_router, prefix="/api/decision_tree")
app.include_router(gradient_descent_router, prefix="/api/gradient_descent")

_frontend = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_frontend, html=True), name="static")
