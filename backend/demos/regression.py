import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from backend.solvers.least_squares import solve as ls_solve
from backend.solvers.robust_regression import solve as rr_solve

router = APIRouter()


class FitRequest(BaseModel):
    points: list[list[float]]
    show_ls: bool = True
    show_rr: bool = False


class LineResult(BaseModel):
    slope: float
    intercept: float
    residuals: list[float]


class FitResponse(BaseModel):
    ls: LineResult | None = None
    rr: LineResult | None = None


@router.post("/fit", response_model=FitResponse)
def fit(req: FitRequest) -> FitResponse:
    if len(req.points) < 2:
        return FitResponse()

    pts = np.array(req.points, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    response = FitResponse()

    if req.show_ls:
        slope, intercept = ls_solve(xs, ys)
        response.ls = LineResult(
            slope=slope,
            intercept=intercept,
            residuals=(ys - (slope * xs + intercept)).tolist(),
        )

    if req.show_rr:
        slope, intercept = rr_solve(xs, ys)
        response.rr = LineResult(
            slope=slope,
            intercept=intercept,
            residuals=(ys - (slope * xs + intercept)).tolist(),
        )

    return response
