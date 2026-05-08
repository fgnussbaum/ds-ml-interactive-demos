from __future__ import annotations

import numpy as np
import seaborn as sns
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# ── Data (loaded once at import time) ─────────────────────────────────────────
_df = sns.load_dataset("tips")
_df["tip_rate"] = _df["tip"] / _df["total_bill"]

_x_raw  = _df["total_bill"].values.astype(float)
_x_mean = float(np.mean(_x_raw))
_x_std  = float(np.std(_x_raw))          # ddof=0 → sum(X²)=n → Hessian=2 exactly
_X      = (_x_raw - _x_mean) / _x_std    # shape (n,), mean=0, var=1
_y      = _df["tip_rate"].values.astype(float)

# OLS: b = ȳ (exact since X is centred), θ = <X,y> / <X,X>
_b_ols     = float(np.mean(_y))
_theta_ols = float(np.dot(_X, _y) / np.dot(_X, _X))

# ── Constants ─────────────────────────────────────────────────────────────────
_THETA_START = _theta_ols - 2.5          # fixed "wrong" starting slope for 1-D GD
_B_START_2D  = _b_ols + 2.5             # match slope deviation so 2-D path is genuinely 2-D
_N_STEPS     = 50
# With Hessian=2, convergence needs α < 1.
# small: monotone slow  |  good: fast smooth  |  large: oscillates (|1-α·H|=0.94<1)
_LR_PRESETS: dict[str, float] = {"small": 0.05, "good": 0.30, "large": 1.00}
_THETA_GRID  = 200
_CONTOUR_RES = 60


def _mse(theta: float, b: float) -> float:
    return float(np.mean((theta * _X + b - _y) ** 2))


def _grad_theta(theta: float, b: float) -> float:
    return float(2.0 * np.mean((theta * _X + b - _y) * _X))


def _grad_b(theta: float, b: float) -> float:
    return float(2.0 * np.mean(theta * _X + b - _y))


# ── Pydantic models ────────────────────────────────────────────────────────────
class InitResponse(BaseModel):
    x_data: list[float]
    y_data: list[float]
    theta_ols: float
    intercept_ols: float
    theta_start: float
    b_start_2d: float
    theta_range: list[float]
    loss_curve: list[float]
    b_range: list[float]
    slope_range: list[float]
    contour_z: list[list[float]]


class RunRequest(BaseModel):
    lr_preset: str = "good"
    theta_start: float | None = None  # defaults to _THETA_START when omitted


class StepData(BaseModel):
    theta: float
    loss_1d: float
    gradient: float
    intercept_2d: float
    slope_2d: float
    loss_2d: float


class RunResponse(BaseModel):
    trajectory: list[StepData]
    lr_used: float


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/init", response_model=InitResponse)
def init() -> InitResponse:
    theta_range = np.linspace(_theta_ols - 3.2, _theta_ols + 3.2, _THETA_GRID)
    loss_curve  = [_mse(float(t), _b_ols) for t in theta_range]

    b_grid     = np.linspace(_b_ols - 3.2, _b_ols + 3.2, _CONTOUR_RES)
    slope_grid = np.linspace(_theta_ols - 2.8, _theta_ols + 2.8, _CONTOUR_RES)
    B, S = np.meshgrid(b_grid, slope_grid)                          # (CR, CR)
    preds = S[:, :, np.newaxis] * _X + B[:, :, np.newaxis] - _y    # (CR, CR, n)
    contour_z = np.mean(preds ** 2, axis=2)                         # (CR, CR)

    return InitResponse(
        x_data=_X.tolist(),
        y_data=_y.tolist(),
        theta_ols=_theta_ols,
        intercept_ols=_b_ols,
        theta_start=_THETA_START,
        b_start_2d=_B_START_2D,
        theta_range=theta_range.tolist(),
        loss_curve=loss_curve,
        b_range=b_grid.tolist(),
        slope_range=slope_grid.tolist(),
        contour_z=contour_z.tolist(),
    )


@router.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    lr = _LR_PRESETS.get(req.lr_preset, 0.30)
    start = req.theta_start if req.theta_start is not None else _THETA_START

    theta_1d = start
    theta_2d = start
    b_2d     = _B_START_2D

    trajectory: list[StepData] = []
    for _ in range(_N_STEPS):
        trajectory.append(StepData(
            theta=theta_1d,
            loss_1d=_mse(theta_1d, _b_ols),
            gradient=_grad_theta(theta_1d, _b_ols),
            intercept_2d=b_2d,
            slope_2d=theta_2d,
            loss_2d=_mse(theta_2d, b_2d),
        ))
        theta_1d -= lr * _grad_theta(theta_1d, _b_ols)
        g_t   = _grad_theta(theta_2d, b_2d)
        g_b   = _grad_b(theta_2d, b_2d)
        theta_2d -= lr * g_t
        b_2d     -= lr * g_b

    # append position after the final step
    trajectory.append(StepData(
        theta=theta_1d,
        loss_1d=_mse(theta_1d, _b_ols),
        gradient=_grad_theta(theta_1d, _b_ols),
        intercept_2d=b_2d,
        slope_2d=theta_2d,
        loss_2d=_mse(theta_2d, b_2d),
    ))

    return RunResponse(trajectory=trajectory, lr_used=lr)
