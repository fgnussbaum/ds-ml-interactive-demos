import warnings

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_SCENARIOS: dict[str, dict] = {
    "cubic": {
        "seed": 40,
        "x_range": (-2.5, 2.5),
        "true_fn": lambda x: 0.5 * x**3 - x + 0.3,
        "noise_std": 1.0,
    },
    "sine": {
        "seed": 7,
        "x_range": (-3.0, 3.0),
        "true_fn": lambda x: np.sin(np.pi * x / 2),
        "noise_std": 0.5,
    },
    "linear": {
        "seed": 13,
        "x_range": (-2.5, 2.5),
        "true_fn": lambda x: 1.2 * x + 0.4,
        "noise_std": 0.5,
    },
    "surprise": {
        "seed": 13,
        "x_range": (-2, 2),
        "true_fn": lambda x: np.where(np.abs(x) <= 1, x**3 - x**2 + 1, 1.0),
        "noise_std": 0.1,
        # Inner region [-1, 1] is the cubic part; outer [-2,-1)∪(1,2] is flat.
        # x_low_fraction=0.8 means 80 of 100 points are drawn from the inner region
        # (shuffled), and the remaining 20 from the outer region (shuffled).
        # Concatenating inner‖outer ensures a sorted-index split at 80% aligns
        # exactly with the distribution boundary, demonstrating covariate shift.
        "x_inner_range": (-1.0, 1.0),
        "x_low_fraction": 0.8,
    },
}

_N_POINTS = 100
_N_CURVE = 200
_MAX_DEGREE = 20


class GenerateRequest(BaseModel):
    scenario: str = "cubic"
    train_pct: float = 0.8


class FitResult(BaseModel):
    degree: int
    train_rmse: float
    test_rmse: float
    curve_y: list[float]


class GenerateResponse(BaseModel):
    train_x: list[float]
    train_y: list[float]
    test_x: list[float]
    test_y: list[float]
    curve_x: list[float]
    true_curve_y: list[float]
    fits: list[FitResult]
    best_degree: int
    x_range: list[float]


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Pre-generate N_POINTS once (shuffled); split first train_pct% as train."""
    cfg = _SCENARIOS[req.scenario]
    x_min, x_max = cfg["x_range"]
    true_fn = cfg["true_fn"]

    rng = np.random.default_rng(cfg["seed"])

    x_low_frac: float | None = cfg.get("x_low_fraction")
    if x_low_frac is not None:
        # Surprise: inner region (cubic) first, outer region (flat) second.
        # Each group is independently shuffled so x-order within a group is random.
        x_inner_min, x_inner_max = cfg["x_inner_range"]
        n_low = round(_N_POINTS * x_low_frac)
        n_high = _N_POINTS - n_low

        x_inner = rng.uniform(x_inner_min, x_inner_max, n_low)
        rng.shuffle(x_inner)

        # Draw equally from the two outer intervals to cover both sides.
        n_left = n_high // 2
        n_right = n_high - n_left
        x_outer = np.concatenate([
            rng.uniform(x_min, x_inner_min, n_left),
            rng.uniform(x_inner_max, x_max, n_right),
        ])
        rng.shuffle(x_outer)

        all_x = np.concatenate([x_inner, x_outer])
    else:
        # Base scenarios: uniform sample then shuffle once; order is fixed by seed.
        all_x = rng.uniform(x_min, x_max, _N_POINTS)
        rng.shuffle(all_x)

    all_y = true_fn(all_x) + rng.normal(0, cfg["noise_std"], _N_POINTS)

    n_train = round(_N_POINTS * req.train_pct)
    train_x, test_x = all_x[:n_train], all_x[n_train:]
    train_y, test_y = all_y[:n_train], all_y[n_train:]

    curve_x = np.linspace(x_min, x_max, _N_CURVE)
    true_curve_y = true_fn(curve_x)

    fits: list[FitResult] = []
    test_rmses: list[float] = []

    for deg in range(1, _MAX_DEGREE + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(train_x, train_y, deg)

        train_pred = np.polyval(coeffs, train_x)
        test_pred = np.polyval(coeffs, test_x)
        curve_y_vals = np.polyval(coeffs, curve_x)

        train_rmse = float(np.sqrt(np.mean((train_y - train_pred) ** 2)))
        test_rmse = float(np.sqrt(np.mean((test_y - test_pred) ** 2)))
        test_rmses.append(test_rmse)

        fits.append(FitResult(
            degree=deg,
            train_rmse=round(train_rmse, 4),
            test_rmse=round(test_rmse, 4),
            curve_y=curve_y_vals.tolist(),
        ))

    best_degree = int(np.argmin(test_rmses)) + 1

    return GenerateResponse(
        train_x=train_x.tolist(),
        train_y=train_y.tolist(),
        test_x=test_x.tolist(),
        test_y=test_y.tolist(),
        curve_x=curve_x.tolist(),
        true_curve_y=true_curve_y.tolist(),
        fits=fits,
        best_degree=best_degree,
        x_range=[x_min, x_max],
    )
