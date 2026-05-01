import warnings

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_SCENARIOS: dict[str, dict] = {
    "cubic": {
        "seed": 42,
        "x_range": (-2.5, 2.5),
        "true_fn": lambda x: 0.5 * x**3 - x + 0.3,
        "noise_std": 0.5,
    },
    "sine": {
        "seed": 7,
        "x_range": (-3.0, 3.0),
        "true_fn": lambda x: np.sin(np.pi * x / 2),
        "noise_std": 0.25,
    },
    "linear": {
        "seed": 13,
        "x_range": (-2.5, 2.5),
        "true_fn": lambda x: 1.2 * x + 0.4,
        "noise_std": 0.5,
    },
}

_N_TEST = 50
_N_CURVE = 200
_MAX_DEGREE = 9


class GenerateRequest(BaseModel):
    scenario: str = "cubic"
    n_train: int = 25


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
    """Fit polynomials of degree 1–9 and return all results upfront."""
    cfg = _SCENARIOS[req.scenario]
    x_min, x_max = cfg["x_range"]
    true_fn = cfg["true_fn"]

    rng_train = np.random.default_rng(cfg["seed"])
    train_x = np.sort(rng_train.uniform(x_min, x_max, req.n_train))
    train_y = true_fn(train_x) + rng_train.normal(0, cfg["noise_std"], req.n_train)

    # Test set uses offset seed so it stays fixed regardless of n_train
    rng_test = np.random.default_rng(cfg["seed"] + 1000)
    test_x = np.sort(rng_test.uniform(x_min, x_max, _N_TEST))
    test_y = true_fn(test_x) + rng_test.normal(0, cfg["noise_std"], _N_TEST)

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
