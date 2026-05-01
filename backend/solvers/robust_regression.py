import numpy as np
from sklearn.linear_model import QuantileRegressor


def solve(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Minimize sum of absolute residuals (L1). Returns (slope, intercept)."""
    model = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
    model.fit(xs.reshape(-1, 1), ys)
    return float(model.coef_[0]), float(model.intercept_)
