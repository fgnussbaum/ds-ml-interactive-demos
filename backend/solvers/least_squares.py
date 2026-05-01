import numpy as np


def solve(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Minimize sum of squared residuals (L2). Returns (slope, intercept)."""
    A = np.column_stack([xs, np.ones(len(xs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    return float(coeffs[0]), float(coeffs[1])
