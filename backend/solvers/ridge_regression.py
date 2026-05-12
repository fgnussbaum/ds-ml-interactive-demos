import numpy as np


def solve(xs: np.ndarray, ys: np.ndarray, alpha: float = 1.0) -> tuple[float, float]:
    """Ridge regression (L2 penalty on slope only, data-normalized). Returns (slope, intercept)."""
    x_mean, x_std = xs.mean(), xs.std()
    y_mean, y_std = ys.mean(), ys.std()

    if x_std < 1e-10 or y_std < 1e-10:
        return 0.0, float(y_mean)

    xn = (xs - x_mean) / x_std
    yn = (ys - y_mean) / y_std

    A = np.column_stack([xn, np.ones(len(xn))])
    lam = np.diag([alpha, 0.0])  # don't penalize the intercept term
    # compute analytic solution (XᵀX + λI) β = Xᵀy for ||y - Xβ||² + λ||β||²
    slope_n, intercept_n = np.linalg.solve(A.T @ A + lam, A.T @ yn)

    slope = float(slope_n * y_std / x_std)
    intercept = float(y_mean - slope * x_mean + intercept_n * y_std)
    return slope, intercept
