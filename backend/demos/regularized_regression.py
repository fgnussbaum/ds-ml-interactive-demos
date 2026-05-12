"""Regularized regression demo: Ridge vs. Lasso on sklearn diabetes dataset."""

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

router = APIRouter()

# ── Pydantic models ───────────────────────────────────────────────────────────


class RegModelResult(BaseModel):
    train_rmse: list[float]
    val_rmse: list[float]
    coef_paths: list[list[float]]  # outer = alpha index, inner = features
    best_alpha_idx: int
    best_val_rmse: float
    test_rmse: float


class InitResponse(BaseModel):
    feature_names: list[str]
    pearson_r: list[float]  # same order as feature_names
    alphas: list[float]     # 11 values, np.linspace(0, 0.1, 11)
    ridge: RegModelResult
    lasso: RegModelResult


class FineResponse(BaseModel):
    alphas: list[float]     # 101 values, np.linspace(0, 0.1, 101)
    ridge: RegModelResult
    lasso: RegModelResult


# ── Fixed data split (computed once at import) ────────────────────────────────

_SEED = 42
_ALPHA_MAX = 0.1
_N_COARSE = 11   # np.linspace(0, 0.1, 11)
_N_FINE = 101    # np.linspace(0, 0.1, 101)


def _prepare_data() -> tuple:
    data = load_diabetes()
    X_raw: np.ndarray = data.data
    y: np.ndarray = data.target
    feature_names: list[str] = list(data.feature_names)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_raw, y, test_size=0.40, random_state=_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.67, random_state=_SEED
    )

    x_scaler = StandardScaler().fit(X_train)
    X_tr = x_scaler.transform(X_train)
    X_va = x_scaler.transform(X_val)
    X_te = x_scaler.transform(X_test)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_tr = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_va = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    y_std = float(y_scaler.scale_[0])

    return feature_names, X_tr, X_va, X_te, y_tr, y_va, y_te, y_std


_feature_names, _X_tr, _X_va, _X_te, _y_tr, _y_va, _y_te, _y_std = _prepare_data()


# ── Compute helpers ───────────────────────────────────────────────────────────


def _pearson_r() -> list[float]:
    return [
        float(np.corrcoef(_X_tr[:, j], _y_tr)[0, 1])
        for j in range(_X_tr.shape[1])
    ]


def _sweep_ridge(
    alphas: np.ndarray,
) -> tuple[list[float], list[float], list[list[float]]]:
    n = len(_y_tr)
    train_errs, val_errs, coef_paths = [], [], []
    for a in alphas:
        m = Ridge(alpha=max(float(a), 1e-10) * n)
        m.fit(_X_tr, _y_tr)
        train_errs.append(float(root_mean_squared_error(_y_tr, m.predict(_X_tr))) * _y_std)
        val_errs.append(float(root_mean_squared_error(_y_va, m.predict(_X_va))) * _y_std)
        coef_paths.append(m.coef_.tolist())
    return train_errs, val_errs, coef_paths


def _sweep_lasso(
    alphas: np.ndarray,
) -> tuple[list[float], list[float], list[list[float]]]:
    train_errs, val_errs, coef_paths = [], [], []
    # warm_start=True reuses previous solution — sweep low→high for efficiency
    m = Lasso(alpha=1.0, warm_start=True, max_iter=10_000, tol=1e-4)
    for a in alphas:
        m.alpha = max(float(a), 1e-6)
        m.fit(_X_tr, _y_tr)
        train_errs.append(float(root_mean_squared_error(_y_tr, m.predict(_X_tr))) * _y_std)
        val_errs.append(float(root_mean_squared_error(_y_va, m.predict(_X_va))) * _y_std)
        coef_paths.append(m.coef_.copy().tolist())
    return train_errs, val_errs, coef_paths


def _test_rmse_ridge(alpha: float) -> float:
    n = len(_y_tr)
    m = Ridge(alpha=max(alpha, 1e-10) * n)
    m.fit(_X_tr, _y_tr)
    return float(root_mean_squared_error(_y_te, m.predict(_X_te))) * _y_std


def _test_rmse_lasso(alpha: float) -> float:
    m = Lasso(alpha=max(alpha, 1e-6), max_iter=10_000)
    m.fit(_X_tr, _y_tr)
    return float(root_mean_squared_error(_y_te, m.predict(_X_te))) * _y_std


def _build_model_result(
    alphas: np.ndarray,
    train_errs: list[float],
    val_errs: list[float],
    coef_paths: list[list[float]],
    test_fn,
) -> RegModelResult:
    best_idx = int(np.argmin(val_errs))
    return RegModelResult(
        train_rmse=train_errs,
        val_rmse=val_errs,
        coef_paths=coef_paths,
        best_alpha_idx=best_idx,
        best_val_rmse=val_errs[best_idx],
        test_rmse=test_fn(float(alphas[best_idx])),
    )


# ── Module-level lazy cache ───────────────────────────────────────────────────

_init_cache: InitResponse | None = None
_fine_cache: FineResponse | None = None


def _build_init() -> InitResponse:
    alphas = np.linspace(0.0, _ALPHA_MAX, _N_COARSE)

    ridge_tr, ridge_va, ridge_coef = _sweep_ridge(alphas)
    lasso_tr, lasso_va, lasso_coef = _sweep_lasso(alphas)

    return InitResponse(
        feature_names=_feature_names,
        pearson_r=_pearson_r(),
        alphas=alphas.tolist(),
        ridge=_build_model_result(alphas, ridge_tr, ridge_va, ridge_coef, _test_rmse_ridge),
        lasso=_build_model_result(alphas, lasso_tr, lasso_va, lasso_coef, _test_rmse_lasso),
    )


def _build_fine() -> FineResponse:
    alphas = np.linspace(0.0, _ALPHA_MAX, _N_FINE)

    ridge_tr, ridge_va, ridge_coef = _sweep_ridge(alphas)
    lasso_tr, lasso_va, lasso_coef = _sweep_lasso(alphas)

    return FineResponse(
        alphas=alphas.tolist(),
        ridge=_build_model_result(alphas, ridge_tr, ridge_va, ridge_coef, _test_rmse_ridge),
        lasso=_build_model_result(alphas, lasso_tr, lasso_va, lasso_coef, _test_rmse_lasso),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/init", response_model=InitResponse)
def get_init() -> InitResponse:
    """Return precomputed coarse grid, Pearson correlations, and test RMSE."""
    global _init_cache
    if _init_cache is None:
        _init_cache = _build_init()
    return _init_cache


@router.get("/fine", response_model=FineResponse)
def get_fine() -> FineResponse:
    """Return fine grid (101 α values), computed lazily and cached."""
    global _fine_cache
    if _fine_cache is None:
        _fine_cache = _build_fine()
    return _fine_cache
