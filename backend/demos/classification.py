import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# ── Toy dataset generation ────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_unseparated(rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """Two overlapping Gaussians; classes are hard to separate."""
    n = 100
    y = (rng.random(n) < 0.5).astype(int)
    logits = np.where(y == 1,
                      rng.normal(0.0, 1.2, n),
                      rng.normal(0.0, 1.2, n))
    probs = _sigmoid(logits)
    # small nudge so class-1 skews slightly right
    probs = np.clip(probs + y * 0.05 - 0.025, 0.02, 0.98)
    return probs.tolist(), y.tolist()


def _make_separated(rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """Well-separated classes with minimal overlap."""
    n = 100
    y = (rng.random(n) < 0.5).astype(int)
    logits = np.where(y == 1,
                      rng.normal(1.5, 0.7, n),
                      rng.normal(-1.5, 0.7, n))
    probs = _sigmoid(logits)
    return probs.tolist(), y.tolist()


def _make_imbalanced(rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """80 % negative, 20 % positive; moderately separated."""
    n = 100
    y = (rng.random(n) < 0.2).astype(int)
    logits = np.where(y == 1,
                      rng.normal(1.0, 0.9, n),
                      rng.normal(-1.0, 0.9, n))
    probs = _sigmoid(logits)
    return probs.tolist(), y.tolist()


_DATASETS: dict[str, tuple[list[float], list[int]]] = {
    "unseparated": _make_unseparated(np.random.default_rng(42)),
    "separated":   _make_separated(np.random.default_rng(7)),
    "imbalanced":  _make_imbalanced(np.random.default_rng(13)),
}

# ── Pydantic models ───────────────────────────────────────────────────────────

class DatasetResponse(BaseModel):
    probs:  list[float]
    labels: list[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/dataset/{name}", response_model=DatasetResponse)
def get_dataset(name: str) -> DatasetResponse:
    if name not in _DATASETS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")
    probs, labels = _DATASETS[name]
    return DatasetResponse(probs=probs, labels=labels)
