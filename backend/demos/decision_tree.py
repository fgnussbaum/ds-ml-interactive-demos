import base64
import io
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import seaborn as sns
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree as _sklearn_tree

from dtreeviz import decision_boundaries

router = APIRouter()

# ── Data (loaded once at module level) ───────────────────────────────────────

_FEATURES    = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
_CLASS_NAMES = ['Adelie', 'Chinstrap', 'Gentoo']
_COLORS      = ['#e84040', '#4682b4', '#2ca02c']   # Adelie · Chinstrap · Gentoo
_FEAT_SHORT  = {
    'bill_length_mm':    'bill length',
    'bill_depth_mm':     'bill depth',
    'flipper_length_mm': 'flipper len.',
    'body_mass_g':       'body mass',
}


def _load() -> tuple:
    df = sns.load_dataset('penguins').dropna()
    le = LabelEncoder().fit(_CLASS_NAMES)
    X  = df[_FEATURES].values
    y  = le.transform(df['species'])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, X, y


_X_train, _X_test, _y_train, _y_test, _X_all, _y_all = _load()


# ── Tree diagram: custom stacked-bar renderer ─────────────────────────────────

def _tree_src(clf: DecisionTreeClassifier) -> str:
    """Render tree with one stacked class-distribution bar per node."""
    t        = clf.tree_
    V_SPACE  = 1.5
    BAR_W    = 0.75
    BAR_H    = 0.28

    x_pos      = [0.0] * t.node_count
    node_depth = [0]   * t.node_count
    leaf_counter = 0

    def _layout(node: int, d: int) -> None:
        nonlocal leaf_counter
        node_depth[node] = d
        if t.children_left[node] == _sklearn_tree.TREE_LEAF:
            x_pos[node] = float(leaf_counter)
            leaf_counter += 1
        else:
            _layout(t.children_left[node], d + 1)
            _layout(t.children_right[node], d + 1)
            x_pos[node] = (
                x_pos[t.children_left[node]] + x_pos[t.children_right[node]]
            ) / 2

    _layout(0, 0)
    n_leaves = leaf_counter
    max_d    = max(node_depth)

    fig, ax = plt.subplots(
        figsize=(max(4.5, n_leaves * 0.95), max(2.5, (max_d + 1) * V_SPACE))
    )
    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-max_d * V_SPACE - 0.72, 0.68)
    ax.axis('off')

    for node in range(t.node_count):
        cx    = x_pos[node]
        cy    = -node_depth[node] * V_SPACE
        vals  = t.value[node][0]
        total = float(vals.sum())

        # Stacked horizontal bar
        x0 = cx - BAR_W / 2
        for count, color in zip(vals, _COLORS):
            w = (count / total) * BAR_W if total > 0 else 0
            if w > 0:
                ax.add_patch(Rectangle(
                    (x0, cy - BAR_H / 2), w, BAR_H,
                    facecolor=color, edgecolor='white', linewidth=0.3,
                ))
                x0 += w

        ax.add_patch(Rectangle(                         # outline
            (cx - BAR_W / 2, cy - BAR_H / 2), BAR_W, BAR_H,
            facecolor='none', edgecolor='#777', linewidth=0.7,
        ))
        ax.text(cx, cy - BAR_H / 2 - 0.04, f'({int(total)})',
                ha='center', va='top', fontsize=6.5, color='#555')

        if t.children_left[node] != _sklearn_tree.TREE_LEAF:
            feat = _FEATURES[t.feature[node]]
            ax.text(cx, cy + BAR_H / 2 + 0.05,
                    f'{_FEAT_SHORT[feat]}\n≤ {t.threshold[node]:.1f}',
                    ha='center', va='bottom', fontsize=6.5,
                    fontweight='bold', color='#222', linespacing=1.2)

            for child in (t.children_left[node], t.children_right[node]):
                ax.plot(
                    [cx, x_pos[child]],
                    [cy - BAR_H / 2, -node_depth[child] * V_SPACE + BAR_H / 2],
                    color='#aaa', linewidth=0.9, zorder=0,
                )

    patches = [Patch(facecolor=c, edgecolor='#666', label=n)
               for c, n in zip(_COLORS, _CLASS_NAMES)]
    ax.legend(handles=patches, loc='upper right', fontsize=7,
              framealpha=0.92, edgecolor='#ccc', handlelength=1.2)

    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Decision boundary helper ──────────────────────────────────────────────────

def _boundary_png_b64(clf: Any, feature_x: str, feature_y: str) -> str:
    """Train a 2-feature version of clf and return a base64 PNG boundary plot."""
    xi     = _FEATURES.index(feature_x)
    yi     = _FEATURES.index(feature_y)
    X_pair = _X_train[:, [xi, yi]]

    if isinstance(clf, RandomForestClassifier):
        clf2 = RandomForestClassifier(n_estimators=clf.n_estimators, random_state=42)
    else:
        clf2 = DecisionTreeClassifier(
            max_depth=clf.max_depth,
            min_samples_leaf=clf.min_samples_leaf,
            random_state=42,
        )
    clf2.fit(X_pair, _y_train)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    decision_boundaries(
        clf2, X_pair, _y_train,
        feature_names=[feature_x, feature_y],
        target_name='species',
        class_names=_CLASS_NAMES,
        ax=ax,
        colors={'classes': [None, None, None, _COLORS], 'tile_alpha': 0.35},
    )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Root caption ──────────────────────────────────────────────────────────────

def _root_caption(clf: DecisionTreeClassifier) -> str:
    t          = clf.tree_
    feat       = _FEATURES[t.feature[0]]
    thresh     = t.threshold[0]
    left_idx   = t.children_left[0]
    right_idx  = t.children_right[0]
    left_n     = int(t.n_node_samples[left_idx])
    right_n    = int(t.n_node_samples[right_idx])
    left_vals  = t.value[left_idx][0]
    right_vals = t.value[right_idx][0]
    left_dom   = _CLASS_NAMES[int(np.argmax(left_vals))]
    left_pct   = 100.0 * float(np.max(left_vals)) / float(left_vals.sum())
    right_dom  = _CLASS_NAMES[int(np.argmax(right_vals))]
    right_pct  = 100.0 * float(np.max(right_vals)) / float(right_vals.sum())
    return (
        f"Root: {feat} ≤ {thresh:.2f} → {left_n} samples "
        f"(mostly {left_dom}, {left_pct:.0f}%) "
        f"| > {thresh:.2f} → {right_n} samples "
        f"(mostly {right_dom}, {right_pct:.0f}%)"
    )


# ── Pydantic models ───────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    max_depth:        int = 3
    min_samples_leaf: int = 5
    feature_x:        str = 'bill_depth_mm'
    feature_y:        str = 'flipper_length_mm'


class DepthResult(BaseModel):
    depth:       int
    tree_src:    str
    boundary_png:str
    train_acc:   float
    test_acc:    float


class TrainResponse(BaseModel):
    depths:              list[DepthResult]
    feature_importances: dict[str, float]
    root_caption:        str
    scatter_x:           list[float]
    scatter_y:           list[float]
    scatter_species:     list[int]


class RFRequest(BaseModel):
    feature_x: str = 'bill_depth_mm'
    feature_y: str = 'flipper_length_mm'


class RFResponse(BaseModel):
    boundary_png:        str
    train_acc:           float
    test_acc:            float
    feature_importances: dict[str, float]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post('/train', response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    depths:     list[DepthResult]          = []
    first_clf:  DecisionTreeClassifier | None = None
    last_clf:   DecisionTreeClassifier | None = None

    for d in range(1, req.max_depth + 1):
        clf = DecisionTreeClassifier(
            max_depth=d, min_samples_leaf=req.min_samples_leaf, random_state=42,
        )
        clf.fit(_X_train, _y_train)
        if first_clf is None:
            first_clf = clf
        last_clf = clf

        depths.append(DepthResult(
            depth=d,
            tree_src=_tree_src(clf),
            boundary_png=_boundary_png_b64(clf, req.feature_x, req.feature_y),
            train_acc=round(float(clf.score(_X_train, _y_train)), 4),
            test_acc=round(float(clf.score(_X_test, _y_test)), 4),
        ))

    importances = {
        f: round(float(v), 4)
        for f, v in zip(_FEATURES, last_clf.feature_importances_)
    }
    xi = _FEATURES.index(req.feature_x)
    yi = _FEATURES.index(req.feature_y)

    return TrainResponse(
        depths=depths,
        feature_importances=importances,
        root_caption=_root_caption(first_clf),
        scatter_x=[float(v) for v in _X_all[:, xi]],
        scatter_y=[float(v) for v in _X_all[:, yi]],
        scatter_species=[int(v) for v in _y_all],
    )


@router.post('/train-rf', response_model=RFResponse)
def train_rf(req: RFRequest) -> RFResponse:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(_X_train, _y_train)
    importances = {
        f: round(float(v), 4)
        for f, v in zip(_FEATURES, rf.feature_importances_)
    }
    return RFResponse(
        boundary_png=_boundary_png_b64(rf, req.feature_x, req.feature_y),
        train_acc=round(float(rf.score(_X_train, _y_train)), 4),
        test_acc=round(float(rf.score(_X_test, _y_test)), 4),
        feature_importances=importances,
    )
