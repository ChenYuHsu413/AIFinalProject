"""Central registry of candidate classifiers.

A registered model is a callable returning a fresh estimator instance.  Models
that depend on optional packages (XGBoost, LightGBM) are registered only if
the import succeeds, so the project still trains end-to-end on a minimal
installation.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear", random_state=random_state
    )


def _decision_tree(random_state: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(class_weight="balanced", random_state=random_state)


def _random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state
    )


def _svm(random_state: int) -> SVC:
    return SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=random_state
    )


def _gradient_boosting(random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=random_state)


def _knn(random_state: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=7)


def _mlp(random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state
    )


def _naive_bayes(random_state: int) -> GaussianNB:
    return GaussianNB()


REGISTRY: Dict[str, Callable[[int], object]] = {
    "logistic_regression": _logistic_regression,
    "decision_tree": _decision_tree,
    "random_forest": _random_forest,
    "svm": _svm,
    "gradient_boosting": _gradient_boosting,
    "knn": _knn,
    "mlp": _mlp,
    "naive_bayes": _naive_bayes,
}


# Optional libraries — only registered when importable.
try:
    from xgboost import XGBClassifier  # type: ignore

    def _xgboost(random_state: int) -> "XGBClassifier":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state,
            n_jobs=-1, tree_method="hist",
        )
    REGISTRY["xgboost"] = _xgboost
except Exception:  # pragma: no cover
    pass

try:
    from lightgbm import LGBMClassifier  # type: ignore

    def _lightgbm(random_state: int) -> "LGBMClassifier":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=31,
            class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1,
        )
    REGISTRY["lightgbm"] = _lightgbm
except Exception:  # pragma: no cover
    pass


# Models that NEED numeric scaling for sensible results.
NEEDS_SCALING = {"logistic_regression", "svm", "knn", "mlp", "naive_bayes"}


def build(name: str, random_state: int = 42):
    """Instantiate a model by name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](random_state)


def available_models(requested: List[str]) -> List[str]:
    """Filter the requested list down to models that are actually registered."""
    return [m for m in requested if m in REGISTRY]
