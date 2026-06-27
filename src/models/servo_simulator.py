"""Training simulator + shared model factories for Module Servo.

Powers the in-browser "AI 訓練模擬器": pick a sample size, a feature set and an
algorithm; train a small model on the demo feature table and report metrics +
training time.  The same factories build the offline reference model
(``train_servo``) so the simulator and the reference model are comparable.

Five algorithms each for classification and regression, all sklearn-only so the
cloud runtime needs no extra dependency.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.features.servo_features import HEALTH_LABELS, feature_set_columns


# --- algorithm factories (name -> callable(random_state) -> estimator) ---
def _clf_factories() -> Dict[str, Any]:
    return {
        "logistic_regression": lambda rs: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs),
        "decision_tree": lambda rs: DecisionTreeClassifier(
            class_weight="balanced", random_state=rs),
        "random_forest": lambda rs: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=rs),
        "gradient_boosting": lambda rs: GradientBoostingClassifier(random_state=rs),
        "mlp": lambda rs: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=400, random_state=rs),
    }


def _reg_factories() -> Dict[str, Any]:
    return {
        "ridge": lambda rs: Ridge(random_state=rs),
        "decision_tree": lambda rs: DecisionTreeRegressor(random_state=rs),
        "random_forest": lambda rs: RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=rs),
        "gradient_boosting": lambda rs: GradientBoostingRegressor(random_state=rs),
        "mlp": lambda rs: MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=600, random_state=rs),
    }


ALGO_LABELS = {
    "logistic_regression": "Logistic Regression",
    "ridge": "Ridge Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP Neural Network",
}

CLASSIFIER_NAMES = list(_clf_factories().keys())
REGRESSOR_NAMES = list(_reg_factories().keys())


def build_classifier(name: str, random_state: int = 42) -> Pipeline:
    est = _clf_factories()[name](random_state)
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def build_regressor(name: str, random_state: int = 42) -> Pipeline:
    est = _reg_factories()[name](random_state)
    return Pipeline([("scaler", StandardScaler()), ("reg", est)])


def _subsample(df: pd.DataFrame, n: int, label_col: str, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    # stratified sample for classification; plain random for the regression target
    if label_col == "ylabel":
        return (
            df.groupby("ylabel", group_keys=False)
            .sample(frac=n / len(df), random_state=seed)
            .reset_index(drop=True)
        )
    return df.sample(n, random_state=seed).reset_index(drop=True)


def run_classification(
    df: pd.DataFrame, feature_set: str, algo: str,
    n_samples: int, random_state: int = 42,
) -> Dict[str, Any]:
    """Train one small classifier and report metrics + timing."""
    cols = feature_set_columns(feature_set)
    data = _subsample(df, n_samples, "ylabel", random_state)
    X, y = data[cols], data["ylabel"]
    # Stratify only when every class has >=2 members; a small/imbalanced
    # subsample otherwise crashes train_test_split.
    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=strat)
    pipe = build_classifier(algo, random_state)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0
    yp = pipe.predict(Xte)
    labels = [c for c in HEALTH_LABELS if c in set(y)]
    return {
        "task": "classification",
        "algo": algo,
        "feature_set": feature_set,
        "n_samples": int(len(data)),
        "n_features": len(cols),
        "train_time_s": round(train_time, 4),
        "accuracy": float(accuracy_score(yte, yp)),
        "macro_f1": float(f1_score(yte, yp, average="macro", labels=labels, zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(yte, yp, labels=labels).tolist(),
    }


def run_regression(
    df: pd.DataFrame, feature_set: str, algo: str,
    n_samples: int, random_state: int = 42,
) -> Dict[str, Any]:
    """Train one small DV regressor and report metrics + timing."""
    cols = feature_set_columns(feature_set)
    data = _subsample(df, n_samples, "DV", random_state)
    X, y = data[cols], data["DV"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=random_state)
    pipe = build_regressor(algo, random_state)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0
    yp = pipe.predict(Xte)
    mae = float(mean_absolute_error(yte, yp))
    rmse = float(np.sqrt(np.mean((np.asarray(yte) - yp) ** 2)))
    return {
        "task": "regression",
        "algo": algo,
        "feature_set": feature_set,
        "n_samples": int(len(data)),
        "n_features": len(cols),
        "train_time_s": round(train_time, 4),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(yte, yp)),
    }


def explain_result(task: str, feature_set: str, n_samples: int) -> List[str]:
    """Plain-language notes on why size / features / algorithm matter."""
    notes = [
        f"資料量：本次用 {n_samples} 筆。資料越多，模型越能學到穩定規律、"
        "在新資料上的表現通常越好；太少則容易過擬合或指標波動大。",
        f"特徵組：選了「{feature_set}」。特徵越貼近退化的物理徵兆"
        "（如位置誤差、電流 RMS），模型越容易區分健康狀態。",
        "演算法：線性模型快但表達力有限；樹模型 / 梯度提升較能抓非線性，"
        "但需要較多資料；MLP 彈性大但對資料量與調參更敏感。",
        "這是教學用小模型，指標通常低於離線訓練的 Reference Model —— 這正是"
        "「資料量 + 特徵 + 模型」三者影響的直觀示範。",
    ]
    return notes
