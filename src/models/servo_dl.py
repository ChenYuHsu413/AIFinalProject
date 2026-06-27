"""Offline deep-learning baseline for Module Servo (read-only on the cloud app).

Deep learning is the SECOND part of the project and must NOT block the main-line
MVP, so it runs OFFLINE and only writes a results JSON that the dashboard reads.
The cloud runtime never trains DL.

What this produces today (sklearn-only, no torch/tensorflow dependency):
  * **MLP baseline** — engineered-feature classification + DV regression metrics.
  * **Reconstruction-error analysis** — fit PCA on healthy (LN) runs only and
    measure reconstruction error per health class; error should rise with
    degradation.  This is a PCA reconstruction baseline that stands in for a
    neural autoencoder — a true 1D-CNN / autoencoder on the raw PHM time series
    is deferred until the real dataset is downloaded (needs offline torch).

Run::

    python -m src.models.servo_dl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.utils.paths import ensure_output_dirs, load_config, resolve


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()["servo"]
    rs = int(cfg.get("random_state", 42))
    df = pd.read_parquet(resolve(cfg["processed_features"]))
    cols = feature_set_columns(cfg.get("reference_feature_set", "engineered"))
    X = df[cols].to_numpy()

    # --- MLP baseline (classification + DV regression) ---
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    yc = df["ylabel"].to_numpy()
    # Stratify only when every class has >=2 members; tiny/imbalanced real data
    # otherwise crashes train_test_split.
    strat = yc if pd.Series(yc).value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(Xs, yc, test_size=0.25,
                                          random_state=rs, stratify=strat)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=rs).fit(Xtr, ytr)
    labels = [c for c in HEALTH_LABELS if c in set(yc)]
    mlp_clf_f1 = float(f1_score(yte, clf.predict(Xte), average="macro",
                                labels=labels, zero_division=0))

    yv = df["DV"].to_numpy()
    Xtr2, Xte2, vtr, vte = train_test_split(Xs, yv, test_size=0.25, random_state=rs)
    reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=rs).fit(Xtr2, vtr)
    pred = reg.predict(Xte2)
    mlp_reg = {"mae": float(mean_absolute_error(vte, pred)),
               "r2": float(r2_score(vte, pred))}

    # --- PCA reconstruction error (healthy-only fit), per class ---
    ln_mask = (df["ylabel"] == "LN").to_numpy()
    n_comp = min(3, len(cols))
    # Fit on healthy runs when there are enough; otherwise fall back to all rows
    # so a dataset without an LN class doesn't crash the baseline.
    fit_rows = Xs[ln_mask] if ln_mask.sum() > n_comp else Xs
    pca = PCA(n_components=n_comp, random_state=rs).fit(fit_rows)
    recon = pca.inverse_transform(pca.transform(Xs))
    err = np.mean((Xs - recon) ** 2, axis=1)
    per_class = {lab: float(np.mean(err[(df["ylabel"] == lab).to_numpy()]))
                 for lab in labels}

    out = {
        "method": "servo_dl_offline_baseline",
        "placeholder": bool(cfg.get("placeholder", True)),
        "note": ("sklearn MLP baseline + PCA 重建誤差（健康資料擬合）。真正的 1D-CNN / "
                 "Autoencoder 需離線 torch、且需真實 PHM 時序資料，列為後續工作。"),
        "mlp_classification_macro_f1": mlp_clf_f1,
        "mlp_regression": mlp_reg,
        "reconstruction_error_by_class": per_class,
        "labels": labels,
    }
    p = resolve(cfg["dl_metrics"])
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Servo DL] MLP macro-F1={mlp_clf_f1:.3f}, reg R²={mlp_reg['r2']:.3f}")
    print(f"    重建誤差/類別：{ {k: round(v,3) for k,v in per_class.items()} }")
    print(f"    -> {p}")
    return p


if __name__ == "__main__":
    run()
