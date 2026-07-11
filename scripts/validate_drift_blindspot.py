#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 — the reconstruction-drift BLIND SPOT (a research finding, read-only).

Core finding, recorded rather than hidden: an unsupervised reconstruction-based
drift detector CANNOT flag the dataset's own noisy-LO "domain shift", because
noisy LO is **class-conditional / on-manifold** — its engineered features
INTERPOLATE between LN and MED (both in training), so PCA reconstructs them fine.
The same detector DOES catch a genuine off-manifold shift (an injected
current-sensor gain). So the blind spot is precise: reconstruction/PSI drift
detection sees distributional/operating-condition shifts, NOT subtle
class-boundary shifts — which is exactly what the §11 domain-shift "triple
evidence" (train-LO n=65 imbalance, SMOTE/CTGAN failure, FLAML holdout collapse)
was about. The two findings corroborate each other.

Writes ``outputs/metrics/drift_blindspot.json`` + ``outputs/figures/drift_blindspot.png``.

Run::

    python scripts/validate_drift_blindspot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.monitor.drift_detector import (  # noqa: E402
    build_drift_baseline, inject_sensor_drift_features, reconstruction_error,
)
from src.utils.paths import load_config, resolve  # noqa: E402

_ORDER = ["LN", "MED", "HI", "LO"]
_GAIN = 1.3


def _mean_err(frame, scaler, pca, cols) -> float:
    if not len(frame):
        return float("nan")
    return float(np.mean([reconstruction_error(scaler, pca, dict(r), cols)
                          for _, r in frame.iterrows()]))


def run() -> Path:
    df = pd.read_parquet(resolve(load_config()["servo"]["processed_features"]))
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

    result = {"finding": ("noisy LO is class-conditional / on-manifold (interpolates "
                          "LN↔MED) → NOT reconstruction-detectable; an injected sensor "
                          "gain IS. Reconstruction drift sees operating-condition shifts, "
                          "not class-boundary shifts."),
              "gain_injected": _GAIN, "spaces": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax, fs in zip(axes, ["engineered", "full"]):
        # drift AE fit on v1-lite training (LN/MED/HI — noisy LO EXCLUDED)
        scaler, pca, bl = build_drift_baseline(tr[tr["ylabel"] != "LO"], "v1-lite",
                                               ae_feature_set=fs)
        cols, p95 = bl["ae_columns"], bl["recon_error_p95"]
        per_class = {c: _mean_err(te[te["ylabel"] == c], scaler, pca, cols) for c in _ORDER}
        injected = _mean_err(inject_sensor_drift_features(te[te["ylabel"] == "HI"], _GAIN),
                             scaler, pca, cols)
        result["spaces"][fs] = {
            "n_components": bl["pca_n_components"],
            "explained_variance": bl["pca_explained_variance"],
            "p95": round(p95, 5),
            "recon_error_by_class": {k: round(v, 5) for k, v in per_class.items()},
            "injected_sensor_drift_recon_error": round(injected, 5),
            "noisy_lo_separable": bool(per_class["LO"] > p95),
            "injected_separable": bool(injected > p95),
        }
        labels = _ORDER + ["HI+injected\nsensor drift"]
        vals = [per_class[c] for c in _ORDER] + [injected]
        colors = ["#4C78A8"] * 4 + ["#E45756"]
        ax.bar(labels, vals, color=colors)
        ax.axhline(p95, color="#555", ls="--", lw=1, label=f"drift P95={p95:.3f}")
        ax.set_yscale("log")
        ax.set_title(f"{fs} space (PCA {bl['pca_n_components']}c, "
                     f"{bl['pca_explained_variance']:.0%} var)")
        ax.set_ylabel("mean reconstruction error (log)")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Drift blind spot: noisy LO (on-manifold, class-conditional) is NOT "
                 "separable; injected sensor drift (off-manifold) IS",
                 fontsize=11)
    fig.tight_layout()
    out_fig = resolve("outputs/figures/drift_blindspot.png")
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)

    out_json = resolve("outputs/metrics/drift_blindspot.json")
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== drift reconstruction blind spot ===")
    for fs, s in result["spaces"].items():
        print(f"[{fs}] P95={s['p95']}  by_class={s['recon_error_by_class']}  "
              f"injected={s['injected_sensor_drift_recon_error']}")
        print(f"    noisy_lo_separable={s['noisy_lo_separable']}  "
              f"injected_separable={s['injected_separable']}")
    print(f"\n  -> {out_json}\n  -> {out_fig}")
    return out_json


if __name__ == "__main__":
    run()
