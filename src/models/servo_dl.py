"""Offline deep-learning baseline for Module Servo (read-only on the cloud app).

Deep learning is the SECOND part of the project and must NOT block the main-line
MVP, so it runs OFFLINE and only writes a results JSON that the dashboard reads.
The cloud runtime never trains DL (torch lives in requirements-dl.txt only;
the cloud / Docker image installs requirements-dev.txt and stays torch-free).

What this produces today (PyTorch; MLP on ``dl_classifier_feature_set`` = full,
autoencoder on ``dl_ae_feature_set`` = engineered — decoupled 2026-07-10):
  * **MLP classifier + DV regressor** — small fully-connected nets (64, 32).
  * **Neural autoencoder** — fit on healthy (LN) runs only and measure
    reconstruction error per health class; error should rise with degradation.
    This is a *real* autoencoder (encoder-decoder MLP), replacing the earlier
    PCA-reconstruction stand-in.

A true **1D-CNN on the raw PHM time series** is the next step (see Module B+ /
the windowing builder) — it needs the raw FMCRD waveforms, not these per-run
aggregated features.

Run::

    python -m src.models.servo_dl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.utils.paths import ensure_output_dirs, load_config, resolve


def _set_seed(rs: int) -> None:
    np.random.seed(rs)
    torch.manual_seed(rs)


class _MLP(nn.Module):
    """Small fully-connected net for classification (logits) or regression."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(64, 32)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D401
        return self.net(x)


class _AE(nn.Module):
    """Encoder-decoder autoencoder: in_dim -> 4 -> bottleneck -> 4 -> in_dim."""

    def __init__(self, in_dim: int, bottleneck: int = 2):
        super().__init__()
        mid = max(bottleneck + 1, min(4, in_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, mid), nn.ReLU(), nn.Linear(mid, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, mid), nn.ReLU(), nn.Linear(mid, in_dim),
        )

    def forward(self, x):  # noqa: D401
        return self.decoder(self.encoder(x))


def _train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
           loss_fn: nn.Module, epochs: int, rs: int, lr: float = 1e-3) -> None:
    """Deterministic full-batch Adam training (data is tiny, CPU-only)."""
    _set_seed(rs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    model.eval()


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()["servo"]
    rs = int(cfg.get("random_state", 42))
    _set_seed(rs)

    df = pd.read_parquet(resolve(cfg["processed_features"]))
    # DL 特徵組拆分（2026-07-10，與 reference_feature_set 解耦，見 docs §11）：
    #   MLP 分類/回歸 -> dl_classifier_feature_set（預設 full，收 engineered->full 增益）
    #   autoencoder   -> dl_ae_feature_set（預設 engineered；重建誤差健康->退化的單調性是
    #                    AE 作為健康指標的核心承諾，不隨參考模型換 full 而放寬）
    clf_fs = cfg.get("dl_classifier_feature_set", cfg.get("reference_feature_set", "engineered"))
    ae_fs = cfg.get("dl_ae_feature_set", "engineered")
    cols_clf = feature_set_columns(clf_fs)
    cols_ae = feature_set_columns(ae_fs)
    yc = df["ylabel"].to_numpy()
    yv = df["DV"].to_numpy(dtype=np.float32)

    # Honour a provided train/test split (real PHM); else random 25% holdout.
    # A single train mask drives both feature sets so the two share the same rows.
    has_split = "split" in df.columns and {"train", "test"} <= set(df["split"])
    eval_mode = "holdout_test" if has_split else "cv"
    if has_split:
        tr = (df["split"] == "train").to_numpy()
    else:
        strat = yc if pd.Series(yc).value_counts().min() >= 2 else None
        itr, _ = train_test_split(np.arange(len(df)), test_size=0.25,
                                  random_state=rs, stratify=strat)
        tr = np.zeros(len(df), dtype=bool)
        tr[itr] = True

    def _scaled(cols):  # fit scaler on TRAIN rows only (no leakage), transform all
        Xr = df[cols].to_numpy()
        return StandardScaler().fit(Xr[tr]).transform(Xr).astype(np.float32)

    Xs_clf, Xs_ae = _scaled(cols_clf), _scaled(cols_ae)
    Xtr, Xte = Xs_clf[tr], Xs_clf[~tr]
    ytr_c, yte_c = yc[tr], yc[~tr]
    vtr, vte = yv[tr], yv[~tr]

    labels = [c for c in HEALTH_LABELS if c in set(yc)]
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    in_dim_clf, in_dim_ae = len(cols_clf), len(cols_ae)

    # --- MLP classifier (dl_classifier_feature_set) ---
    ytr_idx = torch.tensor([lab_to_idx[v] for v in ytr_c], dtype=torch.long)
    clf = _MLP(in_dim_clf, len(labels))
    _train(clf, torch.from_numpy(Xtr), ytr_idx, nn.CrossEntropyLoss(), epochs=400, rs=rs)
    with torch.no_grad():
        pred_idx = clf(torch.from_numpy(Xte)).argmax(1).numpy()
    pred_lab = np.array([labels[i] for i in pred_idx])
    mlp_clf_f1 = float(f1_score(yte_c, pred_lab, average="macro", labels=labels,
                                zero_division=0))

    # --- MLP DV regressor (dl_classifier_feature_set) ---
    reg = _MLP(in_dim_clf, 1)
    _train(reg, torch.from_numpy(Xtr), torch.from_numpy(vtr).unsqueeze(1),
           nn.MSELoss(), epochs=600, rs=rs)
    with torch.no_grad():
        vpred = reg(torch.from_numpy(Xte)).squeeze(1).numpy()
    mlp_reg = {"mae": float(mean_absolute_error(vte, vpred)),
               "r2": float(r2_score(vte, vpred))}

    # --- Neural autoencoder (dl_ae_feature_set): reconstruction error per class ---
    # Fit on healthy (LN) runs — TRAIN-only when a split is present so the AE does
    # not see test rows; fall back to all (train) rows if there are too few LN.
    ln_mask = (df["ylabel"] == "LN").to_numpy()
    fit_mask = (ln_mask & tr) if has_split else ln_mask
    bottleneck = min(2, in_dim_ae)
    if fit_mask.sum() > bottleneck + 1:
        fit_rows = Xs_ae[fit_mask]
    else:
        fit_rows = Xs_ae[tr] if has_split else Xs_ae
    ae = _AE(in_dim_ae, bottleneck=bottleneck)
    _train(ae, torch.from_numpy(fit_rows), torch.from_numpy(fit_rows),
           nn.MSELoss(), epochs=800, rs=rs)
    # Reconstruction error per class on the held-out TEST rows when a split is
    # present (no in-sample deflation of the healthy bucket — consistent with the
    # holdout eval); fall back to all rows in CV/placeholder mode.
    eval_mask = (~tr) if has_split else np.ones(len(df), dtype=bool)
    with torch.no_grad():
        recon = ae(torch.from_numpy(Xs_ae[eval_mask])).numpy()
    err = np.mean((Xs_ae[eval_mask] - recon) ** 2, axis=1)
    ylab_eval = df["ylabel"].to_numpy()[eval_mask]
    per_class = {lab: float(np.mean(err[ylab_eval == lab]))
                 for lab in labels if (ylab_eval == lab).any()}

    out = {
        "method": "servo_dl_torch",
        "eval": eval_mode,
        "placeholder": bool(cfg.get("placeholder", True)),
        "note": (f"PyTorch MLP 分類/回歸（特徵組 {clf_fs}）+ 神經 autoencoder"
                 f"（特徵組 {ae_fs}，健康資料擬合的重建誤差）；輸入為逐段聚合特徵；"
                 "直接吃原始 FMCRD 波形的真 1D-CNN 已另在「1D-CNN（原始波形）」一節完成。"),
        "framework": f"pytorch {torch.__version__}",
        "classifier_feature_set": clf_fs,
        "ae_feature_set": ae_fs,
        "architecture": {
            "mlp_hidden": [64, 32],
            "autoencoder": f"{in_dim_ae}->4->{bottleneck}->4->{in_dim_ae}",
        },
        "mlp_classification_macro_f1": mlp_clf_f1,
        "mlp_regression": mlp_reg,
        "reconstruction_error_by_class": per_class,
        "labels": labels,
    }
    p = resolve(cfg["dl_metrics"])
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Servo DL] (torch {torch.__version__}) MLP macro-F1={mlp_clf_f1:.3f}, "
          f"reg R2={mlp_reg['r2']:.3f}")
    print(f"    AE 重建誤差/類別：{ {k: round(v,3) for k,v in per_class.items()} }")
    print(f"    -> {p}")
    return p


if __name__ == "__main__":
    run()
