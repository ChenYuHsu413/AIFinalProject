"""Phase B — offline 1D-CNN on the raw Servo waveforms (read-only on the cloud).

Trains on the raw-signal *energy-envelope* dataset from ``build_servo_windows``
(8 physical channels × 256 time-blocks — per-block std of the raw waveform): a
**1D-CNN classifier** (health state LN/LO/MED/HI) and a **1D-conv autoencoder**
(fit on healthy-train envelopes; test reconstruction error rises with degradation).
Writes ``outputs/metrics/servo_cnn_results.json`` for the dashboard.

This is the "real CNN" deferred by ``servo_dl`` (which only had MLP + AE on the
per-run aggregated features). torch lives in ``requirements-dl.txt`` (offline only).

Honesty: trained on an MVP **subset** of windows (see build_servo_windows); split
is by source file (no leakage). FMCRD is a high-fidelity *simulation* dataset.

Run::

    python -m src.data.build_servo_windows   # first, build the .npz from the zip
    python -m src.models.servo_cnn
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.paths import ensure_output_dirs, load_config, resolve


def _set_seed(rs: int) -> None:
    np.random.seed(rs)
    torch.manual_seed(rs)


class _CNN(nn.Module):
    """Small 1D-CNN classifier: conv stack -> global avg pool -> linear.

    Width/dropout were tuned on the larger 80-run set: a multi-seed sweep showed
    wider conv + dropout did NOT robustly beat this narrow conv-only model
    (means within one std of each other, ~0.64-0.67 macro-F1), so the simpler
    model is kept. The per-run dataset is small, so the score is seed-sensitive.
    """

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):  # noqa: D401
        return self.head(self.features(x).squeeze(-1))


class _ConvAE(nn.Module):
    """1D conv autoencoder: downsample then reconstruct the window."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(16, 8, 7, stride=4, padding=3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 8, stride=4, padding=2), nn.ReLU(),
            nn.ConvTranspose1d(16, in_ch, 8, stride=4, padding=2),
        )

    def forward(self, x):  # noqa: D401
        return self.decoder(self.encoder(x))


def _train(model, X, y, loss_fn, epochs, rs, batch=32, lr=1e-3):
    """Deterministic mini-batch Adam (fixed per-epoch permutation, CPU)."""
    _set_seed(rs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    rng = np.random.default_rng(rs)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()["servo"]
    rs = int(cfg.get("random_state", 42))
    _set_seed(rs)

    npz_path = resolve(cfg["windows_path"])
    if not npz_path.exists():
        raise FileNotFoundError(
            f"找不到 {npz_path}；請先執行 `python -m src.data.build_servo_windows`。")
    d = np.load(npz_path, allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    channels = [str(c) for c in d["channels"]]
    labels = [str(c) for c in d["labels"]]
    win_len = int(d["win_len"])

    tr = split == "train"
    te = ~tr
    # Per-channel standardisation using TRAIN windows only (no leakage).
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xtr, Xte = torch.from_numpy(Xs[tr]), torch.from_numpy(Xs[te])
    ytr = torch.from_numpy(y[tr])
    yte = y[te]

    # --- 1D-CNN classifier ---
    _set_seed(rs)  # seed before construction so init is independently reproducible
    clf = _CNN(len(channels), len(labels))
    _train(clf, Xtr, ytr, nn.CrossEntropyLoss(), epochs=60, rs=rs, batch=16)
    with torch.no_grad():
        pred = clf(Xte).argmax(1).numpy()
    acc = float((pred == yte).mean())
    macro_f1 = float(f1_score(yte, pred, average="macro",
                              labels=list(range(len(labels))), zero_division=0))
    cm = confusion_matrix(yte, pred, labels=list(range(len(labels)))).tolist()

    # --- 1D conv autoencoder: fit on healthy (LN) TRAIN envelopes ---
    ln_idx = labels.index("LN") if "LN" in labels else 0
    ln_mask = tr & (y == ln_idx)
    fit = torch.from_numpy(Xs[ln_mask]) if ln_mask.sum() >= 8 else Xtr
    _set_seed(rs)  # seed before construction so init is independently reproducible
    ae = _ConvAE(len(channels))
    _train(ae, fit, fit, nn.MSELoss(), epochs=60, rs=rs)
    # Reconstruction error per class on the TEST envelopes only (no in-sample
    # deflation of the healthy bucket — consistent with the holdout eval).
    with torch.no_grad():
        recon = ae(Xte).numpy()
    err = np.mean((Xs[te] - recon) ** 2, axis=(1, 2))
    per_class = {lab: float(err[yte == i].mean()) for i, lab in enumerate(labels)
                 if (yte == i).any()}

    out = {
        "method": "servo_cnn_1d",
        "eval": "holdout_test_by_file",
        "framework": f"pytorch {torch.__version__}",
        "note": ("真正的 1D-CNN（卷積分類）+ 1D conv-autoencoder，吃每段 run 的原始波形能量包絡"
                 "（逐塊 std，{} 通道 × {} 時間塊）。MVP 子集、split 依來源檔分離（無洩漏）；"
                 "FMCRD 為高擬真模擬資料。".format(len(channels), win_len)),
        "window": {
            "len": win_len, "channels": channels,
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "subset": True,
        },
        "architecture": {
            "cnn": f"Conv1d[{len(channels)}>16>32>64] k7/7/5 s2 + GAP + FC{len(labels)}",
            "autoencoder": f"Conv1d[{len(channels)}>16>8] s4 + ConvT decode",
        },
        "classifier": {
            "accuracy": acc, "macro_f1": macro_f1,
            "confusion_matrix": cm, "labels": labels,
        },
        "autoencoder": {"reconstruction_error_by_class": per_class},
    }
    p = resolve(cfg["cnn_metrics"])
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Servo CNN] (torch {torch.__version__}) acc={acc:.3f} macro-F1={macro_f1:.3f}")
    print(f"    AE 重建誤差/類別：{ {k: round(v, 3) for k, v in per_class.items()} }")
    print(f"    -> {p}")
    return p


if __name__ == "__main__":
    run()
