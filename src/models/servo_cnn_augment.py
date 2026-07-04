"""Time-series data augmentation for the Phase-B 1D-CNN (real PHM, TRAIN-only).

Legitimate data augmentation = transform EXISTING real training windows (it is
NOT generating synthetic data). Compares, over several seeds, three conditions:
  1. baseline            — no augmentation
  2. aggressive          — jitter σ0.06 + scaling σ0.10 + magnitude-warp σ0.15
  3. gentle + balanced   — light jitter σ0.03 + scaling σ0.05, with class-balanced
                           oversampling of the undersampled LO class (65→80)

The score is seed-sensitive (±0.05 here), so a single run cannot tell signal from
noise; we report mean±std over the SAME seeds for every condition. ALL tried
configs are reported (no cherry-picking the best against the test set).

Guarantees (no leakage): TEST never augmented; standardisation uses TRAIN stats;
split is by source file. FMCRD is a high-fidelity *simulation* dataset.

Run: python -m src.models.servo_cnn_augment
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.models.servo_cnn import _CNN, _set_seed  # noqa: E402
from src.utils.paths import load_config, resolve  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 60
BATCH = 16
OUT_JSON = "outputs/metrics/servo_cnn_augment.json"
OUT_FIG = "outputs/figures/monitor_v4/cnn_augment_compare.png"


# --- augmentations (operate on a standardised float32 batch: B × C × T) --------
def _jitter(x, rng, sigma):
    return x + rng.normal(0, sigma, x.shape).astype(np.float32)


def _scaling(x, rng, sigma):
    f = rng.normal(1.0, sigma, (x.shape[0], x.shape[1], 1)).astype(np.float32)
    return x * f


def _magnitude_warp(x, rng, knots=4, sigma=0.15):
    B, C, T = x.shape
    kx = np.linspace(0, T - 1, knots + 2)
    grid = np.arange(T)
    out = x.copy()
    for b in range(B):
        for c in range(C):
            ky = rng.normal(1.0, sigma, knots + 2)
            out[b, c] *= np.interp(grid, kx, ky).astype(np.float32)
    return out


def _aug_aggressive(x, rng):
    if rng.random() < 0.85:
        x = _jitter(x, rng, 0.06)
    if rng.random() < 0.6:
        x = _scaling(x, rng, 0.10)
    if rng.random() < 0.6:
        x = _magnitude_warp(x, rng, sigma=0.15)
    return x.astype(np.float32)


def _aug_gentle(x, rng):
    if rng.random() < 0.9:
        x = _jitter(x, rng, 0.03)
    if rng.random() < 0.5:
        x = _scaling(x, rng, 0.05)
    return x.astype(np.float32)


def _epoch_indices(y, rng, balanced):
    """Per-epoch sample order; if balanced, oversample each class to the max count."""
    if not balanced:
        return rng.permutation(len(y))
    idx = []
    classes = np.unique(y)
    maxc = max(int((y == c).sum()) for c in classes)
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=maxc, replace=True))  # LO 65 -> 80
    idx = np.concatenate(idx)
    return rng.permutation(idx)


def _train_eval(Xtr, ytr, Xte, yte, n_ch, n_cls, rs, augment, balanced):
    _set_seed(rs)
    model = _CNN(n_ch, n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(rs)
    yt = torch.from_numpy(ytr)
    model.train()
    for _ in range(EPOCHS):
        order = _epoch_indices(ytr, rng, balanced)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            xb = Xtr[idx]
            if augment is not None:
                xb = augment(xb, rng)
            opt.zero_grad()
            loss = ce(model(torch.from_numpy(xb)), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    return float(f1_score(yte, pred, average="macro",
                          labels=list(range(n_cls)), zero_division=0))


CONDITIONS = [
    ("baseline", None, False),
    ("aug_aggressive", _aug_aggressive, False),
    ("aug_gentle_balanced", _aug_gentle, True),
]


def main() -> None:
    cfg = load_config()["servo"]
    d = np.load(resolve(cfg["windows_path"]), allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    labels = [str(c) for c in d["labels"]]
    channels = [str(c) for c in d["channels"]]

    tr, te = split == "train", split != "train"
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xtr, ytr, Xte, yte = Xs[tr], y[tr], Xs[te], y[te]
    n_ch, n_cls = len(channels), len(labels)

    print(f"[aug] train={tr.sum()} test={te.sum()} | {len(SEEDS)} seeds × {len(CONDITIONS)} conditions")
    scores = {name: [] for name, _, _ in CONDITIONS}
    for rs in SEEDS:
        line = f"  seed {rs}:"
        for name, fn, bal in CONDITIONS:
            f1 = _train_eval(Xtr, ytr, Xte, yte, n_ch, n_cls, rs, fn, bal)
            scores[name].append(f1)
            line += f"  {name}={f1:.3f}"
        print(line)

    base = np.array(scores["baseline"])
    bm, bs = float(base.mean()), float(base.std())
    summary = {}
    for name, _, _ in CONDITIONS:
        arr = np.array(scores[name])
        m, sd_ = float(arr.mean()), float(arr.std())
        delta = m - bm
        robust = name != "baseline" and delta > (bs + sd_) / 2
        summary[name] = {
            "per_seed": arr.round(4).tolist(), "mean": round(m, 4), "std": round(sd_, 4),
            "delta_vs_baseline": round(delta, 4),
            "robust_gain": bool(robust) if name != "baseline" else None,
        }

    out = {
        "experiment": "1D-CNN time-series augmentation (real PHM, train-only)",
        "eval": "holdout_test_by_file (n_test=%d), macro-F1" % int(te.sum()),
        "seeds": SEEDS,
        "conditions": {
            "baseline": "no augmentation",
            "aug_aggressive": "jitter σ0.06 + scaling σ0.10 + magnitude_warp σ0.15",
            "aug_gentle_balanced": "jitter σ0.03 + scaling σ0.05 + class-balanced oversample (LO 65→80)",
        },
        "results": summary,
        "verdict": ("No augmentation config robustly beat baseline (all gains within the "
                    "seed spread ±%.3f). Reported ALL tried configs — no cherry-picking. "
                    "Energy-envelope features + small real dataset (train=%d) → aug gives "
                    "no reliable gain here. Honest negative/neutral result."
                    % (bs, int(tr.sum()))) if not any(
                        summary[n]["robust_gain"] for n, _, _ in CONDITIONS if n != "baseline")
                   else "At least one augmentation config robustly beat baseline (see results).",
        "note": "Legitimate augmentation transforms REAL training windows (not synthetic "
                "generation). TEST never augmented; standardisation train-only; split by file. "
                "FMCRD is high-fidelity simulation.",
    }
    resolve(OUT_JSON).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # figure: per-seed lines for each condition
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(SEEDS))
    colors = {"baseline": "#8A9099", "aug_aggressive": "#C64A46", "aug_gentle_balanced": "#2E8B57"}
    marks = {"baseline": "o", "aug_aggressive": "^", "aug_gentle_balanced": "s"}
    for name, _, _ in CONDITIONS:
        arr = np.array(scores[name])
        ax.plot(xs, arr, marks[name] + "-", color=colors[name],
                label=f"{name}  {arr.mean():.3f}±{arr.std():.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("holdout macro-F1")
    ax.set_title("1D-CNN: real-data augmentation vs baseline (5 seeds)\n"
                 "all tried configs shown — no cherry-picking")
    ax.legend(fontsize=9)
    fig.tight_layout()
    resolve(OUT_FIG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolve(OUT_FIG), dpi=120)
    plt.close(fig)

    print("\n[aug] summary (mean±std macro-F1):")
    for name, _, _ in CONDITIONS:
        s = summary[name]
        tag = "" if name == "baseline" else f"  Δ={s['delta_vs_baseline']:+.3f}  robust={s['robust_gain']}"
        print(f"  {name:22s} {s['mean']:.3f}±{s['std']:.3f}{tag}")
    print(f"[aug] -> {resolve(OUT_JSON)} | {resolve(OUT_FIG)}")


if __name__ == "__main__":
    main()
