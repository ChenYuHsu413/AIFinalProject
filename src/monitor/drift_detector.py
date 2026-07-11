"""S4 Task A — data-drift detector (drift ≠ degradation).

TWO autoencoders with OPPOSITE purposes — do not confuse them:

  * ``servo_dl`` neural AE = HEALTH indicator. Fit on LN (healthy) rows only, so
    its reconstruction error RISES with degradation (LN low → HI high). Rising
    error there is the desired signal.
  * THIS drift AE (a linear/PCA autoencoder) = DRIFT detector. Fit on the model
    version's FULL training distribution (all health levels). Degradation is then
    IN-distribution → low error is the desired signal; only data that is unlike
    anything in training (a genuine distribution shift) scores high. This is what
    makes "degradation ≠ drift" a STRUCTURAL guarantee, not a threshold gamble:
    an HI machine is normal work the model was trained on and must NOT trip drift.

The drift AE and its P95 baseline are VERSION-BOUND: each registry version stores
its own ``drift_ae.joblib`` + ``drift_baseline.json`` fit on THAT version's
training data. So noisy-LO is out-of-distribution for a v1-lite that never saw
it (→ DRIFT), but once retraining folds noisy-LO into v2, v2's drift AE re-fits
and the same data becomes in-distribution (→ drift digested, DRIFT clears).

Two complementary signals, both on the existing window pipeline
(``servo_replay_client``; no separate data channel):
  1. reconstruction error — rolling mean over the last K windows > training P95.
  2. PSI — per-feature Population Stability Index of the last K windows vs the
     training histogram; a feature with PSI > 0.2 counts as drifted.
Trigger (``config.yaml::servo_drift``): rolling error > P95 for N windows, OR
≥ ``psi_min_features`` features drifted for N windows → a DRIFT event.
"""
from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import joblib
import numpy as np

from src.features.servo_features import feature_set_columns
from src.utils.paths import load_config, resolve

DRIFT_AE_FILE = "drift_ae.joblib"
DRIFT_BASELINE_FILE = "drift_baseline.json"

# Injected sensor drift (S4 demo): a current-sensor gain/offset shift. Sensor
# gain drift is a REAL-WORLD failure mode; here it is INJECTED/SIMULATED to give
# the demo a genuine off-manifold distribution shift (the dataset's own noisy-LO
# shift is class-conditional, on-manifold, and NOT reconstruction-detectable —
# see scripts/validate_drift_blindspot.py).
RAW_CURRENT_CHANNELS = ["i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def feature_current_columns() -> List[str]:
    from src.features.servo_features import STATS
    return [f"{s}_{st}" for s in RAW_CURRENT_CHANNELS for st in STATS] + ["current_rms"]


def inject_sensor_drift_raw(row: Dict[str, Any], gain: float,
                            offset: float = 0.0) -> Dict[str, Any]:
    """Apply a current-sensor gain+offset to one RAW_COLUMNS row (publisher side)."""
    r = dict(row)
    for c in RAW_CURRENT_CHANNELS:
        if r.get(c) is not None:
            try:
                r[c] = float(r[c]) * gain + offset
            except (TypeError, ValueError):
                pass
    return r


def inject_sensor_drift_features(df, gain: float):
    """Apply the SAME gain to a feature table's current-derived columns.

    RMS/mean scale linearly with a raw gain, so scaling the aggregated current
    features is consistent with the raw-row injection — this lets the retrain
    fold the drifted operating condition into a version's TRAINING data.
    """
    d = df.copy()
    for c in feature_current_columns():
        if c in d.columns:
            d[c] = d[c] * gain
    return d


# ---------------------------------------------------------------------------
# Baseline building (per model version)
# ---------------------------------------------------------------------------
def build_drift_baseline(train_df, model_version: str,
                         ae_feature_set: str = "engineered",
                         explained_var: float = 0.95, psi_bins: int = 5):
    """Fit the drift AE (StandardScaler + PCA) on a version's TRAIN rows.

    PCA bottleneck is chosen by cumulative explained variance (default 95%) — a
    recorded, principled choice, not a guess. Returns (scaler, pca, baseline_dict).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = feature_set_columns(ae_feature_set)
    X = train_df[cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # n_components = fewest PCs reaching `explained_var`, kept a strict bottleneck.
    ratios = PCA().fit(Xs).explained_variance_ratio_
    cum = np.cumsum(ratios)
    n_comp = int(np.searchsorted(cum, explained_var) + 1)
    n_comp = max(1, min(n_comp, Xs.shape[1] - 1))
    pca = PCA(n_components=n_comp).fit(Xs)
    recon = pca.inverse_transform(pca.transform(Xs))
    err = np.mean((Xs - recon) ** 2, axis=1)

    psi_reference: Dict[str, Any] = {}
    for c in cols:
        v = train_df[c].to_numpy(dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, psi_bins + 1)))
        if len(edges) < 3:  # near-constant feature -> a trivial 2-bin split
            edges = np.array([v.min(), np.median(v), v.max() + 1e-9])
        edges[0], edges[-1] = -np.inf, np.inf  # open ends so live values always bin
        hist, _ = np.histogram(v, bins=edges)
        psi_reference[c] = {"edges": edges.tolist(),
                            "ref": (hist / max(hist.sum(), 1)).tolist()}

    baseline = {
        "model_version": model_version,
        "ae_feature_set": ae_feature_set,
        "ae_columns": cols,
        "pca_n_components": n_comp,
        "pca_explained_variance": round(float(cum[n_comp - 1]), 4),
        "explained_variance_target": explained_var,
        "recon_error_p95": float(np.percentile(err, 95)),
        "recon_error_train_mean": float(err.mean()),
        "psi_bins": psi_bins,
        "psi_reference": psi_reference,
        "n_train": int(len(train_df)),
        "created": _now_iso(),
    }
    return scaler, pca, baseline


def save_drift_baseline(version_dir: Path, scaler, pca, baseline: Dict[str, Any]) -> None:
    version_dir = Path(version_dir)
    joblib.dump({"scaler": scaler, "pca": pca}, version_dir / DRIFT_AE_FILE, compress=3)
    (version_dir / DRIFT_BASELINE_FILE).write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")


def load_drift_baseline(version_dir: Path):
    version_dir = Path(version_dir)
    ae = joblib.load(version_dir / DRIFT_AE_FILE)
    baseline = json.loads((version_dir / DRIFT_BASELINE_FILE).read_text(encoding="utf-8"))
    return ae["scaler"], ae["pca"], baseline


def reconstruction_error(scaler, pca, feat: Dict[str, float], cols: List[str]) -> float:
    x = np.array([[float(feat[c]) for c in cols]])
    xs = scaler.transform(x)
    r = pca.inverse_transform(pca.transform(xs))
    return float(np.mean((xs - r) ** 2))


def _psi(ref: List[float], act: List[float]) -> float:
    total = 0.0
    for e, a in zip(ref, act):
        e = max(e, 1e-6)
        a = max(a, 1e-6)
        total += (a - e) * math.log(a / e)
    return total


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class DriftDetector:
    """Rolling drift detector bound to one model version's baseline."""

    def __init__(self, version_dir: Path,
                 config: Optional[Dict[str, Any]] = None,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 alerts_dir: Optional[Path] = None):
        self.scaler, self.pca, self.baseline = load_drift_baseline(version_dir)
        self.cols = self.baseline["ae_columns"]
        self.p95 = float(self.baseline["recon_error_p95"])
        self.psi_ref = self.baseline["psi_reference"]
        self.model_version = self.baseline["model_version"]

        cfg = config if config is not None else load_config().get("servo_drift", {})
        self.N = int(cfg.get("sustain_windows", 3))
        self.K = int(cfg.get("psi_window", 8))
        self.psi_threshold = float(cfg.get("psi_threshold", 0.2))
        self.psi_min_features = int(cfg.get("psi_min_features", 3))

        acfg = load_config().get("servo_alert", {})
        self.alerts_dir = Path(alerts_dir) if alerts_dir else resolve(acfg.get("alerts_dir", "outputs/alerts"))
        self.on_event = on_event

        self.err_buf: Deque[float] = deque(maxlen=self.K)
        self.feat_buf: Deque[Dict[str, float]] = deque(maxlen=self.K)
        self.ae_streak = 0
        self.active = False
        self._counter = 0

    # -- signals ----------------------------------------------------------
    def error_of(self, feat: Dict[str, float]) -> float:
        return reconstruction_error(self.scaler, self.pca, feat, self.cols)

    def _psi_drifted_features(self) -> Dict[str, float]:
        """Per-feature PSI of the buffered windows vs the training histogram."""
        drifted: Dict[str, float] = {}
        rows = list(self.feat_buf)
        for c in self.cols:
            ref = self.psi_ref[c]
            edges = np.array(ref["edges"], dtype=float)
            vals = np.array([r[c] for r in rows], dtype=float)
            hist, _ = np.histogram(vals, bins=edges)
            act = (hist / max(hist.sum(), 1)).tolist()
            psi = _psi(ref["ref"], act)
            if psi > self.psi_threshold:
                drifted[c] = round(psi, 3)
        return drifted

    # -- event sink -------------------------------------------------------
    def _emit(self, event: Dict[str, Any]) -> None:
        event.setdefault("model_version", self.model_version)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        path = self.alerts_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _event(self, etype: str, roll_err: float, inst_err: float,
               psi_drifted: Dict[str, float], window: Dict[str, Any]) -> Dict[str, Any]:
        self._counter += 1
        return {
            "id": f"drift-{self._counter:04d}", "type": etype, "ts": _now_iso(),
            "stream_t": window.get("_stream_t"),
            "true_label_in_window": window.get("_true"),
            "rolling_recon_error": round(roll_err, 5),
            "instant_recon_error": round(inst_err, 5),
            "baseline_p95": round(self.p95, 5),
            "psi_drifted_features": psi_drifted,
            "reason": ("reconstruction error > P95 (資料不像訓練分布)"
                       if etype == "drift_detected" else "訊號回落至基線內（漂移已被重訓消化或情境結束）"),
        }

    # -- main step --------------------------------------------------------
    def update(self, window: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Feed one window (must carry ``_features`` = aggregate_run dict)."""
        feat = window["_features"]
        inst_err = self.error_of(feat)
        self.err_buf.append(inst_err)
        self.feat_buf.append({c: float(feat[c]) for c in self.cols})

        roll_err = float(np.mean(self.err_buf))
        ae_over = roll_err > self.p95
        # PSI is a DIAGNOSTIC only, NOT a trigger: over a homogeneous degradation
        # phase (e.g. a pure-HI stretch) a window block's marginals differ from the
        # mixed training distribution, so PSI fires on normal degradation too —
        # which would violate the hard "degradation ≠ drift" requirement. The
        # reconstruction error keeps degradation in-distribution, so it is the sole
        # trigger; PSI is reported alongside for interpretability. (See docs §15.)
        psi_drifted = self._psi_drifted_features() if len(self.feat_buf) >= min(self.K, 4) else {}

        self.ae_streak = self.ae_streak + 1 if ae_over else 0

        events: List[Dict[str, Any]] = []
        triggered = self.ae_streak >= self.N
        if triggered and not self.active:
            self.active = True
            ev = self._event("drift_detected", roll_err, inst_err, psi_drifted, window)
            self._emit(ev)
            events.append(ev)
        elif self.active and not ae_over:
            self.active = False
            ev = self._event("drift_cleared", roll_err, inst_err, psi_drifted, window)
            self._emit(ev)
            events.append(ev)
        return events


# ---------------------------------------------------------------------------
# Convenience: (re)build the active (or given) version's drift baseline
# ---------------------------------------------------------------------------
def build_for_version(version: str, train_df, ae_feature_set: Optional[str] = None) -> Dict[str, Any]:
    from src.models import servo_model_registry as registry

    cfg = load_config()["servo"]
    ae_fs = ae_feature_set or cfg.get("dl_ae_feature_set", "engineered")
    scaler, pca, baseline = build_drift_baseline(train_df, version, ae_feature_set=ae_fs)
    save_drift_baseline(registry.version_dir(version), scaler, pca, baseline)
    return baseline
