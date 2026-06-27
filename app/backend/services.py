"""Thin service layer between the FastAPI handlers and the inference code.

Kept deliberately small so unit tests can target it directly.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.predict import (
    REQUIRED_INPUT_COLUMNS,
    load_failure_type_models,
    load_model,
    predict_full,
    predict_records,
    predict_single,
)
from src.utils.paths import load_config, resolve


def health() -> Dict[str, Any]:
    try:
        load_model()
        return {"status": "ok", "model_loaded": True, "message": None}
    except FileNotFoundError as e:
        return {"status": "model_missing", "model_loaded": False, "message": str(e)}


def model_info() -> Dict[str, Any]:
    bundle = load_model()
    return {
        "model_name": bundle.model_name,
        "feature_set": bundle.feature_set,
        "feature_columns": bundle.feature_columns,
        "metrics": bundle.metrics,
    }


def predict_one(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_single(record)


def predict_one_full(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_full(record)


def predict_explain(record: Dict[str, Any]) -> Dict[str, Any]:
    """SHAP explanation for a single record. ``supported`` is False when the
    best model is not a tree the TreeExplainer can handle (mirrors the UI note)."""
    from src.models.explain import explain_record, is_supported

    if not is_supported():
        return {"supported": False}
    e = explain_record(record)
    return {
        "supported": True,
        "feature_names": e.feature_names,
        "feature_values": e.feature_values,
        "shap_values": e.shap_values,
        "base_value": e.base_value,
        "model_output": e.model_output,
    }


def failure_type_metrics() -> List[Dict[str, Any]]:
    cfg = load_config()
    path = resolve(cfg["paths"]["failure_type_metrics"])
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def predict_batch_from_csv(content: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(content))
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"上傳的 CSV 缺少欄位：{missing}。"
            f"必要欄位：{REQUIRED_INPUT_COLUMNS}"
        )
    return predict_records(df[REQUIRED_INPUT_COLUMNS])


def predict_batch_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predict on a JSON list of raw records (What-if 1D/2D sweep, grids)."""
    if not records:
        return []
    return predict_records(records)


def comparison_metrics() -> List[Dict[str, Any]]:
    cfg = load_config()
    path = resolve(cfg["paths"]["metrics_csv"])
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def test_predictions() -> Dict[str, List]:
    """Test-set ground truth + scores for the interactive threshold tuner.

    Returns ``y_true`` / ``y_proba`` arrays; the confusion-matrix math runs on
    the client. Empty arrays if evaluation has not been run."""
    path = resolve(load_config()["paths"]["outputs_metrics"]) / "test_predictions.csv"
    if not path.exists():
        return {"y_true": [], "y_proba": []}
    df = pd.read_csv(path)
    return {
        "y_true": df["y_true"].astype(int).tolist(),
        "y_proba": df["y_proba"].astype(float).tolist(),
    }


def _read_json_or_empty(rel_path: str | Path) -> Dict[str, Any]:
    """Read a small project-relative JSON; empty dict if the file is missing."""
    path = resolve(rel_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Per-module KPI metric sources: module key -> {response key: (config section, path key)}.
# Module A's KPI strip is already served by /metrics + /model_info, so it is not listed here.
_METRICS_SUMMARY_SOURCES: Dict[str, Dict[str, tuple]] = {
    "servo": {"clf": ("servo", "clf_metrics"), "reg": ("servo", "reg_metrics")},
    "B": {"ims_rul": ("ims", "rul_metrics")},
    "Bplus": {
        "generalization": ("xjtu", "gen_metrics"),
        "lobo": ("xjtu", "lobo_metrics"),
        "loco": ("xjtu", "loco_metrics"),
    },
    "C": {"paderborn": ("paderborn", "metrics")},
}


def metrics_summary(module: str) -> Dict[str, Any]:
    """Return the raw KPI metric JSON(s) for a module's overview strip.

    Missing files map to an empty dict (same as the Streamlit reader) so the
    frontend can render placeholders before the offline jobs have run.
    """
    sources = _METRICS_SUMMARY_SOURCES.get(module)
    if sources is None:
        allowed = ", ".join(_METRICS_SUMMARY_SOURCES)
        raise ValueError(f"未知的 module：{module!r}。可用：{allowed}")
    cfg = load_config()
    out: Dict[str, Any] = {}
    for key, (section, path_key) in sources.items():
        out[key] = _read_json_or_empty(cfg[section][path_key])
    return out


def paderborn_eval() -> Dict[str, Any]:
    """Full Paderborn (Module C) evaluation: method / features / results / summary.

    Single source for the whole Module C page (baseline CV vs artificial→real
    generalization + confusion matrices). Empty dict if the job has not run.
    """
    return _read_json_or_empty(load_config()["paderborn"]["metrics"])


def paderborn_samples() -> List[Dict[str, Any]]:
    """A few representative measurements per (damage_origin, fault_class) for the
    live-inference picker: identity + true label + the feature row to score."""
    from src.models.train_paderborn import feature_columns

    fp = resolve(load_config()["paderborn"]["processed_features"])
    if not fp.exists():
        return []
    df = pd.read_parquet(fp)
    feats = feature_columns(df)
    sample = (df.groupby(["damage_origin", "fault_class"], group_keys=False)
              .head(3).reset_index(drop=True))
    rows: List[Dict[str, Any]] = []
    for _, r in sample.iterrows():
        rows.append({
            "bearing_code": str(r["bearing_code"]),
            "condition": str(r["condition"]),
            "measurement": int(r["measurement"]),
            "fault_class": str(r["fault_class"]),
            "damage_origin": str(r["damage_origin"]),
            "features": {c: float(r[c]) for c in feats},
        })
    return rows


def paderborn_predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run the trained Paderborn classifier on one feature row (live inference)."""
    from src.models.predict_paderborn import predict_paderborn

    return predict_paderborn(features)


# --- Module B+ (XJTU multi-trajectory generalization) ------------------------
def xjtu_generalization() -> Dict[str, Any]:
    """Fixed-param per-bearing degradation detection: method / per_bearing / aggregate."""
    return _read_json_or_empty(load_config()["xjtu"]["gen_metrics"])


def xjtu_lobo_loco() -> Dict[str, Any]:
    """Supervised RUL generalization: leave-one-bearing-out + leave-one-condition-out."""
    cfg = load_config()
    return {
        "lobo": _read_json_or_empty(cfg["xjtu"]["lobo_metrics"]),
        "loco": _read_json_or_empty(cfg["xjtu"]["loco_metrics"]),
    }


def xjtu_domain_adapt() -> Dict[str, Any]:
    """E1 cross-condition domain adaptation ablation: results / summary."""
    return _read_json_or_empty(load_config()["xjtu"]["domain_adapt"]["da_metrics"])


def xjtu_rul_predictions() -> List[Dict[str, Any]]:
    """Per-bearing RUL/health prediction table (E2 maintenance-advice input)."""
    path = resolve(load_config()["xjtu"]["rul_predictions"])
    if not path.exists():
        return []
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def xjtu_health_overlay() -> Dict[str, Any]:
    """Smoothed health-indicator curves (vs % of life) for all bearings + FPT markers.

    Each curve is downsampled to <=200 points; coloured by condition on the
    client. ``available`` is False if the XJTU feature table is not built."""
    xj = load_config()["xjtu"]
    feat_path = resolve(xj["processed_features"])
    if not feat_path.exists():
        return {"available": False, "reason": "XJTU 特徵表尚未建立。"}
    feat = pd.read_parquet(feat_path)
    summary_path = resolve(xj["gen_summary"])
    fpt_by: Dict[tuple, int] = {}
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        fpt_by = {(r["condition"], r["bearing"]): int(r["fpt_index"]) for _, r in s.iterrows()}
    ind, sw = xj["health_indicator"], xj["hi_smooth_window"]
    curves, markers = [], []
    for (cond, bearing), g in feat.groupby(["condition", "bearing"], sort=False):
        g = g.sort_values("minute")
        hi = g[ind].rolling(sw, min_periods=1).median().to_numpy()
        n = len(g)
        life_pct = np.linspace(0, 100, n)
        idx = np.linspace(0, n - 1, min(200, n)).astype(int)
        curves.append({"condition": cond, "bearing": bearing,
                       "life_pct": life_pct[idx].tolist(), "hi": hi[idx].tolist()})
        fi = min(fpt_by.get((cond, bearing), 0), n - 1)
        markers.append({"condition": cond, "bearing": bearing,
                        "life_pct": float(life_pct[fi]), "hi": float(hi[fi])})
    return {"available": True, "curves": curves, "fpt_markers": markers}


def xjtu_replay(condition: str, bearing: str) -> Dict[str, Any]:
    """Precomputed streaming-replay frames for one bearing (<=100 frames).

    Reuses build_health_indicator / detect_fpt / extrapolate_rul / maintenance_advice
    once on the full series — the rolling RUL fit is backward-looking, so frame k
    equals what the system would compute having seen only the first k snapshots."""
    from src.models.maintenance_advice import maintenance_advice
    from src.models.rul_extrapolation import (
        build_health_indicator,
        detect_fpt,
        extrapolate_rul,
    )

    cfg = load_config()
    xj = cfg["xjtu"]
    feat_path = resolve(xj["processed_features"])
    if not feat_path.exists():
        return {"available": False, "reason": "XJTU 特徵表尚未建立。"}
    feat = pd.read_parquet(feat_path)
    g = (feat[(feat["condition"] == condition) & (feat["bearing"] == bearing)]
         .sort_values("minute").reset_index(drop=True))
    if g.empty:
        raise ValueError(f"找不到軌跡：condition={condition!r}, bearing={bearing!r}")

    ind, sw = xj["health_indicator"], xj["hi_smooth_window"]
    margin = float(cfg.get("maintenance", {}).get("safety_margin", 0.3))
    alarm = float(xj["alarm_health"])

    n = len(g)
    hi_full, health_full, hi_base, hi_fail = build_health_indicator(
        g[ind], sw, xj["baseline_n"], xj["fail_percentile"])
    fpt_idx = detect_fpt(hi_full, xj["baseline_n"], xj["fpt_n_sigma"], xj["fpt_consecutive"])
    minutes = g["minute"].to_numpy().astype(float)
    hours = (minutes - minutes[0]) / 60.0
    hi_arr = hi_full.to_numpy()
    rul_full = extrapolate_rul(hours, hi_arr, hi_base, hi_fail, fpt_idx,
                               xj["min_fit_points"], xj["fit_window"], float(hours[-1]))

    frame_ks = sorted(set(np.linspace(0, n - 1, min(100, n)).round().astype(int)))
    disp_ks = np.array(sorted(set(np.linspace(0, n - 1, min(180, n)).round().astype(int))))
    disp_min, disp_hi = minutes[disp_ks], hi_arr[disp_ks]

    frames = []
    for k in frame_ks:
        mask = disp_ks <= k
        xs = list(disp_min[mask]) + [float(minutes[k])]
        ys = list(disp_hi[mask]) + [float(hi_arr[k])]
        past = bool(k >= fpt_idx)
        health_now = float(health_full.iloc[k])
        rul_now = float(rul_full[k]) if np.isfinite(rul_full[k]) else None
        adv = maintenance_advice(health_now, rul_now, past,
                                 alarm_health=alarm, safety_margin=margin)
        frames.append({
            "k": int(k), "minute": float(minutes[k]), "hours": float(hours[k]),
            "health": health_now, "rul_hours": rul_now, "past_fpt": past,
            "risk": adv.risk, "risk_label_zh": adv.risk_label_zh,
            "suggested_window_hours": adv.suggested_window_hours,
            "x": [float(v) for v in xs], "y": [float(v) for v in ys],
        })

    return {
        "available": True,
        "condition": condition, "bearing": bearing,
        "hi_base": hi_base, "hi_fail": hi_fail,
        "fpt_index": int(fpt_idx), "fpt_minute": float(minutes[fpt_idx]),
        "fpt_hi": float(hi_arr[fpt_idx]), "last_minute": float(minutes[-1]),
        "frames": frames,
    }


def maintenance_advice(
    health: float,
    rul_hours: Optional[float],
    past_fpt: bool,
    *,
    alarm_health: float = 30.0,
    safety_margin: float = 0.3,
    cost_unplanned: Optional[float] = None,
    cost_planned: Optional[float] = None,
) -> Dict[str, Any]:
    """Map current health / RUL / FPT to a risk level, window and rationale."""
    from dataclasses import asdict

    from src.models.maintenance_advice import maintenance_advice as _advice

    adv = _advice(
        health, rul_hours, past_fpt,
        alarm_health=alarm_health, safety_margin=safety_margin,
        cost_unplanned=cost_unplanned, cost_planned=cost_planned,
    )
    return asdict(adv)


# --- Module B (IMS single-trajectory health curve) ---------------------------
def ims_metrics() -> Dict[str, Any]:
    """IMS RUL metrics/meta: indicator, FPT index/time, lead time, near-failure errors."""
    return _read_json_or_empty(load_config()["ims"]["rul_metrics"])


def ims_health_curve() -> List[Dict[str, Any]]:
    """IMS health/RUL timeseries: timestamp, rul_true, rul_pred, health, is_degrading.

    Meta (FPT index, lead time) is served separately by /ims/metrics. ``to_json``
    turns the pre-degradation NaN ``rul_pred`` into valid JSON ``null``. Empty
    list if the RUL extrapolation job has not run.
    """
    path = resolve(load_config()["ims"]["rul_predictions"])
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


_IMS_INDICATOR_CANDIDATES = ("b1_rms", "b1_kurtosis", "b1_band_BPFO", "b1_crest_factor")


def ims_health_indicator(indicator: Optional[str] = None) -> Dict[str, Any]:
    """Recompute the health curve + degradation onset (FPT) for a chosen indicator.

    Mirrors the Streamlit 'switch the health indicator, recompute FPT live'
    interaction. ``available`` is False if the IMS feature table is not built."""
    from src.models.rul_extrapolation import build_health_indicator, detect_fpt

    cfg = load_config()["ims"]
    path = resolve(cfg["processed_features"])
    if not path.exists():
        return {"available": False, "reason": "IMS 特徵表尚未建立。"}
    df = pd.read_parquet(path).sort_index()
    candidates = [c for c in _IMS_INDICATOR_CANDIDATES if c in df.columns]
    ind = indicator or cfg["health_indicator"]
    if ind not in df.columns:
        raise ValueError(f"未知的 indicator：{ind!r}。可用：{candidates}")
    hi, health, hi_base, hi_fail = build_health_indicator(
        df[ind], cfg["hi_smooth_window"], cfg["baseline_n"], cfg["fail_percentile"]
    )
    fpt_idx = detect_fpt(hi, cfg["baseline_n"], cfg["fpt_n_sigma"], cfg["fpt_consecutive"])
    ts = df.index
    fpt_t = ts[fpt_idx]
    return {
        "available": True,
        "indicator": ind,
        "candidates": candidates,
        "fpt_index": int(fpt_idx),
        "fpt_time": str(fpt_t),
        "lead_time_days": (ts[-1] - fpt_t).total_seconds() / 86400.0,
        "hi_base": hi_base,
        "hi_fail": hi_fail,
        "alarm_health": float(cfg["alarm_health"]),
        "points": [
            {"timestamp": str(t), "health": float(h)}
            for t, h in zip(ts, health.to_numpy())
        ],
    }


def _downsample(arr: np.ndarray, n: int) -> np.ndarray:
    """Evenly subsample ``arr`` to at most ``n`` points (keeps endpoints)."""
    if arr.size <= n:
        return arr
    return arr[np.linspace(0, arr.size - 1, n).astype(int)]


def ims_snapshot(index: int) -> Dict[str, Any]:
    """Raw vibration waveform + FFT spectrum for one IMS snapshot.

    Needs the 1.5 GB raw IMS dataset (not shipped); returns ``available: False``
    when absent (e.g. the cloud demo). Waveform is downsampled and the spectrum
    capped to 2 kHz to keep the payload small."""
    cfg = load_config()["ims"]
    raw_dir = resolve(cfg["raw_dir"])
    if not raw_dir.exists():
        return {
            "available": False,
            "reason": "原始波形需 1.5 GB IMS 原始資料（未隨專案上傳）；雲端 demo 不提供。",
        }
    from src.data.load_ims import list_ims_files, load_ims_file

    files = list_ims_files(raw_dir)
    if not 0 <= index < len(files):
        raise ValueError(f"快照序號超出範圍 0..{len(files) - 1}")
    ts_sel, path_sel = files[index]
    ch = load_ims_file(path_sel)[:, cfg["target_bearing"] - 1]
    fs = float(cfg["sampling_rate_hz"])
    mag = np.abs(np.fft.rfft(ch))
    freqs = np.fft.rfftfreq(ch.size, d=1.0 / fs)
    keep = freqs <= 2000.0  # defect frequencies (BPFO/BPFI/BSF/FTF) are all < 300 Hz
    return {
        "available": True,
        "index": index,
        "n_snapshots": len(files),
        "timestamp": str(ts_sel),
        "target_bearing": int(cfg["target_bearing"]),
        "sampling_rate_hz": fs,
        "waveform": _downsample(ch, 2048).tolist(),
        "spectrum": {"freqs": freqs[keep].tolist(), "mags": mag[keep].tolist()},
        "defect_freqs": cfg["defect_freqs"],
    }


# --- Maintenance knowledge base (TF-IDF RAG) ---------------------------------
def knowledge_documents() -> List[Dict[str, Any]]:
    """List local KB documents (source, title, preview, chars)."""
    from src.knowledge.maintenance_rag import list_documents

    return list_documents()


def knowledge_search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """TF-IDF keyword search over the KB; each hit has text/score/source/title/topic."""
    from src.knowledge.maintenance_rag import search

    return search(query, top_k=top_k)


# --- Module Servo (main line) -------------------------------------------------
def servo_model_info() -> Dict[str, Any]:
    from src.models.servo_predict import load_servo_models

    b = load_servo_models()
    return {
        "feature_set": b.config.get("feature_set"),
        "feature_columns": b.feature_columns,
        "labels": b.config.get("labels"),
        "clf_model": b.config.get("clf_model"),
        "reg_model": b.config.get("reg_model"),
        "clf_macro_f1": b.config.get("clf_macro_f1"),
        "reg_r2": b.config.get("reg_r2"),
        "placeholder": b.config.get("placeholder"),
    }


def servo_predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    from src.models.servo_predict import load_servo_models, predict_servo

    b = load_servo_models()
    missing = [c for c in b.feature_columns if c not in features]
    if missing:
        raise ValueError(f"缺少必要特徵欄位：{missing}")
    return predict_servo(features)


def servo_glossary() -> List[Dict[str, Any]]:
    """Motor field glossary: name / zh / desc / meaning / anomaly."""
    from src.servo.field_glossary import FIELD_DOCS

    return FIELD_DOCS


def servo_feature_sets() -> Dict[str, Any]:
    """Available feature sets: key -> label / desc / columns."""
    from src.features.servo_features import FEATURE_SETS

    return FEATURE_SETS


def servo_samples() -> List[Dict[str, Any]]:
    """Demo sample rows the dashboard can select from (features + ylabel / DV)."""
    path = resolve(load_config()["servo"]["sample_predictions"])
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


# Synthetic equipment identities for the Command Center fleet view. The health
# of each unit is NOT hard-coded — it is computed by running the *real* reference
# model (servo_predict_one) over a representative demo run that matches the unit's
# intended state. So identities are demo, but the health/risk/features are real
# model output on demo data (not real PHM telemetry).
_FLEET_UNITS = [
    {"id": "servo-a01", "name": "Servo-A01", "location": "產線 1 · X 軸", "status": "running", "target": "LN", "uptime": 1840, "updated": "12 秒前"},
    {"id": "servo-a02", "name": "Servo-A02", "location": "產線 1 · Y 軸", "status": "running", "target": "LO", "uptime": 2210, "updated": "8 秒前"},
    {"id": "servo-a03", "name": "Servo-A03", "location": "產線 2 · Z 軸", "status": "warning", "target": "MED", "uptime": 3050, "updated": "5 秒前"},
    {"id": "servo-testbench", "name": "Servo-TestBench", "location": "實驗台 · 加速壽命", "status": "maintenance", "target": "HI", "uptime": 5120, "updated": "3 秒前"},
]


def servo_fleet() -> List[Dict[str, Any]]:
    """Demo fleet whose health is computed by the real reference model.

    For each synthetic unit, pick a demo run matching its target state and run it
    through servo_predict_one; surface the model's health score / state / risk /
    degradation / confidence / top anomalous feature.
    """
    rows = servo_samples()
    if not rows:
        return []
    cols = servo_model_info()["feature_columns"]

    fleet: List[Dict[str, Any]] = []
    for u in _FLEET_UNITS:
        match = next(
            (r for r in rows if str(r.get("ylabel")) == u["target"]),
            rows[len(rows) // 2],
        )
        features = {c: match[c] for c in cols if c in match}
        pred = servo_predict_one(features)
        top = (pred.get("top_features") or [{}])[0]
        fleet.append(
            {
                "id": u["id"],
                "name": u["name"],
                "location": u["location"],
                "status": u["status"],
                "healthScore": round(pred["health_score"]),
                "state": pred["predicted_health_state"],
                "risk": pred["risk_level"],
                "degradation": round(pred["degradation_score"], 2),
                "confidence": round(pred["model_confidence"], 2),
                "placeholder": bool(pred.get("placeholder", True)),
                "topFeature": {
                    "feature": top.get("feature", "-"),
                    "z": top.get("z", 0),
                    "hint": top.get("hint", ""),
                },
                "uptimeHours": u["uptime"],
                "lastUpdated": u["updated"],
            }
        )
    return fleet


# Map a unit's top anomalous feature to an alert type + suggested action.
_FEATURE_ALERT = [
    (("i_3p", "current", "direct", "quadrature"), "電流異常", "檢查繞組絕緣與供電，必要時更換軸承"),
    (("torque",), "扭矩過載", "檢查負載與滾珠導螺桿潤滑"),
    (("position_error", "rod_"), "跟隨誤差升高", "校正位置迴路增益，檢查機械背隙"),
    (("rotor_speed",), "轉速不穩", "檢查編碼器與速度迴路，確認負載穩定"),
]
_ALERT_TIMES = ["10:42", "10:31", "10:18", "09:47"]


def _alert_for_feature(feat: str) -> tuple:
    f = (feat or "").lower()
    for keys, typ, action in _FEATURE_ALERT:
        if any(k in f for k in keys):
            return typ, action
    return "特徵異常", "持續觀察並複核感測訊號"


def _servo_ops() -> tuple:
    """Derive alerts + work orders from the *real* model-driven fleet.

    Risk / state / top feature come from servo_fleet() (real model output); the
    operational wrapping (alert IDs, work-order scheduling) is demo bookkeeping.
    """
    fleet = servo_fleet()
    alerts: List[Dict[str, Any]] = []
    work_orders: List[Dict[str, Any]] = []
    wo_seq = 3307

    for i, u in enumerate(fleet):
        if u["risk"] == "Low":
            continue
        typ, action = _alert_for_feature(u["topFeature"]["feature"])
        high = u["risk"] == "High"
        wo_id = None
        if high:
            wo_id = f"WO-{wo_seq}"
            wo_seq += 1
            work_orders.append({
                "id": wo_id, "equipment": u["name"], "title": f"{typ} — 停機檢修",
                "priority": "high", "status": "in_progress",
                "assignee": "維修班 A", "due": "今日 16:00",
            })
        alerts.append({
            "id": f"ALM-{2041 - i}", "time": _ALERT_TIMES[i % len(_ALERT_TIMES)],
            "equipment": u["name"], "type": typ,
            "severity": "critical" if high else "warning",
            "predictedState": u["state"], "suggestedAction": action,
            "status": "in_progress" if high else "open", "workOrderId": wo_id,
        })

    # schedule one medium work order for the first warning alert lacking one
    for a in alerts:
        if a["severity"] == "warning" and a["workOrderId"] is None:
            wo_id = f"WO-{wo_seq}"
            wo_seq += 1
            a["workOrderId"] = wo_id
            work_orders.append({
                "id": wo_id, "equipment": a["equipment"], "title": f"{a['type']} — 保養複核",
                "priority": "medium", "status": "scheduled",
                "assignee": "維修班 B", "due": "明日 10:00",
            })
            break

    # one resolved info alert for the healthiest unit (demo variety)
    if fleet:
        best = max(fleet, key=lambda x: x["healthScore"])
        alerts.append({
            "id": "ALM-2030", "time": "09:31", "equipment": best["name"],
            "type": "溫升提醒", "severity": "info", "predictedState": best["state"],
            "suggestedAction": "持續觀察，確認散熱與環境溫度",
            "status": "resolved", "workOrderId": None,
        })
    return alerts, work_orders


def servo_alerts() -> List[Dict[str, Any]]:
    """Fleet alerts derived from the real model-driven fleet."""
    return _servo_ops()[0]


def servo_work_orders() -> List[Dict[str, Any]]:
    """Work orders derived from fleet alerts."""
    return _servo_ops()[1]


def servo_reference_metrics() -> Dict[str, Any]:
    """Reference baselines for the training simulator: clf / reg / dl."""
    cfg = load_config()["servo"]
    return {
        "clf": _read_json_or_empty(cfg["clf_metrics"]),
        "reg": _read_json_or_empty(cfg["reg_metrics"]),
        "dl": _read_json_or_empty(cfg["dl_metrics"]),
    }


def servo_simulate_options() -> Dict[str, Any]:
    """Algorithm choices for the training simulator (clf / reg names + labels)."""
    from src.models import servo_simulator as sim

    return {
        "classifiers": list(sim.CLASSIFIER_NAMES),
        "regressors": list(sim.REGRESSOR_NAMES),
        "algo_labels": sim.ALGO_LABELS,
    }


def servo_simulate(task: str, feature_set: str, algo: str, n: int) -> Dict[str, Any]:
    """Train a small clf/reg model in-process and return its metrics + notes.

    Fast on the demo feature table (<0.4s), so this runs synchronously."""
    from src.features.servo_features import FEATURE_SETS
    from src.models import servo_simulator as sim

    path = resolve(load_config()["servo"]["processed_features"])
    if not path.exists():
        raise FileNotFoundError("伺服特徵表尚未建立。")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知的 feature_set：{feature_set!r}。可用：{list(FEATURE_SETS)}")
    valid_algos = sim.CLASSIFIER_NAMES if task == "clf" else sim.REGRESSOR_NAMES
    if algo not in valid_algos:
        raise ValueError(f"未知的 algo：{algo!r}。可用：{list(valid_algos)}")

    df = pd.read_parquet(path)
    n = min(n, len(df))
    runner = sim.run_classification if task == "clf" else sim.run_regression
    res = runner(df, feature_set, algo, n)
    res["explanation"] = sim.explain_result(
        task, FEATURE_SETS[feature_set]["label"], res["n_samples"]
    )
    return res


# --- LLM maintenance assistant (server-side keys + RAG; stateless context) ----
def assistant_providers() -> Dict[str, Any]:
    """Configured LLM providers tried in order (empty -> deterministic fallback)."""
    from src.llm.maintenance_assistant import available_providers

    return {"providers": available_providers()}


def assistant_report(prediction: Dict[str, Any]) -> Dict[str, str]:
    """Full maintenance write-up grounded in the prediction + retrieved KB chunks."""
    from src.knowledge.maintenance_rag import retrieve_for_prediction
    from src.llm.maintenance_assistant import generate_report

    chunks = retrieve_for_prediction(prediction)
    return generate_report(prediction, chunks)


def assistant_qa(question: str, prediction: Dict[str, Any]) -> Dict[str, str]:
    """Conservative Q&A grounded in the prediction + retrieved KB chunks."""
    from src.knowledge.maintenance_rag import retrieve_for_prediction
    from src.llm.maintenance_assistant import answer_question

    chunks = retrieve_for_prediction(prediction)
    return answer_question(question, prediction, chunks)
