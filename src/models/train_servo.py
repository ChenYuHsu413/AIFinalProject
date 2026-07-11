"""Train the Module Servo reference models (offline).

Produces the artifacts the server loads for inference (it never trains large
models online):
  * ``servo.clf_model``      — health-state classifier bundle
  * ``servo.reg_model``      — DV regressor bundle
  * ``servo.feature_config`` — feature set / columns / label map / healthy
                               baseline stats / DV risk bands / metrics
  * ``servo.clf_metrics`` / ``servo.reg_metrics`` — eval JSON for the dashboard

The classifier is chosen by stratified-CV macro-F1 across ``servo.enabled_models``;
the regressor uses Random Forest (robust default for the DV target).

Run (after ``python -m src.data.build_servo_dataset``)::

    python -m src.models.train_servo
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.models.servo_simulator import build_classifier, build_regressor
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import ensure_output_dirs, load_config, resolve


def _healthy_baseline(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-feature mean/std over healthy (LN) runs -> z-score for top features."""
    ln = df[df["ylabel"] == "LN"]
    base = {}
    for c in cols:
        mu = float(ln[c].mean()) if len(ln) else float(df[c].mean())
        sd = float(ln[c].std()) if len(ln) else float(df[c].std())
        base[c] = {"mean": mu, "std": sd if sd > 1e-9 else 1.0}
    return base


def run(out_dir: "Path | None" = None,
        data_config: "Dict | None" = None) -> Path:
    """Train the reference models.

    Default (``out_dir=None``): writes to the config paths exactly as before.
    ``out_dir`` (S3 retrain pipeline): writes servo_clf/reg.joblib +
    servo_feature_config.json + metrics.json into that directory instead, and
    NEVER touches the deployed/active artifacts.
    ``data_config`` (optional): subset the TRAIN rows, e.g. ``{"train_frac":0.1}``
    — used to synthesise a deliberately-degraded candidate (and, later, S4 drift
    scenarios). ``None`` = full training data (unchanged behaviour).
    """
    ensure_output_dirs()
    cfg = load_config()
    sv = cfg["servo"]
    rs = int(sv.get("random_state", 42))

    feat_path = resolve(sv["processed_features"])
    if not feat_path.exists():
        raise FileNotFoundError(
            f"找不到 Servo 特徵表：{feat_path}\n請先執行 python -m src.data.build_servo_dataset。"
        )
    df = pd.read_parquet(feat_path)
    feature_set = sv.get("reference_feature_set", "engineered")
    cols = feature_set_columns(feature_set)

    # Honour a provided train/test split when present (the real PHM data ships
    # one); otherwise cross-validate on the whole table (placeholder fallback).
    has_split = "split" in df.columns and {"train", "test"} <= set(df["split"])
    df_tr = df[df["split"] == "train"].reset_index(drop=True) if has_split else df
    df_te = df[df["split"] == "test"].reset_index(drop=True) if has_split else None
    eval_mode = "holdout_test" if has_split else "cv"

    # Optional TRAIN subsetting (retrain-pipeline / S4 drift). The TEST split is
    # never touched. `exclude_ylabel` drops whole classes (S4 v1-lite excludes the
    # noisy-LO class); `train_frac` stratified-subsamples what remains.
    if data_config and data_config.get("exclude_ylabel"):
        excl = set(data_config["exclude_ylabel"])
        df_tr = df_tr[~df_tr["ylabel"].isin(excl)].reset_index(drop=True)
        print(f"[Servo] data_config: exclude_ylabel={sorted(excl)} -> 訓練剩 {len(df_tr)} 段")
    if data_config and data_config.get("train_frac") is not None:
        frac = float(data_config["train_frac"])
        seed = int(data_config.get("seed", rs))
        df_tr = (df_tr.groupby("ylabel", group_keys=False)
                 .sample(frac=frac, random_state=seed).reset_index(drop=True))
        print(f"[Servo] data_config: train_frac={frac} -> 訓練縮減為 {len(df_tr)} 段")
    if data_config and data_config.get("inject_drift"):
        # S4 closed loop: fold the drifted operating condition into TRAIN so the new
        # version DIGESTS it (its drift AE then treats that condition as in-distribution).
        from src.monitor.drift_detector import inject_sensor_drift_features
        inj = data_config["inject_drift"]
        gain, frac = float(inj.get("gain", 1.3)), float(inj.get("frac", 1.0))
        sample = df_tr.sample(frac=frac, random_state=rs) if frac < 1.0 else df_tr
        df_tr = pd.concat([df_tr, inject_sensor_drift_features(sample, gain)],
                          ignore_index=True)
        print(f"[Servo] data_config: inject_drift gain={gain} -> 併入漂移工況，訓練 {len(df_tr)} 段")

    labels = [c for c in HEALTH_LABELS if c in set(df_tr["ylabel"])]
    print(f"[Servo] 特徵組 {feature_set}（{len(cols)} 維）、訓練 {len(df_tr)} 段"
          + (f"、留出測試 {len(df_te)} 段" if has_split else "")
          + f"、類別 {labels}（評估={eval_mode}）")

    # --- classifier: pick best by stratified-CV macro-F1 on TRAIN ---
    X, y = df_tr[cols], df_tr["ylabel"]
    min_class = int(y.value_counts().min())
    if min_class < 2:
        rare = sorted(y.value_counts()[lambda s: s < 2].index.tolist())
        raise ValueError(
            f"類別 {rare} 僅有 1 段，無法做分層交叉驗證。"
            "請確認真實資料每個健康類別至少有 2 段，或調整 ylabel_map / 聚合粒度。"
        )
    n_splits = max(2, min(int(sv.get("cv_folds", 5)), min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    per_model: Dict[str, float] = {}
    best_name, best_macro = None, -np.inf
    for name in sv.get("enabled_models", ["random_forest"]):
        try:
            scores = cross_val_score(
                build_classifier(name, rs), X, y, cv=skf,
                scoring="f1_macro", n_jobs=-1)
        except Exception as e:  # pragma: no cover
            print(f"    {name:>20}: 跳過（{type(e).__name__}）")
            continue
        macro = float(scores.mean())
        per_model[name] = macro
        print(f"    {name:>20}: CV macro-F1={macro:.3f}")
        if macro > best_macro:
            best_name, best_macro = name, macro
    if best_name is None:
        raise RuntimeError("沒有可用的分類器。")

    # Final model trained on TRAIN; honest metrics from the held-out TEST split
    # (or CV on the whole table when there is no split).
    clf_pipe = build_classifier(best_name, rs).fit(X, y)
    if has_split:
        y_true, y_pred = df_te["ylabel"], clf_pipe.predict(df_te[cols])
    else:
        y_true = y
        y_pred = cross_val_predict(build_classifier(best_name, rs), X, y, cv=skf)
    clf_eval = {
        "model": best_name,
        "feature_set": feature_set,
        "eval": eval_mode,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_model_macro_f1": per_model,
        "n": int(len(y_true)),
        "placeholder": bool(sv.get("placeholder", True)),
    }

    # --- regressor for DV (held-out test, or CV when no split) ---
    reg_name = "random_forest"
    yv = df_tr["DV"]
    reg_pipe = build_regressor(reg_name, rs).fit(X, yv)
    if has_split:
        dv_true, dv_pred = df_te["DV"].to_numpy(), reg_pipe.predict(df_te[cols])
    else:
        from sklearn.model_selection import KFold, cross_val_predict as _cvp
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        dv_true = yv.to_numpy()
        dv_pred = _cvp(build_regressor(reg_name, rs), X, yv, cv=kf)
    reg_eval = {
        "model": reg_name,
        "feature_set": feature_set,
        "eval": eval_mode,
        "mae": float(mean_absolute_error(dv_true, dv_pred)),
        "rmse": float(np.sqrt(np.mean((dv_true - dv_pred) ** 2))),
        "r2": float(r2_score(dv_true, dv_pred)),
        "n": int(len(dv_true)),
        "placeholder": bool(sv.get("placeholder", True)),
    }
    print(f"    回歸 {reg_name}: MAE={reg_eval['mae']:.3f} R2={reg_eval['r2']:.3f}（{eval_mode}）")

    # --- artifacts (compress=3: keeps the RF regressor well under GitHub's
    #     50 MB file recommendation; ~96 MB -> ~29 MB) ---
    clf_bundle = {"pipeline": clf_pipe, "feature_columns": cols, "labels": labels,
                  "model_name": best_name, "metrics": clf_eval}
    reg_bundle = {"pipeline": reg_pipe, "feature_columns": cols,
                  "model_name": reg_name, "metrics": reg_eval}
    feature_config = {
        "feature_set": feature_set,
        "feature_columns": cols,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "dv_risk": sv.get("dv_risk", {"low_max": 0.33, "medium_max": 0.66}),
        "healthy_baseline": _healthy_baseline(df_tr, cols),
        "clf_model": best_name,
        "reg_model": reg_name,
        "clf_macro_f1": clf_eval["macro_f1"],
        "reg_r2": reg_eval["r2"],
        "eval_mode": eval_mode,
        "placeholder": bool(sv.get("placeholder", True)),
    }

    if out_dir is not None:
        # Retrain-pipeline path: write a self-contained version dir (never touches
        # the deployed/active artifacts) + a registry-style metrics.json.
        from src.models.servo_model_registry import assemble_metrics
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_bundle, out_dir / "servo_clf.joblib", compress=3)
        joblib.dump(reg_bundle, out_dir / "servo_reg.joblib", compress=3)
        (out_dir / "servo_feature_config.json").write_text(
            json.dumps(feature_config, indent=2, ensure_ascii=False), encoding="utf-8")
        extra = {"n_train": int(len(df_tr))}
        if data_config:
            extra["data_config"] = data_config
        metrics = assemble_metrics("candidate", out_dir, feature_config,
                                   clf_eval, reg_eval, config_snapshot=dict(sv),
                                   extra=extra)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        # S4: fit this version's drift baseline on ITS OWN training rows, so the
        # drift AE follows the version (in-distribution == what this model trained on).
        from src.monitor.drift_detector import build_drift_baseline, save_drift_baseline
        ae_fs = sv.get("dl_ae_feature_set", "engineered")
        scaler, pca, baseline = build_drift_baseline(df_tr, "candidate", ae_feature_set=ae_fs)
        save_drift_baseline(out_dir, scaler, pca, baseline)
        print(f"[Servo] 候選模型 -> {out_dir}（macro-F1={clf_eval['macro_f1']:.3f} "
              f"R²={reg_eval['r2']:.3f}；drift P95={baseline['recon_error_p95']:.4f}）")
        return out_dir

    joblib.dump(clf_bundle, resolve(sv["clf_model"]), compress=3)
    joblib.dump(reg_bundle, resolve(sv["reg_model"]), compress=3)
    resolve(sv["feature_config"]).write_text(
        json.dumps(feature_config, indent=2, ensure_ascii=False), encoding="utf-8")
    resolve(sv["clf_metrics"]).write_text(
        json.dumps(clf_eval, indent=2, ensure_ascii=False), encoding="utf-8")
    resolve(sv["reg_metrics"]).write_text(
        json.dumps(reg_eval, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Servo] 最佳分類器 {best_name}（macro-F1={clf_eval['macro_f1']:.3f}）"
          f"；模型/設定已寫入 outputs/。")
    if sv.get("placeholder", True):
        print("    [!] 以 placeholder 合成資料訓練；下載真實 PHM 後請重訓並設 placeholder=false。")
    return resolve(sv["clf_model"])


if __name__ == "__main__":
    run()
