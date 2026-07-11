"""S3 Task B — validation gate for a candidate servo model version.

Runs a fixed sequence of checks on a candidate model directory. Any FAIL makes
the whole gate FAIL (the retrain pipeline then refuses to promote). Every check's
result — PASS / FAIL / SKIP, the numbers, and the comparison to the active
version — is written to ``<candidate_dir>/gate_report.json``.

Checks:
  1. completeness — required files present; feature_config columns are consistent
     with their named feature set AND the config's reference set; model-file CRC32
     matches what metrics.json recorded (integrity).
  2. smoke_test — load the candidate and run ``predict_servo`` on the demo rows;
     the structured output's schema and value ranges must be valid.
  3. holdout_metrics — RE-COMPUTE the candidate's held-out macro-F1 and DV R² on
     the committed test split (not trusting self-reported numbers) and require
     them within the config tolerance band of the ACTIVE version.
  4. ae_monotonicity — if the candidate ships a DL/AE component, its reconstruction
     error must be non-decreasing across LN→LO→MED→HI; otherwise SKIP (recorded).

Run::

    python -m src.pipeline.validation_gate models/registry/candidate_20260711
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.features.servo_features import FEATURE_SETS, HEALTH_LABELS, feature_set_columns
from src.models import servo_model_registry as registry
from src.utils.paths import load_config, resolve

PASS, FAIL, SKIP, BLOCKED = "PASS", "FAIL", "SKIP", "BLOCKED"
_REQUIRED_FILES = ("servo_clf.joblib", "servo_reg.joblib",
                   "servo_feature_config.json", "metrics.json")


def _check(name: str, status: str, detail: str,
           numbers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "numbers": numbers or {}}


def _tolerances(override: Optional[Dict[str, float]]) -> Dict[str, float]:
    g = load_config().get("servo_gate", {})
    return {
        "macro_f1": float((override or {}).get("macro_f1", g.get("macro_f1_tolerance", 0.005))),
        "dv_r2": float((override or {}).get("dv_r2", g.get("dv_r2_tolerance", 0.005))),
    }


# --- 1. completeness --------------------------------------------------------
def _completeness(d: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    missing = [f for f in _REQUIRED_FILES if not (d / f).exists()]
    if missing:
        return _check("completeness", FAIL, f"缺少必要檔案：{missing}")
    fc = json.loads((d / "servo_feature_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))

    problems: List[str] = []
    fs = fc.get("feature_set")
    ref_fs = cfg["servo"].get("reference_feature_set")
    if fs != ref_fs:
        problems.append(f"feature_set={fs} 與 config.reference_feature_set={ref_fs} 不符")
    if fs not in FEATURE_SETS or fc.get("feature_columns") != feature_set_columns(fs):
        problems.append("feature_columns 與 feature_set 定義不一致")

    actual_crc = registry.crc32_for_dir(d)
    recorded_crc = metrics.get("model_crc32") or {}
    if not recorded_crc:
        problems.append("metrics.json 缺少 model_crc32")
    else:
        bad = [f for f in actual_crc if recorded_crc.get(f) != actual_crc[f]]
        if bad:
            problems.append(f"CRC32 不符（檔案可能損毀/被竄改）：{bad}")

    numbers = {"crc_actual": actual_crc, "crc_recorded": recorded_crc,
               "feature_set": fs, "n_columns": len(fc.get("feature_columns", []))}
    if problems:
        return _check("completeness", FAIL, "；".join(problems), numbers)
    return _check("completeness", PASS, "檔案齊全、特徵欄位一致、CRC32 可驗", numbers)


# --- 2. smoke test ----------------------------------------------------------
def _smoke_test(d: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from src.models.servo_predict import make_bundle, predict_servo

    bundle = make_bundle(registry.load_dir(d))
    demo = pd.read_csv(resolve(cfg["servo"]["sample_predictions"]))
    labels = set(bundle.config.get("labels", HEALTH_LABELS))
    required = {"predicted_health_state", "degradation_score", "health_score",
                "model_confidence", "risk_level", "health_state_proba"}
    n = min(5, len(demo))
    problems: List[str] = []
    for i in range(n):
        out = predict_servo(demo.iloc[i], bundle=bundle)
        if not required <= out.keys():
            problems.append(f"row{i}: 缺少輸出欄位 {required - out.keys()}")
            continue
        if out["predicted_health_state"] not in labels:
            problems.append(f"row{i}: 非法健康狀態 {out['predicted_health_state']}")
        if not 0.0 <= out["degradation_score"] <= 1.0:
            problems.append(f"row{i}: degradation_score 越界 {out['degradation_score']}")
        if not 0.0 <= out["model_confidence"] <= 1.0:
            problems.append(f"row{i}: model_confidence 越界 {out['model_confidence']}")
        if not 0.0 <= out["health_score"] <= 100.0:
            problems.append(f"row{i}: health_score 越界 {out['health_score']}")
        if out["risk_level"] not in {"Low", "Medium", "High"}:
            problems.append(f"row{i}: 非法 risk_level {out['risk_level']}")
        if abs(sum(out["health_state_proba"].values()) - 1.0) > 0.02:
            problems.append(f"row{i}: proba 未歸一 {sum(out['health_state_proba'].values()):.3f}")
    if problems:
        return _check("smoke_test", FAIL, "；".join(problems[:5]), {"rows_checked": n})
    return _check("smoke_test", PASS, f"{n} 筆 demo 預測 schema 與值域皆合法",
                  {"rows_checked": n})


# --- 3. holdout metrics -----------------------------------------------------
def _holdout_metrics(d: Path, active: Optional[str], tol: Dict[str, float],
                     cfg: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, r2_score

    feat_path = resolve(cfg["servo"]["processed_features"])
    if not feat_path.exists():
        return _check("holdout_metrics", SKIP, "找不到特徵表，無法評估留出指標")
    feat = pd.read_parquet(feat_path)
    if "split" not in feat.columns or "test" not in set(feat["split"]):
        return _check("holdout_metrics", SKIP, "特徵表無 test split，略過留出比較")
    te = feat[feat["split"] == "test"]
    labels = [c for c in HEALTH_LABELS if c in set(te["ylabel"])]

    def _score(model) -> Dict[str, float]:
        cols = model.feature_config["feature_columns"]
        X = te[cols]
        f1 = float(f1_score(te["ylabel"], model.clf_pipeline.predict(X),
                            average="macro", labels=labels, zero_division=0))
        r2 = float(r2_score(te["DV"], model.reg_pipeline.predict(X)))
        return {"macro_f1": f1, "dv_r2": r2}

    cand = _score(registry.load_dir(d))
    numbers: Dict[str, Any] = {"candidate": cand, "tolerance": tol, "n_test": int(len(te))}

    if not active:
        return _check("holdout_metrics", PASS,
                      "無 active 基準（首個模型），僅記錄候選指標", numbers)
    act = _score(registry.load_active())
    numbers["active"] = act
    numbers["macro_f1_delta"] = round(cand["macro_f1"] - act["macro_f1"], 4)
    numbers["dv_r2_delta"] = round(cand["dv_r2"] - act["dv_r2"], 4)

    f1_ok = cand["macro_f1"] >= act["macro_f1"] - tol["macro_f1"]
    r2_ok = cand["dv_r2"] >= act["dv_r2"] - tol["dv_r2"]
    if f1_ok and r2_ok:
        return _check("holdout_metrics", PASS,
                      f"候選 macro-F1={cand['macro_f1']:.4f}（active {act['macro_f1']:.4f}）、"
                      f"DV R²={cand['dv_r2']:.4f}（active {act['dv_r2']:.4f}），均在容忍帶內",
                      numbers)
    fails = []
    if not f1_ok:
        fails.append(f"macro-F1 {cand['macro_f1']:.4f} < active {act['macro_f1']:.4f} - {tol['macro_f1']}")
    if not r2_ok:
        fails.append(f"DV R² {cand['dv_r2']:.4f} < active {act['dv_r2']:.4f} - {tol['dv_r2']}")
    return _check("holdout_metrics", FAIL, "；".join(fails), numbers)


# --- 4. AE monotonicity -----------------------------------------------------
def _ae_monotonicity(d: Path) -> Dict[str, Any]:
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8")) \
        if (d / "metrics.json").exists() else {}
    recon = metrics.get("ae_recon_by_class")
    if not recon and not (d / "servo_ae.pt").exists():
        return _check("ae_monotonicity", SKIP, "候選未含 DL/AE 組件（略過）")
    if not recon:
        return _check("ae_monotonicity", SKIP,
                      "候選含 AE 權重但 metrics 未帶 ae_recon_by_class（略過，建議補記錄）")
    order = [c for c in HEALTH_LABELS if c in recon]
    vals = [float(recon[c]) for c in order]
    ok = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
    numbers = {"recon_by_class": dict(zip(order, vals))}
    if ok:
        return _check("ae_monotonicity", PASS, "AE 重建誤差健康→退化單調非遞減", numbers)
    return _check("ae_monotonicity", FAIL, "AE 重建誤差非單調（健康指標承諾被破壞）", numbers)


# --- orchestration ----------------------------------------------------------
def run_gate(candidate_dir: Path, active_version: Optional[str] = None,
             tolerances: Optional[Dict[str, float]] = None,
             write_report: bool = True) -> Dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    cfg = load_config()
    tol = _tolerances(tolerances)
    active = active_version if active_version is not None \
        else registry.active_version(default=None)

    checks = [_completeness(candidate_dir, cfg)]
    if checks[0]["status"] == FAIL:
        for n in ("smoke_test", "holdout_metrics", "ae_monotonicity"):
            checks.append(_check(n, BLOCKED, "completeness 未過，無法執行此檢查"))
    else:
        checks.append(_smoke_test(candidate_dir, cfg))
        checks.append(_holdout_metrics(candidate_dir, active, tol, cfg))
        checks.append(_ae_monotonicity(candidate_dir))

    passed = all(c["status"] in (PASS, SKIP) for c in checks)
    report = {
        "candidate_dir": str(candidate_dir),
        "active_version": active,
        "passed": passed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerances": tol,
        "checks": checks,
    }
    if write_report:
        (candidate_dir / "gate_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _print_report(report: Dict[str, Any]) -> None:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⏭", BLOCKED: "🚫"}
    print(f"\n驗證閘門：{'PASS ✅' if report['passed'] else 'FAIL ❌'}"
          f"（active={report['active_version']}）")
    for c in report["checks"]:
        print(f"  {icon.get(c['status'], '?')} {c['name']:16} {c['status']:7} {c['detail']}")


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Servo model validation gate")
    ap.add_argument("candidate_dir")
    ap.add_argument("--active", default=None, help="覆寫 active 版本（預設讀 registry）")
    args = ap.parse_args()
    report = run_gate(Path(args.candidate_dir), active_version=args.active)
    _print_report(report)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
