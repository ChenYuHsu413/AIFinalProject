"""S3 — Module Servo model VERSION registry (Windows-friendly, no symlinks).

Named ``servo_model_registry`` to avoid colliding with ``model_registry`` — that
is the Module-A ALGORITHM FACTORY (name -> estimator constructor), a different
concept. This module manages DEPLOYED VERSIONS (v1 / v2 / …) of the servo
reference model.

Deployment source of truth: inference ALWAYS loads through this registry
(``load_active``). ``outputs/models/servo_*.joblib`` is only ``train_servo``'s
default working output — a stale copy there does not affect what is served, and
the backend logs a warning if it diverges from the active version (see
``outputs_models_consistent_with_active``).

Layout::

    models/registry/
      registry.json          # {active_version, versions:{v1:{summary}, ...}}
      v1/  servo_clf.joblib  servo_reg.joblib  servo_feature_config.json  metrics.json
      v2/  ...
      candidate_<ts>/        # transient (retrain pipeline output; gitignored)

The registry is the single source of truth for which reference model is live.
``load_active()`` / ``load_version(v)`` centralise loading so ``predict_servo``
and the FastAPI backend read exactly one place; ``active_version()`` is what the
alert engine stamps onto events. The active version is switched by editing the
``active_version`` field in ``registry.json`` (no symlink), so rollback is a
one-line change.

Per-version ``metrics.json`` records the evidence a deploy decision needs:
held-out macro-F1 / DV R², training-data summary, feature set, model-file CRC32
(integrity), a training-time config snapshot, the FMCRD data-provenance
fingerprint, and a timestamp.
"""
from __future__ import annotations

import json
import shutil
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.utils.paths import load_config, resolve

_REGISTRY = "models/registry"
ARTIFACT_FILES = ("servo_clf.joblib", "servo_reg.joblib", "servo_feature_config.json")
CRC_FILES = ("servo_clf.joblib", "servo_reg.joblib")


# --- paths ------------------------------------------------------------------
def registry_dir() -> Path:
    return resolve(_REGISTRY)


def registry_json() -> Path:
    return registry_dir() / "registry.json"


def version_dir(version: str) -> Path:
    return registry_dir() / version


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- registry.json ----------------------------------------------------------
def read_registry() -> Dict[str, Any]:
    p = registry_json()
    if not p.exists():
        raise FileNotFoundError(
            f"找不到模型登記表：{p}\n請先執行遷移：python -m src.models.servo_model_registry --migrate")
    return json.loads(p.read_text(encoding="utf-8"))


def write_registry(reg: Dict[str, Any]) -> None:
    registry_dir().mkdir(parents=True, exist_ok=True)
    registry_json().write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                               encoding="utf-8")


def active_version(default: Optional[str] = None) -> str:
    """The live version. ``default`` is returned if the registry is absent
    (lets callers like the alert engine degrade gracefully pre-migration)."""
    if not registry_json().exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"找不到模型登記表：{registry_json()}")
    return str(read_registry().get("active_version", default or "unknown"))


def _vnum(v: str) -> int:
    try:
        return int(v[1:]) if v.startswith("v") else 0
    except ValueError:
        return 0


def list_versions() -> List[str]:
    return sorted(read_registry().get("versions", {}), key=_vnum)


def next_version_name() -> str:
    versions = read_registry().get("versions", {}) if registry_json().exists() else {}
    n = max((_vnum(v) for v in versions), default=0)
    return f"v{n + 1}"


def set_active(version: str) -> None:
    reg = read_registry()
    if version not in reg.get("versions", {}):
        raise ValueError(f"版本 {version} 不在登記表（有：{list(reg.get('versions', {}))}）。")
    reg["active_version"] = version
    reg["updated"] = _now()
    write_registry(reg)


# --- integrity --------------------------------------------------------------
def crc32_file(path: Path) -> str:
    h = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h = zlib.crc32(chunk, h)
    return f"{h & 0xFFFFFFFF:08x}"


def crc32_for_dir(model_dir: Path) -> Dict[str, str]:
    return {name: crc32_file(model_dir / name) for name in CRC_FILES}


def outputs_models_consistent_with_active() -> Optional[bool]:
    """Whether ``outputs/models/servo_*.joblib`` matches the ACTIVE version's CRC32.

    Guards against a "two sources of truth" drift: ``outputs/models`` is only
    ``train_servo``'s working output (deployment reads the registry), so a
    mismatch means that copy is stale — worth a warning, not a hard failure.
    Returns None if either side is missing.
    """
    try:
        sv = load_config()["servo"]
        om = {"servo_clf.joblib": resolve(sv["clf_model"]),
              "servo_reg.joblib": resolve(sv["reg_model"])}
        if not all(p.exists() for p in om.values()):
            return None
        active = version_dir(active_version())
        return all(crc32_file(om[n]) == crc32_file(active / n) for n in CRC_FILES)
    except Exception:
        return None


# --- loading (single source of truth for inference) -------------------------
@dataclass
class LoadedModel:
    version: str
    clf_pipeline: Any
    reg_pipeline: Any
    feature_config: Dict[str, Any]
    metrics: Dict[str, Any]


def load_dir(model_dir: Path) -> LoadedModel:
    """Load a model from ANY directory (a registered version OR a candidate)."""
    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目錄：{model_dir}")
    clf = joblib.load(model_dir / "servo_clf.joblib")
    reg = joblib.load(model_dir / "servo_reg.joblib")
    fc = json.loads((model_dir / "servo_feature_config.json").read_text(encoding="utf-8"))
    mp = model_dir / "metrics.json"
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    version = metrics.get("version") or model_dir.name
    return LoadedModel(version, clf["pipeline"], reg["pipeline"], fc, metrics)


def load_version(version: str) -> LoadedModel:
    return load_dir(version_dir(version))


def load_active() -> LoadedModel:
    return load_version(active_version())


# --- metrics.json assembly (shared by migration + retrain pipeline) ---------
def _data_provenance_fingerprint() -> Optional[Dict[str, Any]]:
    p = resolve("outputs/metrics/servo_data_provenance.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("source")


def assemble_metrics(version: str, model_dir: Path, feature_config: Dict[str, Any],
                     clf_eval: Dict[str, Any], reg_eval: Dict[str, Any],
                     config_snapshot: Dict[str, Any],
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a version's metrics.json (evidence for the deploy decision)."""
    m = {
        "version": version,
        "created": _now(),
        "feature_set": feature_config.get("feature_set"),
        "feature_columns": feature_config.get("feature_columns"),
        "clf_model": feature_config.get("clf_model") or clf_eval.get("model"),
        "reg_model": feature_config.get("reg_model") or reg_eval.get("model"),
        "macro_f1": clf_eval.get("macro_f1"),
        "dv_r2": reg_eval.get("r2"),
        "dv_mae": reg_eval.get("mae"),
        "eval_mode": clf_eval.get("eval"),
        "placeholder": bool(clf_eval.get("placeholder", feature_config.get("placeholder", True))),
        "training_data": {
            "n_eval_clf": clf_eval.get("n"),
            "n_eval_reg": reg_eval.get("n"),
            "labels": clf_eval.get("labels"),
            "confusion_matrix": clf_eval.get("confusion_matrix"),
            "per_model_macro_f1": clf_eval.get("per_model_macro_f1"),
        },
        "model_crc32": crc32_for_dir(model_dir),
        "config_snapshot": config_snapshot,
        "data_provenance": _data_provenance_fingerprint(),
    }
    if extra:
        m.update(extra)
    return m


def registry_summary(metrics: Dict[str, Any], note: str = "") -> Dict[str, Any]:
    """The short per-version entry stored in registry.json."""
    return {
        "created": metrics.get("created"),
        "macro_f1": metrics.get("macro_f1"),
        "dv_r2": metrics.get("dv_r2"),
        "feature_set": metrics.get("feature_set"),
        "placeholder": metrics.get("placeholder"),
        "note": note,
    }


def register_version(model_dir: Path, metrics: Dict[str, Any], note: str = "",
                     make_active: bool = True) -> str:
    """Record ``model_dir`` (already named v<n>) in registry.json."""
    version = metrics["version"]
    reg = read_registry() if registry_json().exists() else {"versions": {}}
    reg.setdefault("versions", {})[version] = registry_summary(metrics, note)
    if make_active:
        reg["active_version"] = version
    reg["updated"] = _now()
    write_registry(reg)
    return version


def promote_candidate(candidate_dir: Path, metrics: Dict[str, Any],
                      note: str = "") -> str:
    """Rename a passing candidate dir to the next v<n>, register + activate it."""
    version = next_version_name()
    metrics = {**metrics, "version": version}
    target = version_dir(version)
    if target.exists():
        raise FileExistsError(f"版本目錄已存在：{target}")
    shutil.move(str(candidate_dir), str(target))
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    register_version(target, metrics, note=note, make_active=True)
    return version


# --- migration: register the currently-deployed model as v1 -----------------
def migrate_v1(note: str = "migrated from outputs/models (S3 initial deploy)") -> str:
    """Copy the config-path artifacts into models/registry/v1 and write registry.json.

    Idempotent: if the registry already exists, returns the active version.
    """
    if registry_json().exists():
        return active_version()
    sv = load_config()["servo"]
    d = version_dir("v1")
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(sv["clf_model"]), d / "servo_clf.joblib")
    shutil.copy(resolve(sv["reg_model"]), d / "servo_reg.joblib")
    shutil.copy(resolve(sv["feature_config"]), d / "servo_feature_config.json")

    fc = json.loads((d / "servo_feature_config.json").read_text(encoding="utf-8"))
    clf_eval = json.loads(resolve(sv["clf_metrics"]).read_text(encoding="utf-8"))
    reg_eval = json.loads(resolve(sv["reg_metrics"]).read_text(encoding="utf-8"))
    metrics = assemble_metrics("v1", d, fc, clf_eval, reg_eval, config_snapshot=dict(sv))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    write_registry({"active_version": "v1", "updated": _now(),
                    "versions": {"v1": registry_summary(metrics, note)}})
    return "v1"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Servo model version registry")
    ap.add_argument("--migrate", action="store_true", help="register current model as v1")
    ap.add_argument("--list", action="store_true", help="list versions + active")
    args = ap.parse_args()
    if args.migrate:
        v = migrate_v1()
        print(f"[registry] active={v}  dir={version_dir(v)}")
    if args.list or not args.migrate:
        reg = read_registry()
        print(f"active_version: {reg['active_version']}")
        for ver in list_versions():
            s = reg["versions"][ver]
            print(f"  {ver}: macro_f1={s.get('macro_f1')} dv_r2={s.get('dv_r2')} "
                  f"{'(active)' if ver == reg['active_version'] else ''}")
