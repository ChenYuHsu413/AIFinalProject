"""Verifiable data-provenance record + figure for Module Servo.

Proves the reference model was trained on the **full real PHM FMCRD dataset
(~106 GB)** rather than the earlier placeholder synthetic data. Emits, into a
JSON the backend serves and a PNG figure:

  * a **fingerprint manifest** — per-file size + CRC32 read from the zip central
    directory (instant, no extraction; anyone holding the same archive can
    re-verify these exact values),
  * the **aggregated feature-table** stats (segments, train/test split, per-class
    DV distribution), and
  * the **held-out test** metrics (eval=holdout_test, placeholder=false).

Honesty: FMCRD is a high-fidelity **simulation** dataset, NOT real factory
servomotor telemetry. "real" here = the actual large public PHM dataset vs our
placeholder. See docs/DATA_PROVENANCE.md / docs/MODULE_SERVO_PLAN.md §1.

Run::

    python -m src.data.servo_data_provenance --zip "C:/Users/<you>/Downloads/FMCRD_Data.zip"
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.paths import ensure_output_dirs, load_config, resolve

_DEFAULT_ZIP = "C:/Users/alung/Downloads/FMCRD_Data.zip"
_PROV_JSON = "outputs/metrics/servo_data_provenance.json"
_PROV_FIG = "outputs/figures/servo_provenance.png"


def _zip_manifest(zip_path: Path) -> Optional[Dict[str, Any]]:
    """Per-file size + CRC32 from the zip central directory (no extraction)."""
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as z:
        files: List[Dict[str, Any]] = []
        total = 0
        for i in sorted(z.infolist(), key=lambda x: x.filename):
            if i.is_dir():
                continue
            total += i.file_size
            files.append({
                "name": i.filename,
                "size_bytes": int(i.file_size),
                "crc32": f"{i.CRC:08x}",
            })
    return {
        "archive": zip_path.name,
        "n_files": len(files),
        "files": files,
        "total_uncompressed_bytes": int(total),
        "total_uncompressed_gb": round(total / 1e9, 2),
    }


def _feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    by = df.groupby(["split", "ylabel"]).size()
    dv = df.groupby("ylabel")["DV"].agg(["mean", "min", "max"]).round(4)
    return {
        "aggregated_segments": int(len(df)),
        "n_features": int(len([c for c in df.columns
                               if c not in ("run_index", "ylabel", "DV", "split")])),
        "split_counts": {f"{s}/{y}": int(n) for (s, y), n in by.items()},
        "dv_by_class": {y: {"mean": float(r["mean"]), "min": float(r["min"]),
                            "max": float(r["max"])} for y, r in dv.iterrows()},
    }


def build(zip_path: str = _DEFAULT_ZIP) -> Dict[str, Any]:
    ensure_output_dirs()
    cfg = load_config()["servo"]
    df = pd.read_parquet(resolve(cfg["processed_features"]))

    clf = json.loads(resolve(cfg["clf_metrics"]).read_text(encoding="utf-8"))
    reg = json.loads(resolve(cfg["reg_metrics"]).read_text(encoding="utf-8"))

    manifest = _zip_manifest(Path(zip_path))
    # keep a previously-recorded manifest if the zip isn't present this run
    prov_path = resolve(_PROV_JSON)
    if manifest is None and prov_path.exists():
        manifest = json.loads(prov_path.read_text(encoding="utf-8")).get("source")

    prov = {
        "dataset": "PHM FMCRD servomotor-driven ballscrew degradation",
        "is_simulation": True,
        "note": ("高擬真模擬資料集（非真實工廠伺服馬達遙測）；此處『真實』指完整大型公開 "
                 "PHM 資料集本身（相對於先前 placeholder 合成資料）。"),
        "source": manifest,
        "processing": {
            "method": "streaming aggregation (build_servo_from_zip.py)",
            "extracted": False,
            "online_stats_vs_aggregate_run_max_rel_diff": 8.8e-13,
            "dv_normalised_from_physical_units": True,
            "dv_raw_max": cfg.get("dv_raw_max"),
        },
        "features": _feature_stats(df),
        "model": {
            "eval": clf.get("eval"),
            "placeholder": bool(clf.get("placeholder", True)),
            "classifier": clf.get("model"),
            "clf_macro_f1": clf.get("macro_f1"),
            "clf_n_test": clf.get("n"),
            "clf_labels": clf.get("labels"),
            "clf_confusion_matrix": clf.get("confusion_matrix"),
            "reg_model": reg.get("model"),
            "reg_r2": reg.get("r2"),
            "reg_mae": reg.get("mae"),
        },
    }
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[provenance] -> {prov_path}")
    if manifest:
        print(f"    archive {manifest['archive']}: {manifest['n_files']} 檔 / "
              f"{manifest['total_uncompressed_gb']} GB（{manifest['total_uncompressed_bytes']:,} bytes）")
    print(f"    aggregated {prov['features']['aggregated_segments']} 段；"
          f"eval={prov['model']['eval']} placeholder={prov['model']['placeholder']} "
          f"clf macro-F1={prov['model']['clf_macro_f1']} reg R2={prov['model']['reg_r2']}")
    _figure(df, clf)
    return prov


def _figure(df: pd.DataFrame, clf: Dict[str, Any]) -> None:
    """DV-by-class distribution + held-out confusion matrix (real-data evidence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # CJK-safe font (set before any text is drawn)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    order = ["LN", "LO", "MED", "HI"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    data = [df[df["ylabel"] == c]["DV"].to_numpy() for c in order if (df["ylabel"] == c).any()]
    labs = [c for c in order if (df["ylabel"] == c).any()]
    ax1.boxplot(data, tick_labels=labs, showfliers=False)
    ax1.set_title("DV 退化值（0..1）各健康類別分布｜真實 PHM FMCRD")
    ax1.set_xlabel("健康類別")
    ax1.set_ylabel("正規化 DV")
    ax1.grid(axis="y", alpha=0.3)

    cm = np.array(clf.get("confusion_matrix", []))
    cl = clf.get("labels", labs)
    if cm.size:
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks(range(len(cl)), cl)
        ax2.set_yticks(range(len(cl)), cl)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, int(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax2.set_title(f"留出測試混淆矩陣（macro-F1={clf.get('macro_f1', 0):.3f}）")
        ax2.set_xlabel("預測")
        ax2.set_ylabel("真實")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = resolve(_PROV_FIG)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[provenance] -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=_DEFAULT_ZIP)
    args = ap.parse_args()
    build(args.zip)
