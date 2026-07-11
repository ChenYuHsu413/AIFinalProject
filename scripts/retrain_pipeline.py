#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Task C — one-command retrain → validate → promote / reject.

Flow:
  1. Train the reference models into ``models/registry/candidate_<timestamp>/``
     (NEVER touches the active/deployed version).
  2. Run the validation gate (``src.pipeline.validation_gate``) against the
     ACTIVE version.
  3. PASS  → rename the candidate to the next ``v<n+1>``, switch
     ``registry.json`` active, log the version comparison. Exit 0.
     FAIL  → keep the candidate dir + ``gate_report.json`` for inspection, leave
     active unchanged, exit non-zero.

Options:
  --dry-run            train + validate, but DO NOT switch active (candidate kept).
  --data-config <j>    JSON (inline or file path) subsetting the training data,
                       e.g. '{"train_frac":0.1}' to synthesise a degraded
                       candidate (also the hook S4's drift scenarios will use).
  --note <text>        note stored in registry.json for the promoted version.

Run::

    python scripts/retrain_pipeline.py
    python scripts/retrain_pipeline.py --dry-run
    python scripts/retrain_pipeline.py --data-config '{"train_frac":0.1}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import servo_model_registry as registry  # noqa: E402
from src.models import train_servo  # noqa: E402
from src.pipeline.validation_gate import _print_report, run_gate  # noqa: E402


def _parse_data_config(arg: str | None):
    if not arg:
        return None
    if os.path.exists(arg):
        return json.loads(Path(arg).read_text(encoding="utf-8"))
    return json.loads(arg)


def _log_comparison(report: dict, old_active: str | None, new_version: str) -> None:
    hm = next((c for c in report["checks"] if c["name"] == "holdout_metrics"), None)
    nums = (hm or {}).get("numbers", {})
    cand, act = nums.get("candidate", {}), nums.get("active", {})
    print(f"\n[pipeline] ✅ 已轉正 {new_version}（原 active={old_active} → 新 active={new_version}）")
    if cand and act:
        print(f"    macro-F1: {act.get('macro_f1'):.4f} → {cand.get('macro_f1'):.4f} "
              f"(Δ{nums.get('macro_f1_delta'):+})")
        print(f"    DV R²   : {act.get('dv_r2'):.4f} → {cand.get('dv_r2'):.4f} "
              f"(Δ{nums.get('dv_r2_delta'):+})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="訓練+驗證，但不切換 active")
    ap.add_argument("--data-config", default=None,
                    help="JSON（字串或檔案路徑）指定訓練資料子集，如 '{\"train_frac\":0.1}'")
    ap.add_argument("--note", default=None, help="轉正版本的備註（寫入 registry.json）")
    args = ap.parse_args()
    data_config = _parse_data_config(args.data_config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_dir = registry.registry_dir() / f"candidate_{ts}"
    old_active = registry.active_version(default=None)
    print(f"[pipeline] 訓練候選 -> {cand_dir}（active={old_active}"
          + (f", data_config={data_config}" if data_config else "") + ")")
    train_servo.run(out_dir=cand_dir, data_config=data_config)

    report = run_gate(cand_dir, active_version=old_active)
    _print_report(report)

    if not report["passed"]:
        print(f"\n[pipeline] ❌ FAIL — 候選未通過驗證閘門；active 維持 {old_active} 不變。")
        print(f"    候選保留供人工檢視：{cand_dir}")
        print(f"    gate_report：{cand_dir / 'gate_report.json'}")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[pipeline] dry-run：通過但不切換（active 仍為 {old_active}）。候選：{cand_dir}")
        sys.exit(0)

    metrics = json.loads((cand_dir / "metrics.json").read_text(encoding="utf-8"))
    new_version = registry.promote_candidate(
        cand_dir, metrics, note=args.note or f"retrain {ts}")
    _log_comparison(report, old_active, new_version)
    sys.exit(0)


if __name__ == "__main__":
    main()
