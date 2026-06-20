"""Generate the EDA report (text summary + figures).

Run::

    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src.*`` importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import basic_eda, load_raw  # noqa: E402
from src.features.feature_engineering import add_engineered_features  # noqa: E402
from src.utils.paths import ensure_output_dirs, load_config, resolve  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_correlation_heatmap,
    plot_failure_type_counts,
    plot_numeric_distributions,
    plot_target_distribution,
    plot_type_distribution,
)


def main() -> None:
    ensure_output_dirs()
    cfg = load_config()
    fig_dir = resolve(cfg["paths"]["outputs_figures"])

    df = load_raw()
    eda = basic_eda(df)

    print("資料維度：", eda["shape"])
    print("故障比例：", eda["target_rate"])
    print("故障類型計數：", eda["failure_type_counts"])

    target = cfg["columns"]["target_primary"]
    numeric_cols = cfg["columns"]["numeric_raw"]
    failure_types = cfg["columns"]["failure_types"]

    plot_target_distribution(df, target, fig_dir / "eda_target_distribution.png")
    plot_type_distribution(df, fig_dir / "eda_type_distribution.png")
    plot_failure_type_counts(df, failure_types, fig_dir / "eda_failure_types.png")
    plot_numeric_distributions(
        df, numeric_cols, target, fig_dir / "eda_numeric_distributions.png"
    )

    # Correlation heatmap on raw + engineered features for a richer view
    df_eng = add_engineered_features(df)
    full_numeric = numeric_cols + [
        "temp_diff", "power_proxy", "wear_torque_interaction",
        "wear_speed_interaction", "temp_wear_interaction",
    ]
    plot_correlation_heatmap(df_eng, full_numeric, fig_dir / "eda_correlation.png")
    print(f"EDA 圖表已寫入 {fig_dir}")


if __name__ == "__main__":
    main()
