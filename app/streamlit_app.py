"""Streamlit dashboard for the predictive-maintenance prototype.

Run::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script and only puts its own directory on
# ``sys.path``.  Prepend the project root so the ``src.*`` imports resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except Exception:  # pragma: no cover
    HAS_SHADCN = False

from src.models.predict import (
    FAILURE_TYPE_NOTES,
    REQUIRED_INPUT_COLUMNS,
    load_failure_type_models,
    load_model,
    predict_failure_types,
    predict_full,
    predict_records,
    predict_single,
)
from src.models.explain import explain_record, is_supported as shap_supported
from src.ui import style
from src.ui.charts import (
    confusion_heatmap,
    failure_type_bar,
    one_d_sweep,
    risk_landscape,
    shap_bar,
)
from src.utils.paths import load_config, resolve


# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="伺服馬達預測性維護原型系統",
    page_icon=":wrench:",
    layout="wide",
    initial_sidebar_state="expanded",
)
style.inject()

style.hero(
    eyebrow="AI Final Project · Predictive Maintenance",
    title="伺服馬達故障風險預測與預測性維護建議",
    subtitle=(
        "以 UCI AI4I 2020（合成資料集）建立的端到端原型系統。"
        "提供故障機率、健康分數、SHAP 模型解釋與規則式維護建議，"
        "作為維護工程師的決策輔助。系統不直接控制馬達。"
    ),
)


# ---------------------------------------------------------------------------
# Model status banner
# ---------------------------------------------------------------------------
try:
    bundle = load_model()
except FileNotFoundError:
    style.note(
        "找不到已訓練的模型。請先在終端執行 "
        "<code>python -m src.models.train</code> 產生 "
        "<code>outputs/models/best_model.joblib</code>。",
        kind="danger",
    )
    st.stop()

cols_status = st.columns([2, 1, 1, 1, 1])
with cols_status[0]:
    style.note(
        f"已載入最佳模型 <b>{bundle.model_name}</b> · 特徵組合 "
        f"<b>{bundle.feature_set}</b>",
        kind="info",
    )


def _shadcn_metric(label: str, value: str, delta: str | None = None, key: str | None = None) -> None:
    """Render a shadcn-ui metric card if available, otherwise fall back to HTML."""
    if HAS_SHADCN:
        try:
            ui.metric_card(
                title=label, content=value,
                description=delta or "", key=key or f"m-{label}",
            )
            return
        except Exception:
            pass
    style.fallback_metric_card(label, value, delta)


m = bundle.metrics
with cols_status[1]:
    _shadcn_metric("Recall", f"{m['recall']:.3f}", "在故障樣本上的命中率", key="hdr-rec")
with cols_status[2]:
    _shadcn_metric("F1", f"{m['f1']:.3f}", "Precision／Recall 調和平均", key="hdr-f1")
with cols_status[3]:
    _shadcn_metric("ROC-AUC", f"{m['roc_auc']:.3f}", "閾值無關的整體鑑別力", key="hdr-roc")
with cols_status[4]:
    _shadcn_metric("PR-AUC", f"{m['pr_auc']:.3f}", "不平衡資料的首選指標", key="hdr-pr")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 導覽")
page = st.sidebar.radio(
    "頁面",
    [
        "手動單筆預測",
        "What-if 敏感度分析",
        "批次 CSV 上傳",
        "模型評估結果",
        "關於本專案",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**資料集**：UCI AI4I 2020（合成）\n\n"
    "**用途**：維護決策輔助，**不**直接控制馬達。\n\n"
    "Repo · `ChenYuHsu413/AIFinalProject`"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
RISK_LABEL = {"Low": "低", "Medium": "中", "High": "高"}


def render_result_card(result: dict) -> None:
    """Top result strip: probability + health + risk pill (with shadcn cards)."""
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        _shadcn_metric(
            "故障機率", f"{result['failure_probability']*100:.1f}%",
            "≥ 0.7 視為高風險", key=f"res-prob-{id(result)}",
        )
    with c2:
        _shadcn_metric(
            "健康分數", f"{result['health_score']:.1f}",
            "(1 − 故障機率) × 100", key=f"res-health-{id(result)}",
        )
    with c3:
        st.markdown("<div style='padding-top:10px'></div>", unsafe_allow_html=True)
        style.risk_pill(result["risk_level"],
                        RISK_LABEL.get(result["risk_level"], result["risk_level"]))


def render_advice(result: dict) -> None:
    style.section("維護建議")
    advice = result.get("maintenance_advice", [])
    if not advice:
        return
    for tip in advice:
        st.markdown(
            f"<div class='note-box'>{tip}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page 1: manual prediction
# ---------------------------------------------------------------------------
if page == "手動單筆預測":
    style.section("輸入運轉條件")
    with st.form("manual"):
        c1, c2 = st.columns(2)
        with c1:
            type_ = st.selectbox("產品類別 Type", ["L", "M", "H"], index=0)
            air_t = st.number_input(
                "Air temperature [K]", min_value=270.0, max_value=320.0,
                value=298.1, step=0.1,
            )
            proc_t = st.number_input(
                "Process temperature [K]", min_value=270.0, max_value=330.0,
                value=308.6, step=0.1,
            )
        with c2:
            rpm = st.number_input(
                "Rotational speed [rpm]", min_value=0.0, value=1551.0, step=10.0,
            )
            torque = st.number_input(
                "Torque [Nm]", min_value=0.0, value=42.8, step=0.1,
            )
            wear = st.number_input(
                "Tool wear [min]", min_value=0.0, value=108.0, step=1.0,
            )
        submitted = st.form_submit_button(":rocket: 執行預測", type="primary",
                                          use_container_width=True)

    if submitted:
        record = {
            "Type": type_,
            "Air temperature [K]": air_t,
            "Process temperature [K]": proc_t,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        try:
            result = predict_full(record)
            has_ft = True
        except FileNotFoundError:
            result = predict_single(record)
            has_ft = False

        style.section("預測結果")
        render_result_card(result)
        render_advice(result)

        if has_ft:
            style.section("第二階段：可能的故障類型")
            fig = failure_type_bar(
                result["failure_type_probabilities"],
                result["likely_failure_types"],
            )
            st.plotly_chart(fig, use_container_width=True)
            for note in result.get("failure_type_notes", []):
                style.note(note)
        else:
            style.note(
                "尚未訓練第二階段故障類型模型。請執行 "
                "<code>python -m src.models.train_failure_types</code>",
                kind="warn",
            )

        with st.expander("系統自動計算的衍生特徵"):
            st.json({
                "temp_diff（製程／環境溫差）": proc_t - air_t,
                "power_proxy（扭矩 × 轉速，代理功率）": torque * rpm,
                "wear_torque_interaction（磨耗 × 扭矩）": wear * torque,
                "wear_speed_interaction（磨耗 × 轉速）": wear * rpm,
                "temp_wear_interaction（製程溫度 × 磨耗）": proc_t * wear,
            })

        # SHAP per-prediction explanation
        style.section("SHAP 模型解釋")
        if not shap_supported():
            style.note("目前的最佳模型不支援 TreeExplainer，因此略過 SHAP 解釋。",
                       kind="warn")
        else:
            try:
                with st.spinner("計算 SHAP 貢獻..."):
                    exp = explain_record(record)
                df_sv = (
                    pd.DataFrame({
                        "feature": exp.feature_names,
                        "value": exp.feature_values,
                        "shap": exp.shap_values,
                    })
                    .assign(abs_shap=lambda d: d["shap"].abs())
                    .sort_values("abs_shap", ascending=False)
                    .head(10)
                    .iloc[::-1]
                    .reset_index(drop=True)
                )
                st.plotly_chart(shap_bar(df_sv), use_container_width=True)
                style.note(
                    f"基準 log-odds = <b>{exp.base_value:.3f}</b>；"
                    f"本筆 log-odds = <b>{exp.model_output:.3f}</b>"
                    "（sigmoid 後 ≈ 機率）。紅色長條把預測推向「故障」，"
                    "綠色長條拉回「健康」。",
                )
            except Exception as e:
                style.note(f"無法產生 SHAP 解釋：{e}", kind="warn")


# ---------------------------------------------------------------------------
# Page 2: What-if sensitivity analysis
# ---------------------------------------------------------------------------
elif page == "What-if 敏感度分析":
    style.section("即時操控運轉條件")
    st.caption(
        "拖動下方滑桿，觀察故障機率與各故障類型機率如何隨運轉條件變化。"
        "下半部還可選擇要掃描的特徵與 2D 風險地景。"
    )

    c1, c2 = st.columns(2)
    with c1:
        wif_type = st.selectbox("Type", ["L", "M", "H"], key="wif_type")
        wif_air = st.slider(
            "Air temperature [K]", 295.0, 305.0, 298.1, step=0.1, key="wif_air"
        )
        wif_proc = st.slider(
            "Process temperature [K]", 305.0, 315.0, 308.6, step=0.1, key="wif_proc"
        )
    with c2:
        wif_rpm = st.slider(
            "Rotational speed [rpm]", 1100.0, 2900.0, 1551.0, step=10.0, key="wif_rpm"
        )
        wif_torque = st.slider(
            "Torque [Nm]", 0.0, 80.0, 42.8, step=0.5, key="wif_torque"
        )
        wif_wear = st.slider(
            "Tool wear [min]", 0.0, 260.0, 108.0, step=1.0, key="wif_wear"
        )

    base_record = {
        "Type": wif_type,
        "Air temperature [K]": wif_air,
        "Process temperature [K]": wif_proc,
        "Rotational speed [rpm]": wif_rpm,
        "Torque [Nm]": wif_torque,
        "Tool wear [min]": wif_wear,
    }
    try:
        wif_result = predict_full(base_record)
        has_ft = True
    except FileNotFoundError:
        wif_result = predict_single(base_record)
        has_ft = False

    style.section("即時預測")
    render_result_card(wif_result)

    if has_ft:
        st.plotly_chart(
            failure_type_bar(
                wif_result["failure_type_probabilities"],
                wif_result["likely_failure_types"],
            ),
            use_container_width=True,
        )

    st.divider()

    # ---- 1D sweep ----
    style.section("1D 掃描：單一特徵變化時的故障機率")
    sweep_feature = st.selectbox(
        "選擇要掃描的特徵",
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
        index=3,
        key="sweep_feature",
    )
    feature_ranges = {
        "Air temperature [K]": (295.0, 305.0, 41),
        "Process temperature [K]": (305.0, 315.0, 41),
        "Rotational speed [rpm]": (1100.0, 2900.0, 41),
        "Torque [Nm]": (0.0, 80.0, 41),
        "Tool wear [min]": (0.0, 260.0, 41),
    }
    lo, hi, n = feature_ranges[sweep_feature]
    sweep_values = np.linspace(lo, hi, n)
    sweep_probs = []
    for v in sweep_values:
        rec = dict(base_record)
        rec[sweep_feature] = float(v)
        sweep_probs.append(predict_single(rec)["failure_probability"])
    st.plotly_chart(
        one_d_sweep(sweep_values, sweep_probs, sweep_feature,
                    base_record[sweep_feature]),
        use_container_width=True,
    )

    # ---- 2D heatmap ----
    style.section("2D 風險地景：兩特徵同時變動")
    c3, c4 = st.columns(2)
    with c3:
        feat_x = st.selectbox("X 軸特徵", list(feature_ranges.keys()),
                              index=3, key="heat_x")
    with c4:
        feat_y = st.selectbox("Y 軸特徵", list(feature_ranges.keys()),
                              index=4, key="heat_y")
    if feat_x == feat_y:
        style.note("請選擇兩個不同的特徵。", kind="warn")
    else:
        with st.spinner("計算風險地景..."):
            xlo, xhi, _ = feature_ranges[feat_x]
            ylo, yhi, _ = feature_ranges[feat_y]
            grid_n = 25
            xs = np.linspace(xlo, xhi, grid_n)
            ys = np.linspace(ylo, yhi, grid_n)
            grid_records = []
            for yv in ys:
                for xv in xs:
                    rec = dict(base_record)
                    rec[feat_x] = float(xv)
                    rec[feat_y] = float(yv)
                    grid_records.append(rec)
            grid_probs = [r["failure_probability"]
                          for r in predict_records(grid_records)]
            Z = np.array(grid_probs).reshape(grid_n, grid_n)
        st.plotly_chart(
            risk_landscape(xs, ys, Z, feat_x, feat_y,
                           base_record[feat_x], base_record[feat_y]),
            use_container_width=True,
        )
        style.note(
            "紅 = 高故障風險、綠 = 健康。白點是目前運轉設定。"
            "把白點移向綠色區域就是「最安全的工作點」。"
        )


# ---------------------------------------------------------------------------
# Page 3: batch upload
# ---------------------------------------------------------------------------
elif page == "批次 CSV 上傳":
    style.section("批次預測（CSV 上傳）")
    style.note(
        "上傳至少包含這些欄位的 CSV："
        f"<code>{', '.join(REQUIRED_INPUT_COLUMNS)}</code>"
        "；系統會自動補上衍生特徵並執行批次推論。"
    )
    uploaded = st.file_uploader("選擇 CSV 檔案", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
            if missing:
                style.note(f"CSV 缺少欄位：{missing}", kind="danger")
            else:
                style.section("輸入預覽")
                st.dataframe(df.head(), use_container_width=True)
                results = predict_records(df[REQUIRED_INPUT_COLUMNS])
                out = df.copy()
                out["failure_probability"] = [r["failure_probability"] for r in results]
                out["predicted_class"] = [r["predicted_class"] for r in results]
                out["health_score"] = [r["health_score"] for r in results]
                out["risk_level"] = [r["risk_level"] for r in results]

                style.section(f"預測結果（{len(results)} 筆）")
                # quick summary metrics
                n_high = sum(r["risk_level"] == "High" for r in results)
                n_med = sum(r["risk_level"] == "Medium" for r in results)
                n_low = sum(r["risk_level"] == "Low" for r in results)
                k1, k2, k3 = st.columns(3)
                with k1:
                    _shadcn_metric("Low", str(n_low), "風險等級分布",
                                   key="bat-low")
                with k2:
                    _shadcn_metric("Medium", str(n_med), "風險等級分布",
                                   key="bat-med")
                with k3:
                    _shadcn_metric("High", str(n_high), "風險等級分布",
                                   key="bat-high")
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    ":inbox_tray: 下載預測結果 CSV",
                    out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as e:
            style.note(f"處理 CSV 時發生錯誤：{e}", kind="danger")


# ---------------------------------------------------------------------------
# Page 4: model evaluation
# ---------------------------------------------------------------------------
elif page == "模型評估結果":
    style.section("模型比較與評估")
    cfg = load_config()
    metrics_csv = resolve(cfg["paths"]["metrics_csv"])
    if not metrics_csv.exists():
        style.note("尚未產生比較表，請先執行 "
                   "<code>python -m src.models.train</code>。", kind="warn")
    else:
        comp = pd.read_csv(metrics_csv)
        st.dataframe(comp, use_container_width=True)

        # ---- Interactive threshold tuner ----
        style.section("互動式決策門檻（即時重算測試集指標）")
        preds_path = resolve(cfg["paths"]["outputs_metrics"]) / "test_predictions.csv"
        if not preds_path.exists():
            style.note(
                "尚未產生測試集機率檔。請先執行 "
                "<code>python -m src.models.evaluate</code>。",
                kind="warn",
            )
        else:
            tp = pd.read_csv(preds_path)
            n_total = len(tp)
            n_pos = int(tp["y_true"].sum())
            st.caption(
                f"測試集共 {n_total} 筆；其中故障樣本 {n_pos} 筆"
                f"（比例 {n_pos / n_total * 100:.2f}%）。"
            )
            thr = st.slider(
                "決策門檻 threshold（機率 ≥ threshold ⇒ 預測為故障）",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            )
            y_true = tp["y_true"].astype(int).values
            y_pred = (tp["y_proba"].values >= thr).astype(int)
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tp_ = int(((y_true == 1) & (y_pred == 1)).sum())
            precision = tp_ / (tp_ + fp) if (tp_ + fp) else 0.0
            recall = tp_ / (tp_ + fn) if (tp_ + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                _shadcn_metric("Precision", f"{precision:.3f}",
                               "誤報越少越高", key="thr-prec")
            with m2:
                _shadcn_metric("Recall", f"{recall:.3f}",
                               "漏報越少越高", key="thr-rec")
            with m3:
                _shadcn_metric("F1", f"{f1:.3f}",
                               "兩者調和平均", key="thr-f1")
            with m4:
                _shadcn_metric("漏報 FN", str(fn),
                               f"誤報 FP = {fp}", key="thr-fn")

            cm_mat = np.array([[tn, fp], [fn, tp_]])
            st.plotly_chart(confusion_heatmap(cm_mat, thr),
                            use_container_width=True)
            style.note(
                "<b>怎麼讀</b>：往左拉門檻 → 預測為故障的樣本變多 → "
                "Recall 上升、FN（漏報）下降，但 FP（誤報）通常會上升、"
                "Precision 下降。預測性維護情境通常偏好較低的門檻以提高 Recall。"
            )

        # ---- Static charts ----
        style.section("已儲存的訓練 / 評估圖表")
        fig_dir = resolve(cfg["paths"]["outputs_figures"])
        captions = {
            "confusion_matrix.png": "混淆矩陣",
            "roc_curve.png": "ROC 曲線",
            "pr_curve.png": "Precision-Recall 曲線",
            "compare_recall.png": "各模型 Recall 比較",
            "compare_f1.png": "各模型 F1 比較",
            "compare_pr_auc.png": "各模型 PR-AUC 比較",
            "compare_roc_auc.png": "各模型 ROC-AUC 比較",
            "feature_importance_native.png": "原生特徵重要性",
            "feature_importance_permutation.png": "Permutation 特徵重要性",
            "feature_count_vs_f1.png": "特徵數 vs F1",
            "feature_count_vs_recall.png": "特徵數 vs Recall",
            "feature_count_vs_pr_auc.png": "特徵數 vs PR-AUC",
        }
        cols = st.columns(2)
        for i, (img, cap) in enumerate(captions.items()):
            p = fig_dir / img
            if p.exists():
                with cols[i % 2]:
                    st.image(str(p), caption=cap, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5: about
# ---------------------------------------------------------------------------
else:
    style.section("關於本專案")
    st.markdown(
        """
**專案定位**：以 UCI **AI4I 2020**（合成資料集）建立的預測性維護**原型系統**。

**本系統是**：以目前運轉條件作為輸入，估計故障風險、健康分數，
並依規則提供維護建議，作為維護工程師的決策輔助工具。

**本系統不是**：
- 即時控制馬達的控制器；
- 剩餘壽命（RUL, Remaining Useful Life）的精準預測器
  （AI4I 資料集本身不含長期 run-to-failure 軌跡資料）；
- 已在實際工廠資料上驗證過的成熟系統。

詳細內容請見專案的 `README.md` 與 `outputs/reports/REPORT_OUTLINE.md`。

---

### 技術棧
- **資料 / ML**：scikit-learn · pandas · numpy · XGBoost · LightGBM
- **可解釋性**：SHAP（TreeExplainer）· Permutation Importance
- **調參**：Optuna
- **UI**：Streamlit · Plotly · streamlit-shadcn-ui
- **服務 / 部署**：FastAPI · Uvicorn · Docker · docker-compose · GitHub Actions

### Repo
<a href="https://github.com/ChenYuHsu413/AIFinalProject" target="_blank">
github.com/ChenYuHsu413/AIFinalProject
</a>
        """,
        unsafe_allow_html=True,
    )
