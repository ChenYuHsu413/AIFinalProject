"""Streamlit dashboard for the predictive-maintenance prototype.

Run::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Streamlit runs this file as a script and only puts its own directory on
# ``sys.path``.  Prepend the project root so the ``src.*`` imports resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

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
from src.utils.paths import load_config, resolve

st.set_page_config(
    page_title="伺服馬達預測性維護原型系統",
    page_icon=":wrench:",
    layout="wide",
)

st.title("AI 伺服馬達故障風險預測與預測性維護建議（原型）")
st.caption(
    "本系統使用 UCI AI4I 2020（合成）資料集所訓練，作為維護決策輔助使用。"
    "系統不會直接控制馬達，僅提供故障風險預測、健康分數與維護建議。"
)

# ---------------------------------------------------------------------------
# Model status banner
# ---------------------------------------------------------------------------
try:
    bundle = load_model()
    st.success(
        f"已載入最佳模型：**{bundle.model_name}**，使用的特徵組合："
        f"**{bundle.feature_set}**。"
    )
    with st.expander("模型測試集評估指標"):
        st.json(bundle.metrics)
except FileNotFoundError as e:
    st.error("找不到已訓練的模型，請先執行：`python -m src.models.train`")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: choose page
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "頁面",
    [
        "手動單筆預測",
        "What-if 敏感度分析",
        "批次 CSV 上傳",
        "模型評估結果",
        "關於本專案",
    ],
)


# ---------------------------------------------------------------------------
# Failure-type bar chart helper (used by manual + what-if pages)
# ---------------------------------------------------------------------------
def _render_failure_type_chart(types_proba: dict, likely: list) -> None:
    import matplotlib.pyplot as plt
    order = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    probs = [types_proba.get(k, 0.0) for k in order]
    colors = ["#dc2626" if k in likely else "#94a3b8" for k in order]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(order, probs, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("機率")
    ax.set_title("第二階段：各故障類型機率")
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{p:.2%}", ha="center", fontsize=9,
        )
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="顯著門檻 0.30")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Risk-card helper
# ---------------------------------------------------------------------------
RISK_LABEL = {"Low": "低", "Medium": "中", "High": "高"}


def _render_result_card(result: dict) -> None:
    risk = result["risk_level"]
    color = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}.get(risk, "#444")
    label_zh = RISK_LABEL.get(risk, risk)
    c1, c2, c3 = st.columns(3)
    c1.metric("故障機率", f"{result['failure_probability']*100:.1f}%")
    c2.metric("健康分數", f"{result['health_score']:.1f}")
    c3.markdown(
        f"<div style='padding:14px;border-radius:8px;background:{color};"
        f"color:white;font-weight:600;text-align:center;font-size:1.2em;'>"
        f"風險等級：{label_zh}（{risk}）</div>",
        unsafe_allow_html=True,
    )
    st.subheader("維護建議")
    for tip in result["maintenance_advice"]:
        st.write("- " + tip)


# ---------------------------------------------------------------------------
# Page 1: manual prediction
# ---------------------------------------------------------------------------
if page == "手動單筆預測":
    st.header("手動輸入單筆資料")
    with st.form("manual"):
        c1, c2 = st.columns(2)
        with c1:
            type_ = st.selectbox("產品類別 Type", ["L", "M", "H"], index=0)
            air_t = st.number_input(
                "環境空氣溫度 Air temperature [K]",
                min_value=270.0, max_value=320.0, value=298.1, step=0.1,
            )
            proc_t = st.number_input(
                "製程溫度 Process temperature [K]",
                min_value=270.0, max_value=330.0, value=308.6, step=0.1,
            )
        with c2:
            rpm = st.number_input(
                "轉速 Rotational speed [rpm]",
                min_value=0.0, value=1551.0, step=10.0,
            )
            torque = st.number_input(
                "扭矩 Torque [Nm]",
                min_value=0.0, value=42.8, step=0.1,
            )
            wear = st.number_input(
                "刀具磨耗 Tool wear [min]",
                min_value=0.0, value=108.0, step=1.0,
            )
        submitted = st.form_submit_button("執行預測")

    if submitted:
        record = {
            "Type": type_,
            "Air temperature [K]": air_t,
            "Process temperature [K]": proc_t,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        # If second-stage models are available, get the richer payload
        try:
            result = predict_full(record)
            has_ft = True
        except FileNotFoundError:
            result = predict_single(record)
            has_ft = False
        _render_result_card(result)

        if has_ft:
            st.subheader("第二階段：可能的故障類型")
            _render_failure_type_chart(
                result["failure_type_probabilities"],
                result["likely_failure_types"],
            )
            for note in result.get("failure_type_notes", []):
                st.info(note)
        else:
            st.info(
                "尚未訓練第二階段故障類型模型。請執行："
                "`python -m src.models.train_failure_types`"
            )

        with st.expander("系統自動計算的衍生特徵"):
            st.json({
                "temp_diff（製程／環境溫差）": proc_t - air_t,
                "power_proxy（扭矩 × 轉速，代理功率）": torque * rpm,
                "wear_torque_interaction（磨耗 × 扭矩）": wear * torque,
                "wear_speed_interaction（磨耗 × 轉速）": wear * rpm,
                "temp_wear_interaction（製程溫度 × 磨耗）": proc_t * wear,
            })

        # ----- SHAP per-prediction explanation -----
        st.subheader("SHAP 模型解釋：這筆預測為什麼這樣判定？")
        if not shap_supported():
            st.info("目前的最佳模型不支援 TreeExplainer，因此略過 SHAP 解釋。")
        else:
            try:
                with st.spinner("計算 SHAP 貢獻..."):
                    exp = explain_record(record)
                import matplotlib.pyplot as plt
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
                fig, ax = plt.subplots(figsize=(7, 4.5))
                colors = ["#dc2626" if v > 0 else "#16a34a" for v in df_sv["shap"]]
                ax.barh(df_sv["feature"], df_sv["shap"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP contribution (log-odds of failure)")
                ax.set_title("Top 10 features pushing this prediction")
                for i, (val, sv) in enumerate(zip(df_sv["value"], df_sv["shap"])):
                    ax.text(
                        sv, i, f" {sv:+.2f} (x={val:.3g})",
                        va="center", ha="left" if sv >= 0 else "right",
                        fontsize=8,
                    )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"基準 log-odds (base value) = **{exp.base_value:.3f}**；"
                    f"本筆 log-odds 預測 = **{exp.model_output:.3f}**（sigmoid 後 ≈ 機率）。"
                    "紅色長條 → 把預測往「故障」推進；綠色長條 → 拉回「健康」。"
                )
            except Exception as e:
                st.warning(f"無法產生 SHAP 解釋：{e}")

# ---------------------------------------------------------------------------
# Page 2: What-if sensitivity analysis
# ---------------------------------------------------------------------------
elif page == "What-if 敏感度分析":
    st.header("What-if 敏感度分析")
    st.caption(
        "拖動下方滑桿，觀察故障機率與各故障類型機率如何隨運轉條件變化。"
        "可用來探索風險邊界與決策的敏感區間。"
    )

    # ---- live sliders ----
    c1, c2 = st.columns(2)
    with c1:
        wif_type = st.selectbox("產品類別 Type", ["L", "M", "H"], key="wif_type")
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

    _render_result_card(wif_result)
    if has_ft:
        _render_failure_type_chart(
            wif_result["failure_type_probabilities"],
            wif_result["likely_failure_types"],
        )

    st.divider()

    # ---- 1D sweep ----
    st.subheader("1D 掃描：單一特徵變化時的故障機率")
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
    import numpy as np
    import matplotlib.pyplot as plt
    sweep_values = np.linspace(lo, hi, n)
    sweep_probs = []
    for v in sweep_values:
        rec = dict(base_record)
        rec[sweep_feature] = float(v)
        sweep_probs.append(predict_single(rec)["failure_probability"])
    fig_sw, ax_sw = plt.subplots(figsize=(7, 3.5))
    ax_sw.plot(sweep_values, sweep_probs, color="#2563eb", linewidth=2)
    ax_sw.axvline(base_record[sweep_feature], color="orange", linestyle="--",
                  label="目前值")
    ax_sw.axhline(0.5, color="gray", linestyle=":", label="決策門檻 0.5")
    ax_sw.set_xlabel(sweep_feature)
    ax_sw.set_ylabel("故障機率")
    ax_sw.set_ylim(0, 1.02)
    ax_sw.set_title(f"其他條件固定，{sweep_feature} 變動時故障機率變化")
    ax_sw.legend(loc="best", fontsize=9)
    fig_sw.tight_layout()
    st.pyplot(fig_sw)
    plt.close(fig_sw)

    # ---- 2D heatmap ----
    st.subheader("2D 風險地景：兩特徵同時變動")
    c3, c4 = st.columns(2)
    with c3:
        feat_x = st.selectbox(
            "X 軸特徵",
            list(feature_ranges.keys()),
            index=3,
            key="heat_x",
        )
    with c4:
        feat_y = st.selectbox(
            "Y 軸特徵",
            list(feature_ranges.keys()),
            index=4,
            key="heat_y",
        )
    if feat_x == feat_y:
        st.warning("請選擇兩個不同的特徵。")
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
            grid_probs = [r["failure_probability"] for r in predict_records(grid_records)]
            Z = np.array(grid_probs).reshape(grid_n, grid_n)
        fig_h, ax_h = plt.subplots(figsize=(7, 5))
        im = ax_h.imshow(
            Z, origin="lower", aspect="auto",
            extent=[xlo, xhi, ylo, yhi], cmap="RdYlGn_r", vmin=0, vmax=1,
        )
        ax_h.scatter(
            [base_record[feat_x]], [base_record[feat_y]],
            color="white", edgecolor="black", s=80, zorder=5, label="目前值",
        )
        ax_h.set_xlabel(feat_x)
        ax_h.set_ylabel(feat_y)
        ax_h.set_title(f"故障機率（其他特徵固定）")
        ax_h.legend(loc="best", fontsize=9)
        fig_h.colorbar(im, ax=ax_h, label="failure probability")
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)
        st.caption(
            "紅 = 高故障風險、綠 = 健康。白點是目前的運轉設定。"
            "把白點往綠色區域移動就是「最安全的工作點」。"
        )


# ---------------------------------------------------------------------------
# Page 3: batch upload
# ---------------------------------------------------------------------------
elif page == "批次 CSV 上傳":
    st.header("批次預測（CSV 上傳）")
    st.write(
        "請上傳至少包含下列欄位的 CSV 檔："
        f"`{REQUIRED_INPUT_COLUMNS}`。系統會自動補上衍生特徵並進行預測。"
    )
    uploaded = st.file_uploader("選擇 CSV 檔案", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"CSV 缺少必要欄位：{missing}")
            else:
                st.write("輸入資料預覽：")
                st.dataframe(df.head())
                results = predict_records(df[REQUIRED_INPUT_COLUMNS])
                out = df.copy()
                out["failure_probability"] = [r["failure_probability"] for r in results]
                out["predicted_class"] = [r["predicted_class"] for r in results]
                out["health_score"] = [r["health_score"] for r in results]
                out["risk_level"] = [r["risk_level"] for r in results]
                st.success(f"已完成 {len(results)} 筆預測。")
                st.dataframe(out)
                st.download_button(
                    "下載預測結果 CSV",
                    out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"處理 CSV 時發生錯誤：{e}")

# ---------------------------------------------------------------------------
# Page 3: model evaluation
# ---------------------------------------------------------------------------
elif page == "模型評估結果":
    st.header("模型比較與評估")
    cfg = load_config()
    metrics_csv = resolve(cfg["paths"]["metrics_csv"])
    if not metrics_csv.exists():
        st.info("尚未產生比較表，請先執行：`python -m src.models.train`")
    else:
        comp = pd.read_csv(metrics_csv)
        st.subheader("跨模型 × 特徵組合比較表")
        st.dataframe(comp)

        # ----- Interactive threshold tuner -----
        st.subheader("互動式決策門檻調整（在測試集上即時重算）")
        preds_path = resolve(cfg["paths"]["outputs_metrics"]) / "test_predictions.csv"
        if not preds_path.exists():
            st.info(
                "尚未產生測試集機率檔。請先執行：`python -m src.models.evaluate`。"
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
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision 精準率", f"{precision:.3f}")
            m2.metric("Recall 召回率", f"{recall:.3f}")
            m3.metric("F1", f"{f1:.3f}")
            m4.metric("漏報故障 FN", fn, delta=f"誤報 FP = {fp}", delta_color="off")

            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
            cm_mat = np.array([[tn, fp], [fn, tp_]])
            sns.heatmap(
                cm_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0 (健康)", "pred 1 (故障)"],
                yticklabels=["true 0 (健康)", "true 1 (故障)"],
                ax=ax_cm, cbar=False,
            )
            ax_cm.set_title(f"Threshold = {thr:.2f}")
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.markdown(
                "**怎麼讀**：往左拉門檻 → 預測為故障的樣本變多 → Recall 上升、"
                "FN（漏報）下降，但 FP（誤報）通常會上升、Precision 下降。"
                "預測性維護情境通常偏好較低的門檻以提高 Recall。"
            )

        # ----- Static charts -----
        st.subheader("已儲存的圖表")
        c1, c2 = st.columns(2)
        fig_dir = resolve(cfg["paths"]["outputs_figures"])
        captions = {
            "confusion_matrix.png": "混淆矩陣 Confusion matrix",
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
        with c1:
            for img in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
                p = fig_dir / img
                if p.exists():
                    st.image(str(p), caption=captions[img])
        with c2:
            for img in [
                "compare_recall.png",
                "compare_f1.png",
                "compare_pr_auc.png",
                "feature_importance_native.png",
                "feature_importance_permutation.png",
            ]:
                p = fig_dir / img
                if p.exists():
                    st.image(str(p), caption=captions[img])

# ---------------------------------------------------------------------------
# Page 4: about
# ---------------------------------------------------------------------------
else:
    st.header("關於本專案")
    st.markdown(
        """
**專案定位**：以 UCI **AI4I 2020**（合成資料集）建立的預測性維護**原型系統**。

**本系統是**：以目前運轉條件作為輸入，估計故障風險、健康分數，並依據規則
提供維護建議，作為維護工程師的決策輔助工具。

**本系統不是**：
- 即時控制馬達的控制器；
- 剩餘壽命（RUL, Remaining Useful Life）的精準預測器
  （AI4I 資料集本身不含長期 run-to-failure 軌跡資料）；
- 已在實際工廠資料上驗證過的成熟系統。

詳細內容請見專案的 `README.md` 與 `outputs/reports/REPORT_OUTLINE.md`。
        """
    )
