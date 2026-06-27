# Prompt 紀錄 — 2026-06-20 session

> 由 `conversation-2026-06-20-174842.txt`(Claude Code 終端匯出)抽取,共 11 則使用者
> prompt。純 UI 斜線指令(`/model`)未列入。本日為專案 Day 1:從零建立模組 A(AI4I
> 2020 預測性維護原型),以 CRISP-DM 流程完成全棧 → 中文化 → 加 SHAP / 互動閾值 →
> 二階段故障類型 + What-if → 超參數調整 + 模型卡 → Docker + CI → 推送 GitHub +
> LICENSE / repo description。

## Prompt 1

請在實作前先列出你打算建立或修改的檔案清單,並簡短說明每個檔案的用途。若專案
中已有既有檔案,請先分析現有架構,再決定如何整合,不要直接覆蓋重要內容。你是
一位資深 Machine Learning Engineer、Data Scientist 與 Full-stack Developer。請協
助我從零到一實作一個 AI 職訓專題專案。

專案主題:「AI 伺服馬達故障風險預測與預測性維護建議系統」

請注意:本專案目前使用 UCI AI4I 2020 Predictive Maintenance Dataset 作為工業預測
性維護原型資料集。它是合成資料集,不是真實伺服馬達長期運轉資料。因此專案定位應
該是「預測性維護原型系統」、「故障風險預測」、「健康狀態評估」、「維護建議」,
不要誇大成已完成真實工廠驗證的精準壽命預測系統。

我希望你實作一個完整、可展示、可部署、可閱讀的專案。請依照 CRISP-DM 流程完成
(Business Understanding / Data Understanding / Data Preparation / Modeling /
Evaluation / Deployment)。

(原始 prompt 約 330 行,涵蓋:資料集欄位定義與 leakage 防範、CRISP-DM 各步驟細節
要求、特徵工程清單(`temp_diff` / `power_proxy` / `wear_torque` / `wear_speed` /
`temp_wear`)、10 模型 × 5 特徵組合的比較表、評估指標(F1 / Recall / PR-AUC,不
能只看 Accuracy)、Streamlit + FastAPI 雙 UI、GCP VM 部署流程、報告大綱
(REPORT_OUTLINE)、需建立的檔案清單、執行方式、十三項注意事項。終端匯出在中段
省略了 317 行,但完整檔案結構已體現在後續 commits。)

## Prompt 2

can you run all of these of me

## Prompt 3

ModuleNotFoundError: No module named 'src'

File
"C:\Users\alung\Documents\WorkSpace\AIHW\FinalProject\app\streamlit_app.py",
line 15, in <module>
    from src.models.predict import (
    ...<4 lines>...
    )

## Prompt 4

可以幫我中文化嗎

## Prompt 5

有看到了,接下來先不要做任何改動
純討論,你覺得這看起來像是個成熟完整的大專案嗎?
我覺得東西感覺有一點點少,要不要再多增加一點可互動的部分? 或者更專業一點的

## Prompt 6

那就先做 SHAP 解釋跟互動式閾值

## Prompt 7

再加二階段故障類型 還有what if吧

## Prompt 8

好 繼續做超參數調整 還有模型卡

## Prompt 9

好 這兩個都做吧

## Prompt 10

https://github.com/ChenYuHsu413/AIFinalProject.git 幫我把這份檔案推送到這邊

## Prompt 11

加 LICENSE 跟 repo description
