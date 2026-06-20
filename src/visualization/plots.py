"""Plotting helpers used by EDA, training and evaluation.

All plots write to disk; nothing displays interactively so the same code can
run in scripts, notebooks and CI.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend by default
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")


def _ensure(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_type_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Type"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2")
    ax.set_title("Type distribution")
    ax.set_xlabel("Type")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_failure_type_counts(
    df: pd.DataFrame, failure_types: Sequence[str], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [int(df[ft].sum()) for ft in failure_types]
    sns.barplot(x=list(failure_types), y=counts, ax=ax, palette="rocket")
    ax.set_title("Failure-type counts (TWF/HDF/PWF/OSF/RNF)")
    ax.set_ylabel("count")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_numeric_distributions(
    df: pd.DataFrame, numeric_cols: Sequence[str], target: str, path: Path
) -> None:
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.3))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(numeric_cols):
        ax = axes[i // cols, i % cols]
        sns.kdeplot(
            data=df, x=col, hue=target, common_norm=False, ax=ax, fill=True, alpha=0.4
        )
        ax.set_title(col)
    for j in range(len(numeric_cols), rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Numeric distributions by failure label", y=1.02)
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Sequence[str], path: Path) -> None:
    corr = df[list(numeric_cols)].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model-comparison plots
# ---------------------------------------------------------------------------
def plot_metric_comparison(df: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df, x="model_name", y=metric, hue="feature_set",
        ax=ax, palette="Set2",
    )
    ax.set_title(f"{metric} by model and feature set")
    ax.set_xlabel("model")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_count_vs_metric(df: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, grp in df.groupby("model_name"):
        g = grp.sort_values("feature_count")
        ax.plot(g["feature_count"], g[metric], marker="o", label=model)
    ax.set_xlabel("feature count")
    ax.set_ylabel(metric)
    ax.set_title(f"Feature count vs {metric}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, labels: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_pr_curve(precision, recall, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    sns.barplot(data=df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(_ensure(path), dpi=120)
    plt.close(fig)
