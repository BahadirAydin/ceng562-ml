from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _plot_curve(x, y, label: str, xlabel: str, ylabel: str, fname: str):
    plt.figure()
    plt.plot(x, y, lw=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=300)
    plt.close()


def plot_curves(name: str, stats: dict[str, float]):
    _plot_curve(
        stats["fpr"],
        stats["tpr"],
        label=f"{name} (ROC AUC = {stats['roc_auc']:.3f})",
        xlabel="False-Positive Rate",
        ylabel="True-Positive Rate",
        fname=f"{name}_roc.png",
    )
    _plot_curve(
        stats["recall"],
        stats["precision"],
        label=f"{name} (PR AUC = {stats['pr_auc']:.3f})",
        xlabel="Recall",
        ylabel="Precision",
        fname=f"{name}_pr.png",
    )


def plot_metric_bars(summary_df):
    plt.figure()
    summary_df["pr_auc"].plot(kind="bar")
    plt.ylabel("PR AUC")
    plt.title("Precision–Recall AUC by Model")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "auc_bar.png", dpi=150)
    plt.close()


def _get_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return -model.decision_function(x)
    if hasattr(model, "score_samples"):
        return -model.score_samples(x)


def evaluate(
    model,
    x_test: np.ndarray,
    y_test,
    anomaly_label: int = 0,
) -> Dict[str, float]:
    y_true = (y_test == anomaly_label).astype(int)  # 1 = anomaly
    scores = _get_scores(model, x_test)

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
    }
