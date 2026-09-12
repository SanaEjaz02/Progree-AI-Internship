"""
Model Evaluation Module for Multi-Class Text Sentiment Classifier.

Computes Macro and Weighted F1-scores, Precision, Recall, Classification Report,
and Confusion Matrix (both terminal formatted and Seaborn heatmap visualization).
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
# Use Agg backend for non-interactive server/script environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def evaluate_model(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    save_cm_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for multi-class classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of unique class names (e.g. ['negative', 'neutral', 'positive']).
        save_cm_path: File path to save the confusion matrix plot (PNG).

    Returns:
        Dictionary containing calculated metrics and formatted reports.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    # Primary Required Metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Detailed per-class classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Generate and save Seaborn heatmap if path requested
    if save_cm_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_cm_path)), exist_ok=True)
        plt.figure(figsize=(7, 6))
        sns.set_theme(style="white")
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            linewidths=1.0,
            linecolor="white",
            square=True,
            annot_kws={"size": 13, "weight": "bold"}
        )
        plt.title("Multi-Class Sentiment Confusion Matrix", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Predicted Sentiment", fontsize=12, labelpad=10)
        plt.ylabel("Actual Sentiment", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=300)
        plt.close()

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "labels": labels,
        "cm_plot_path": save_cm_path
    }
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Nicely prints the evaluation results to the console.
    """
    labels = metrics["labels"]
    cm = np.array(metrics["confusion_matrix"])

    print("\n" + "=" * 65)
    print("           MULTI-CLASS SENTIMENT MODEL EVALUATION")
    print("=" * 65)
    print(f" Overall Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f" Macro F1-Score:        {metrics['macro_f1']:.4f}  <-- Primary Metric (Unweighted)")
    print(f" Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    print(f" Macro Precision:       {metrics['macro_precision']:.4f}")
    print(f" Macro Recall:          {metrics['macro_recall']:.4f}")
    print("-" * 65)
    print("Classification Report:")
    print(metrics["classification_report_text"])
    print("-" * 65)
    print("Confusion Matrix:")
    
    # Formatted confusion matrix display
    header = f"{'Actual \\ Pred':<16} " + " ".join(f"{lbl:>10}" for lbl in labels)
    print(header)
    for i, row_lbl in enumerate(labels):
        row_vals = " ".join(f"{cm[i, j]:>10d}" for j in range(len(labels)))
        print(f"{row_lbl:<16} {row_vals}")
    
    if metrics.get("cm_plot_path"):
        print(f"\n[Artifact Saved] Confusion Matrix Heatmap -> {metrics['cm_plot_path']}")
    print("=" * 65 + "\n")
