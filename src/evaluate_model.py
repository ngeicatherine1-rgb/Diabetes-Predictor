"""
Model evaluation metrics and visualization
"""

import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (for ROC-AUC)
        model_name (str): Name of the model

    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])

    logger.info(
        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )

    return metrics


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    return plt


def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models.

    Args:
        models_dict (dict): Dictionary of model names and model objects
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary of evaluation results
    """
    results = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name=name)
        results[name] = metrics

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage would go here
    pass
