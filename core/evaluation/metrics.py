"""
Core metrics calculations for classification evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score
)


@dataclass
class EvaluationMetrics:
    """
    Container for classification evaluation metrics.

    Attributes:
        accuracy: Overall accuracy
        balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        precision_macro: Macro-averaged precision
        precision_micro: Micro-averaged precision
        precision_weighted: Weighted precision (by support)
        recall_macro: Macro-averaged recall
        recall_micro: Micro-averaged recall
        recall_weighted: Weighted recall
        f1_macro: Macro-averaged F1 score
        f1_micro: Micro-averaged F1 score
        f1_weighted: Weighted F1 score
        matthews_corrcoef: Matthews correlation coefficient
        cohen_kappa: Cohen's kappa score
        per_class_metrics: Metrics for each class
        confusion_matrix: Confusion matrix
        classification_report: Full classification report
        num_samples: Total number of samples
        num_classes: Number of classes
        class_distribution: Distribution of true labels
    """

    # Overall metrics
    accuracy: float
    balanced_accuracy: float

    # Precision
    precision_macro: float
    precision_micro: float
    precision_weighted: float

    # Recall
    recall_macro: float
    recall_micro: float
    recall_weighted: float

    # F1 Score
    f1_macro: float
    f1_micro: float
    f1_weighted: float

    # Other metrics
    matthews_corrcoef: float
    cohen_kappa: float

    # Detailed metrics
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: str = ""

    # Metadata
    num_samples: int = 0
    num_classes: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)

    # Optional: ROC AUC for binary/multiclass
    roc_auc_ovr: Optional[float] = None  # One-vs-Rest
    roc_auc_ovo: Optional[float] = None  # One-vs-One

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'overall': {
                'accuracy': self.accuracy,
                'balanced_accuracy': self.balanced_accuracy,
                'matthews_corrcoef': self.matthews_corrcoef,
                'cohen_kappa': self.cohen_kappa,
            },
            'precision': {
                'macro': self.precision_macro,
                'micro': self.precision_micro,
                'weighted': self.precision_weighted,
            },
            'recall': {
                'macro': self.recall_macro,
                'micro': self.recall_micro,
                'weighted': self.recall_weighted,
            },
            'f1_score': {
                'macro': self.f1_macro,
                'micro': self.f1_micro,
                'weighted': self.f1_weighted,
            },
            'roc_auc': {
                'ovr': self.roc_auc_ovr,
                'ovo': self.roc_auc_ovo,
            },
            'per_class': self.per_class_metrics,
            'metadata': {
                'num_samples': self.num_samples,
                'num_classes': self.num_classes,
                'class_distribution': self.class_distribution,
            },
            'confusion_matrix': self.confusion_matrix.tolist() if isinstance(self.confusion_matrix, np.ndarray) else self.confusion_matrix
        }

    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        lines = [
            "=" * 60,
            "Classification Evaluation Summary",
            "=" * 60,
            f"Total Samples: {self.num_samples}",
            f"Number of Classes: {self.num_classes}",
            "",
            "Overall Metrics:",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  Balanced Accuracy: {self.balanced_accuracy:.4f}",
            f"  Matthews Corr:     {self.matthews_corrcoef:.4f}",
            f"  Cohen's Kappa:     {self.cohen_kappa:.4f}",
            "",
            "Macro-Averaged Metrics:",
            f"  Precision: {self.precision_macro:.4f}",
            f"  Recall:    {self.recall_macro:.4f}",
            f"  F1-Score:  {self.f1_macro:.4f}",
            "",
            "Weighted Metrics:",
            f"  Precision: {self.precision_weighted:.4f}",
            f"  Recall:    {self.recall_weighted:.4f}",
            f"  F1-Score:  {self.f1_weighted:.4f}",
        ]

        if self.roc_auc_ovr is not None:
            lines.extend([
                "",
                "ROC AUC:",
                f"  One-vs-Rest: {self.roc_auc_ovr:.4f}",
            ])
            if self.roc_auc_ovo is not None:
                lines.append(f"  One-vs-One:  {self.roc_auc_ovo:.4f}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def get_metric(self, metric_name: str, averaging: str = 'macro') -> float:
        """
        Get a specific metric by name.

        Args:
            metric_name: Name of the metric (accuracy, precision, recall, f1)
            averaging: Averaging method (macro, micro, weighted) - only for precision/recall/f1

        Returns:
            Metric value
        """
        if metric_name == 'accuracy':
            return self.accuracy
        elif metric_name == 'balanced_accuracy':
            return self.balanced_accuracy
        elif metric_name == 'matthews_corrcoef':
            return self.matthews_corrcoef
        elif metric_name == 'cohen_kappa':
            return self.cohen_kappa
        elif metric_name == 'precision':
            return getattr(self, f'precision_{averaging}')
        elif metric_name == 'recall':
            return getattr(self, f'recall_{averaging}')
        elif metric_name == 'f1':
            return getattr(self, f'f1_{averaging}')
        elif metric_name == 'roc_auc_ovr':
            return self.roc_auc_ovr
        elif metric_name == 'roc_auc_ovo':
            return self.roc_auc_ovo
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    probabilities: Optional[np.ndarray] = None
) -> EvaluationMetrics:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (can be string or numeric)
        y_pred: Predicted labels (can be string or numeric)
        labels: List of label names (optional)
        probabilities: Prediction probabilities for ROC AUC calculation (optional)

    Returns:
        EvaluationMetrics object with all calculated metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique labels
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_labels)

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Calculate precision, recall, f1 for different averaging methods
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    # Build per-class metrics dictionary
    if labels is not None and len(labels) == num_classes:
        label_names = labels
    else:
        label_names = [str(label) for label in unique_labels]

    per_class_metrics = {}
    for i, label_name in enumerate(label_names):
        per_class_metrics[label_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }

    # Matthews correlation coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # Cohen's kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Classification report
    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names, zero_division=0)

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = {label_names[list(unique_labels).index(label)]: int(count)
                  for label, count in zip(unique, counts)}

    # ROC AUC (if probabilities provided)
    roc_auc_ovr = None
    roc_auc_ovo = None
    if probabilities is not None and len(probabilities.shape) == 2:
        try:
            # For multiclass, use one-vs-rest
            if num_classes > 2:
                roc_auc_ovr = roc_auc_score(
                    y_true, probabilities,
                    multi_class='ovr',
                    average='macro',
                    labels=unique_labels
                )
                roc_auc_ovo = roc_auc_score(
                    y_true, probabilities,
                    multi_class='ovo',
                    average='macro',
                    labels=unique_labels
                )
            else:
                # Binary classification
                roc_auc_ovr = roc_auc_score(y_true, probabilities[:, 1])
        except (ValueError, IndexError):
            # ROC AUC calculation failed (e.g., only one class present)
            pass

    return EvaluationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced_acc),
        precision_macro=float(precision_macro),
        precision_micro=float(precision_micro),
        precision_weighted=float(precision_weighted),
        recall_macro=float(recall_macro),
        recall_micro=float(recall_micro),
        recall_weighted=float(recall_weighted),
        f1_macro=float(f1_macro),
        f1_micro=float(f1_micro),
        f1_weighted=float(f1_weighted),
        matthews_corrcoef=float(mcc),
        cohen_kappa=float(kappa),
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
        classification_report=report,
        num_samples=len(y_true),
        num_classes=num_classes,
        class_distribution=class_dist,
        roc_auc_ovr=roc_auc_ovr,
        roc_auc_ovo=roc_auc_ovo
    )
