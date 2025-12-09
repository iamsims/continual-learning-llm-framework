"""
Evaluation module for classification tasks.

Provides comprehensive metrics and reporting for model evaluation.
"""

from .evaluator import ClassificationEvaluator
from .metrics import EvaluationMetrics

__all__ = ['ClassificationEvaluator', 'EvaluationMetrics']
