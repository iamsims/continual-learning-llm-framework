"""
Classification Evaluator - Main interface for model evaluation.

Integrates seamlessly with InferenceEngine output format.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
import logging

from .metrics import calculate_metrics, EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationEvaluator:

    def __init__(self, verbose: bool = True):
        """
        Initialize the evaluator.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.last_metrics: Optional[EvaluationMetrics] = None

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def evaluate(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        predicted_label_column: str = 'predicted_label',
        predicted_id_column: str = 'predicted_label_id',
        confidence_column: str = 'confidence',
        include_probabilities: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate predictions against true labels.

        Args:
            df: DataFrame with predictions (from InferenceEngine) and true labels
            true_label_column: Column name containing true labels
            predicted_label_column: Column name containing predicted labels (default: 'predicted_label')
            predicted_id_column: Column name containing predicted label IDs (default: 'predicted_label_id')
            confidence_column: Column name containing confidence scores (default: 'confidence')
            include_probabilities: Whether to include probability columns for ROC AUC calculation

        Returns:
            EvaluationMetrics object with comprehensive metrics

        Raises:
            ValueError: If required columns are missing
        """
        self._log("Starting evaluation...")

        # Validate required columns
        required_cols = [true_label_column, predicted_label_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

        # Extract predictions and true labels
        y_true = df[true_label_column].values
        y_pred = df[predicted_label_column].values

        # Get unique labels for label list
        unique_labels = sorted(df[true_label_column].unique())
        label_names = [str(label) for label in unique_labels]

        # Extract probability matrix if available
        probabilities = None
        if include_probabilities:
            prob_cols = [col for col in df.columns if col.startswith('prob_')]
            if prob_cols:
                self._log(f"Found {len(prob_cols)} probability columns")
                # Build probability matrix
                # Need to map prob columns to label order
                prob_matrix = []
                for label in label_names:
                    # Try to find matching prob column
                    prob_col = f"prob_{label.replace(' ', '_').replace('&', 'and')}"
                    if prob_col in df.columns:
                        prob_matrix.append(df[prob_col].values)
                    else:
                        self._log(f"Warning: Could not find probability column for label '{label}'")

                if len(prob_matrix) == len(label_names):
                    probabilities = np.column_stack(prob_matrix)
                    self._log(f"Using probability matrix of shape {probabilities.shape} for ROC AUC calculation")
                else:
                    self._log("Warning: Probability columns don't match all labels. Skipping ROC AUC calculation.")

        # Calculate metrics
        self._log("Calculating metrics...")
        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=label_names,
            probabilities=probabilities
        )

        # Store for later use
        self.last_metrics = metrics

        self._log("Evaluation complete!")
        return metrics

    def evaluate_from_inference(
        self,
        df: pd.DataFrame,
        true_label_column: str
    ) -> EvaluationMetrics:
        """
        Convenience method for evaluating InferenceEngine output.

        Automatically detects standard InferenceEngine column names.

        Args:
            df: DataFrame with InferenceEngine predictions
            true_label_column: Column name containing true labels

        Returns:
            EvaluationMetrics object
        """
        return self.evaluate(
            df=df,
            true_label_column=true_label_column,
            predicted_label_column='predicted_label',
            predicted_id_column='predicted_label_id',
            confidence_column='confidence',
            include_probabilities=True
        )

    def get_misclassified_samples(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        predicted_label_column: str = 'predicted_label',
        confidence_column: str = 'confidence',
        text_column: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get samples that were misclassified.

        Args:
            df: DataFrame with predictions
            true_label_column: Column name containing true labels
            predicted_label_column: Column name containing predicted labels
            confidence_column: Column name containing confidence scores
            text_column: Optional text column to include in output
            top_n: Return only top N misclassified samples (sorted by confidence, descending)

        Returns:
            DataFrame with misclassified samples
        """
        # Find misclassified samples
        misclassified = df[df[true_label_column] != df[predicted_label_column]].copy()

        # Select relevant columns
        cols = [true_label_column, predicted_label_column, confidence_column]
        if text_column and text_column in df.columns:
            cols.insert(0, text_column)

        result = misclassified[cols]

        # Sort by confidence (descending) - high confidence mistakes are more interesting
        if confidence_column in result.columns:
            result = result.sort_values(confidence_column, ascending=False)

        # Limit to top N if specified
        if top_n is not None:
            result = result.head(top_n)

        self._log(f"Found {len(misclassified)} misclassified samples (returning {len(result)})")

        return result

    def analyze_confidence(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        predicted_label_column: str = 'predicted_label',
        confidence_column: str = 'confidence',
        bins: int = 10
    ) -> pd.DataFrame:
        """
        Analyze model confidence vs accuracy.

        Args:
            df: DataFrame with predictions
            true_label_column: Column name containing true labels
            predicted_label_column: Column name containing predicted labels
            confidence_column: Column name containing confidence scores
            bins: Number of bins for confidence grouping

        Returns:
            DataFrame with confidence analysis
        """
        df = df.copy()
        df['correct'] = (df[true_label_column] == df[predicted_label_column]).astype(int)

        # Create confidence bins
        df['confidence_bin'] = pd.cut(df[confidence_column], bins=bins)

        # Analyze per bin
        analysis = df.groupby('confidence_bin', observed=True).agg({
            'correct': ['count', 'sum', 'mean'],
            confidence_column: 'mean'
        }).round(4)

        analysis.columns = ['total_samples', 'correct_predictions', 'accuracy', 'avg_confidence']
        analysis = analysis.reset_index()

        self._log("Confidence analysis complete!")
        return analysis
