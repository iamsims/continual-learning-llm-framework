"""
Domain Classifier Drift Detection
==================================
TF-IDF + SGDClassifier based drift detection using Evidently.

How it works:
- Trains a binary classifier to distinguish reference vs current data
- Higher ROC-AUC = easier to distinguish = more drift
- Returns characteristic words for each dataset
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from evidently.legacy.report import Report
from evidently.legacy.metrics import DatasetDriftMetric
from evidently.legacy.metrics.data_drift.text_domain_classifier_drift_metric import TextDomainClassifierDriftMetric

from .structures import MethodResult, MethodInput


class DomainClassifierDriftDetector:
    """
    Domain classifier-based drift detection.

    Uses TF-IDF vectorization + SGDClassifier to detect if current data
    can be distinguished from reference data.
    """

    def __init__(
        self,
        column_name: str = 'text',
        report_dir: str = 'drift_reports',
        verbose: bool = True
    ):
        """
        Initialize domain classifier drift detector.

        Args:
            column_name: Name of the text column in dataframes
            report_dir: Directory to save HTML reports
            verbose: Whether to print progress messages
        """
        self.column_name = column_name
        self.report_dir = Path(report_dir)
        self.verbose = verbose

        # Create report directory
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Cache for reference data (just the texts, DataFrame creation is trivial)
        self._reference_texts: Optional[List[str]] = None

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def fit(self, reference_texts: List[str]):
        """
        Fit detector on reference data.

        Note: For domain classifier, we can only cache the reference texts.
        The actual classifier must be trained on each detect() call since it
        requires both reference and current data simultaneously.

        Args:
            reference_texts: Reference dataset texts

        Returns:
            self (for method chaining)
        """
        self._log("Caching reference texts for domain classifier...")
        self._reference_texts = reference_texts
        self._log("✓ Fitting complete!")
        return self

    def detect(
        self,
        texts: MethodInput,
    ) -> Dict:
        """
        Detect drift using domain classifier.

        Args:
            texts: MethodInput containing:
                - current_data: Current dataset texts
                - report_filename: Filename for saving the HTML report
            
        """
        # Use cached reference texts if available
       
        reference_df = pd.DataFrame({self.column_name: self._reference_texts})
        current_df = pd.DataFrame({self.column_name: texts.current_data})

        report_filename = texts.report_filename

        self._log("Training domain classifier...")
        report = Report(metrics=[
            TextDomainClassifierDriftMetric(text_column_name=self.column_name),
            DatasetDriftMetric(),
        ])

        report.run(reference_data=reference_df, current_data=current_df)

        report_path = self.report_dir / report_filename
        report.save_html(str(report_path))
        self._log(f"✓ Report saved to {report_path}")

        # Extract results
        result = report.as_dict()['metrics'][0]['result']

        return MethodResult(
            method="domain_classifier",
            drift_detected=result['content_drift'],
            details=report.as_dict()
        )