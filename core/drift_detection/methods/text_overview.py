"""
Text Overview Drift Analysis
============================
Statistical text feature analysis using Evidently (exploratory only).

How it works:
- Analyzes text length distribution, word count, sentence count
- Computes sentiment, OOV words, non-letter character percentage
- Provides detailed HTML reports for human analysis
- NOT used for automated drift decisions (exploratory only)
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TextEvals


class TextOverviewDetector:
    """
    Text overview detector for exploratory analysis.

    Generates comprehensive reports analyzing text statistical features.
    Used for understanding what changed, not for automated drift detection.
    """

    def __init__(
        self,
        column_name: str = 'text',
        report_dir: str = 'drift_reports',
        verbose: bool = True
    ):
        """
        Initialize text overview detector.

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

    def fit(self, reference_texts: List[str]) -> 'TextOverviewDetector':
        """
        Fit detector on reference data.

        Note: Text overview only caches the reference texts. Creating DataFrames
        is trivial and doesn't warrant caching.

        Args:
            reference_texts: Reference dataset texts

        Returns:
            self (for method chaining)
        """
        self._log("Caching reference texts for text overview...")
        self._reference_texts = reference_texts
        self._log("✓ Fitting complete!")
        return self

    def generate_report(
        self,
        current_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        report_filename: str = "text_overview.html"
    ) -> Dict:
        """
        Generate comprehensive text overview report.

        Args:
            current_texts: Current dataset texts
            reference_texts: Reference dataset texts (optional if fit() was called)
            report_filename: Filename for HTML report

        Returns:
            Dictionary with report info:
                - report_path: str
                - status: str
                - used_cache: bool
                - note: str (reminder that this is exploratory)
        """
        # Use cached reference texts if available
        if reference_texts is None:
            if self._reference_texts is not None:
                self._log("Using cached reference texts...")
                reference_texts = self._reference_texts
                used_cache = True
            else:
                raise ValueError("reference_texts must be provided if fit() was not called.")
        else:
            used_cache = False

        # Create DataFrames (trivial operation)
        reference_df = pd.DataFrame({self.column_name: reference_texts})
        current_df = pd.DataFrame({self.column_name: current_texts})

        self._log("Generating text overview report...")
        report = Report(metrics=[
            DataDriftPreset(),
            TextEvals(column_name=self.column_name),
        ])

        report.run(reference_data=reference_df, current_data=current_df)

        report_path = self.report_dir / report_filename
        report.save_html(str(report_path))
        self._log(f"✓ Report saved to {report_path}")

        return {
            'report_path': str(report_path),
            'status': 'success',
            'used_cache': used_cache,
            'note': 'Text overview is exploratory - check HTML report for insights'
        }
