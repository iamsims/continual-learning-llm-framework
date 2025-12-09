from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from .methods.domain_classifier import DomainClassifierDriftDetector
from .methods.embedding import EmbeddingDriftDetector
from .structures import DriftMethod, DriftSeverity
from .methods.structures import MethodInput, MethodResult


@dataclass
class DriftDetectionOutput:
    """
    Aggregated output from drift detection.

    Attributes:
        batch_id: Batch identifier
        drift_detected: Overall drift detection (True if ANY method detects drift)
        drift_severity: Aggregated severity level
        timestamp: When detection was performed
        time_start: Start of batch window
        time_end: End of batch window
        method_results: Results from individual methods
        aggregation_details: How severity was determined
    """
    batch_id: str
    drift_detected: bool
    drift_severity: DriftSeverity
    timestamp: datetime
    time_start: datetime
    time_end: datetime
    method_results: Dict[str, MethodResult]
    aggregation_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'batch_id': self.batch_id,
            'drift_detected': self.drift_detected,
            'drift_severity': self.drift_severity.value,
            'timestamp': self.timestamp.isoformat(),
            'time_start': self.time_start.isoformat(),
            'time_end': self.time_end.isoformat(),
            'method_results': {
                method: {
                    'method': result.method,
                    'drift_detected': result.drift_detected,
                    'details': result.details
                }
                for method, result in self.method_results.items()
            },
            'aggregation_details': self.aggregation_details
        }


@dataclass
class DriftDetectionConfig:
    """
    Configuration for drift detection.

    Attributes:
        methods: List of drift detection methods to use
        text_column: Column name containing text data
        batch_id_column: Optional column name for batch IDs
        time_start_column: Optional column name for time start
        time_end_column: Optional column name for time end
        embedding_model: Sentence transformer model for embedding method
        embedding_drift_method: Statistical test for embedding drift
        use_bootstrap: Use bootstrap for adaptive threshold calculation
        quantile_probability: Bootstrap quantile (0.05 = 95th percentile)
        report_dir: Directory to save HTML reports
        verbose: Whether to print progress messages
    """
    methods: List[DriftMethod] = field(default_factory=lambda: [
        DriftMethod.DOMAIN_CLASSIFIER,
        DriftMethod.EMBEDDING,
    ])
    text_column: str = 'text'
    batch_id_column: Optional[str] = None
    time_start_column: Optional[str] = None
    time_end_column: Optional[str] = None
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_drift_method: str = 'mmd'
    use_bootstrap: bool = True
    quantile_probability: float = 0.05
    report_dir: str = 'drift_reports'
    verbose: bool = True


class DriftDetector:
    """
    Drift detector that works with pandas DataFrames.

    Supports multiple drift detection methods and seamlessly integrates
    with InferenceEngine output format.
    """

    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize drift detector.

        Args:
            config: Configuration for drift detection
        """
        self.config = config
        self.detectors = {}

        for method in config.methods:
            if method == DriftMethod.DOMAIN_CLASSIFIER:
                self.detectors['domain_classifier'] = DomainClassifierDriftDetector(
                    column_name=config.text_column,
                    report_dir=config.report_dir,
                    verbose=config.verbose
                )
            elif method == DriftMethod.EMBEDDING:
                self.detectors['embedding'] = EmbeddingDriftDetector(
                    column_name=config.text_column,
                    report_dir=config.report_dir,
                    model_name=config.embedding_model,
                    drift_method=config.embedding_drift_method,
                    use_bootstrap=config.use_bootstrap,
                    quantile_probability=config.quantile_probability,
                    verbose=config.verbose
                )
        self._is_fitted = False

    def fit(self, reference_df: pd.DataFrame) -> 'DriftDetector':
        """
        Fit detector on reference data.

        Pre-computes embeddings and caches reference data for faster detection.

        Args:
            reference_df: DataFrame with reference data
                         Must contain the configured text_column

        Returns:
            self (for method chaining)
        """
        # Validate required columns
        if self.config.text_column not in reference_df.columns:
            raise ValueError(
                f"Column '{self.config.text_column}' not found in reference DataFrame. "
                f"Available columns: {reference_df.columns.tolist()}"
            )

        # Extract texts
        reference_texts = reference_df[self.config.text_column].astype(str).tolist()

        if len(reference_texts) == 0:
            raise ValueError("Reference DataFrame cannot be empty")

        # Fit each detector on reference data
        for detector in self.detectors.values():
            detector.fit(reference_texts)

        if self.config.verbose:
            print(f"Fitted drift detector on {len(reference_texts)} reference samples")

        self._is_fitted = True
        return self

    def detect(
        self,
        df: pd.DataFrame,
        batch_id: Optional[str] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None
    ) -> DriftDetectionOutput:
        """
        Detect drift on a DataFrame.

        Args:
            df: DataFrame with current batch data
                Must contain the configured text_column
            batch_id: Optional batch identifier (used if batch_id_column is not configured)
            time_start: Optional start timestamp (used if time_start_column is not configured)
            time_end: Optional end timestamp (used if time_end_column is not configured)

        Returns:
            DriftDetectionOutput with aggregated results
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection. Call fit() first.")

        # Validate required columns
        if self.config.text_column not in df.columns:
            raise ValueError(
                f"Column '{self.config.text_column}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        timestamp = datetime.utcnow()

        # Get batch_id
        if self.config.batch_id_column and self.config.batch_id_column in df.columns:
            batch_ids = df[self.config.batch_id_column].values
            batch_id_value = batch_ids[0] if len(batch_ids) > 0 else "unknown"
        elif batch_id:
            batch_id_value = batch_id
        else:
            batch_id_value = f"batch_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Get time_start
        if self.config.time_start_column and self.config.time_start_column in df.columns:
            time_start_value = pd.to_datetime(df[self.config.time_start_column].iloc[0])
        elif time_start:
            time_start_value = time_start
        else:
            time_start_value = timestamp

        # Get time_end
        if self.config.time_end_column and self.config.time_end_column in df.columns:
            time_end_value = pd.to_datetime(df[self.config.time_end_column].iloc[-1])
        elif time_end:
            time_end_value = time_end
        else:
            time_end_value = timestamp

        # Extract texts
        texts = df[self.config.text_column].astype(str).tolist()

        # Run each configured method
        method_results = {}
        for method_name, detector in self.detectors.items():
            input_data = MethodInput(
                current_data=texts,
                report_filename=f"{method_name}_{batch_id_value}_{time_start_value.strftime('%Y%m%d_%H%M%S')}.html"
            )
            method_results[method_name] = detector.detect(input_data)

        # Aggregate results
        aggregated = self._aggregate_results(method_results)

        if self.config.verbose:
            print(f"\nDrift Detection Results:")
            print(f"  Batch ID: {batch_id_value}")
            print(f"  Total samples: {len(texts)}")
            print(f"  Drift detected: {aggregated['drift_detected']}")
            print(f"  Severity: {aggregated['severity']}")
            print(f"  Detection rate: {aggregated['details']['detection_rate']:.2%}")

        return DriftDetectionOutput(
            batch_id=batch_id_value,
            drift_detected=aggregated['drift_detected'],
            drift_severity=aggregated['severity'],
            timestamp=timestamp,
            time_start=time_start_value,
            time_end=time_end_value,
            method_results=method_results,
            aggregation_details=aggregated['details']
        )

    def _aggregate_results(self, method_results: Dict[str, MethodResult]) -> Dict[str, Any]:
        """
        Aggregate results from multiple drift detection methods.

        Args:
            method_results: Results from each detection method

        Returns:
            Dictionary with aggregated drift detection results
        """
        # Filter out text_overview (it's exploratory only)
        detection_methods = {
            name: result for name, result in method_results.items()
            if name != 'text_overview'
        }

        if not detection_methods:
            return {
                'drift_detected': False,
                'severity': DriftSeverity.NONE,
                'details': {'reason': 'No detection methods configured', 'detection_rate': 0.0}
            }

        # Count how many methods detected drift
        total_methods = len(detection_methods)
        methods_detected_drift = sum(
            1 for result in detection_methods.values()
            if result.drift_detected
        )

        # Determine overall drift detection
        drift_detected = methods_detected_drift > 0

        # Determine severity based on consensus
        if methods_detected_drift == 0:
            severity = DriftSeverity.NONE
        elif methods_detected_drift == 1:
            severity = DriftSeverity.LOW
        elif methods_detected_drift == total_methods:
            severity = DriftSeverity.HIGH
        else:  # more than one but not all
            severity = DriftSeverity.MEDIUM

        # Collect details
        details = {
            'total_methods': total_methods,
            'no_of_methods_detected_drift': methods_detected_drift,
            'detection_rate': methods_detected_drift / total_methods if total_methods > 0 else 0,
        }

        return {
            'drift_detected': drift_detected,
            'severity': severity,
            'details': details
        }
