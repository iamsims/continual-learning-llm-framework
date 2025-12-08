from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from methods.domain_classifier import DomainClassifierDriftDetector
from methods.embedding import EmbeddingDriftDetector
from structures import DriftMethod, DriftSeverity
from methods.structures import MethodInput, MethodResult

@dataclass
class DriftDetectionInput:
    """
    Input data for drift detection.

    Attributes:
        batch_id: Unique identifier for the batch
        texts: List of text samples to check for drift
        time_start: Start timestamp of the batch window
        time_end: End timestamp of the batch window
    """
    batch_id: str
    texts: List[str]
    time_start: datetime
    time_end: datetime

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
            'drift_severity': self.drift_severity,
            'timestamp': self.timestamp.isoformat(),
            'time_start': self.time_start.isoformat(),
            'time_end': self.time_end.isoformat(),
            'method_results': {
                method: {
                    'method': result.method,
                    'drift_detected': result.drift_detected,
                    'drift_score': result.drift_score,
                    'severity': result.severity,
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
    reference_texts: Optional[List[str]] = None
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_drift_method: str = 'mmd'
    use_bootstrap: bool = True
    quantile_probability: float = 0.05
    report_dir: str = 'drift_reports'
    verbose: bool = True

class BatchDriftDetector:

    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize batch drift detector.

        Args:
            config: Configuration for drift detection
        """
        self.config = config

        self.detectors = {}

        for method in config.methods:
            if method == DriftMethod.DOMAIN_CLASSIFIER:
                self.detectors['domain_classifier'] = DomainClassifierDriftDetector(
                    column_name='text',
                    report_dir=config.report_dir,
                    verbose=config.verbose
                )
            elif method == DriftMethod.EMBEDDING:
                self.detectors['embedding'] = EmbeddingDriftDetector(
                    column_name='text',
                    report_dir=config.report_dir,
                    model_name=config.embedding_model,
                    drift_method=config.embedding_drift_method,
                    use_bootstrap=config.use_bootstrap,
                    quantile_probability=config.quantile_probability,
                    verbose=config.verbose
                )
        self._is_fitted = False

    def fit(
        self,
        reference_texts: List[str]
    ) -> 'BatchDriftDetector':
        """
        Fit detector on reference data.

        Pre-computes embeddings and caches reference data for faster detection.

        Args:
            reference_texts: Reference dataset texts

        Returns:
            self (for method chaining)
        """
        # Fit each detector on reference data
        for detector in self.detectors.values():
            detector.fit(reference_texts)

        self._is_fitted = True
        return self

    def detect(
        self,
        batch_id: str,
        texts: List[str],
        time_start: datetime,
        time_end: datetime,
    ) -> DriftDetectionOutput:
        """
        Detect drift on a batch using configured methods.

        Args:
            batch_id: Unique identifier for this batch
            texts: Current batch texts
            time_start: Start of batch time window
            time_end: End of batch time window
            reference_texts: Reference data (optional if fit() was called)

        Returns:
            DriftDetectionOutput with aggregated results
        """
        timestamp = datetime.utcnow()
        method_results = {}

        # Run each configured method
        for method_name, detector in self.detectors.items():
            input = MethodInput(
                current_data=texts,
                report_filename=f"{method_name}_{batch_id}_{time_start.strftime('%Y%m%d_%H%M%S')}.html"
            )
            method_results[method_name] = detector.detect(input)


        aggregated = self._aggregate_results(method_results)

        return DriftDetectionOutput(
            batch_id=batch_id,
            drift_detected=aggregated['drift_detected'],
            drift_severity=aggregated['severity'],
            timestamp=timestamp,
            time_start=time_start,
            time_end=time_end,
            method_results=method_results,
            aggregation_details=aggregated['details']
        )

    def _aggregate_results(self, method_results: Dict[str, MethodResult]) -> Dict[str, Any]:
        # Filter out text_overview (it's exploratory only)
        detection_methods = {
            name: result for name, result in method_results.items()
            if name != 'text_overview'
        }

        if not detection_methods:
            return {
                'drift_detected': False,
                'severity': DriftSeverity.NONE.value,
                'details': {'reason': 'No detection methods configured'}
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
            severity = DriftSeverity.NONE.value
        elif methods_detected_drift == 1:
            severity = DriftSeverity.LOW.value
        elif methods_detected_drift == total_methods:
            severity = DriftSeverity.HIGH.value
        else:  # more than one but not all 
            severity = DriftSeverity.MEDIUM.value

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
