from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .structures import OODSeverity


@dataclass
class OODDetectionConfig:
    """
    Configuration for OOD detection.

    Attributes:
        alpha: Significance level for threshold calculation (default: 0.05)
               Threshold will be set at the alpha percentile of reference confidences
        text_column: Column name containing text data
        confidence_column: Column name containing confidence scores
        predicted_label_column: Column name containing predicted labels
        batch_id_column: Optional column name for batch IDs (will use batch_id parameter if not provided)
        verbose: Whether to print progress messages
    """
    alpha: float = 0.05
    text_column: str = 'text'
    confidence_column: str = 'confidence'
    predicted_label_column: str = 'predicted_label'
    batch_id_column: Optional[str] = None
    verbose: bool = True


class OODDetector:
    """
    OOD detection based on confidence thresholds.

    Uses the confidence scores from model predictions to identify
    out-of-distribution samples. Threshold is computed from reference
    data using a configurable alpha percentile.

    Works with pandas DataFrames for seamless integration with InferenceEngine.
    """

    def __init__(self, config: OODDetectionConfig):
        """
        Initialize OOD detector.

        Args:
            config: Configuration for OOD detection
        """
        self.config = config
        self.threshold: Optional[float] = None
        self._is_fitted = False

    def fit(self, reference_df: pd.DataFrame) -> 'OODDetector':
        """
        Fit detector on reference data.

        Computes the confidence threshold based on the alpha percentile
        of reference confidence scores.

        Args:
            reference_df: DataFrame with reference (in-distribution) data
                          Must contain the configured confidence_column

        Returns:
            self (for method chaining)
        """
        # Validate required columns
        if self.config.confidence_column not in reference_df.columns:
            raise ValueError(
                f"Column '{self.config.confidence_column}' not found in reference DataFrame. "
                f"Available columns: {reference_df.columns.tolist()}"
            )

        # Extract confidence scores
        reference_confidences = reference_df[self.config.confidence_column].values

        if len(reference_confidences) == 0:
            raise ValueError("Reference DataFrame cannot be empty")

        # Compute threshold at alpha percentile
        # Samples with confidence below this threshold are considered OOD
        self.threshold = float(np.percentile(reference_confidences, self.config.alpha * 100))

        if self.config.verbose:
            print(f"Fitted OOD detector:")
            print(f"  Alpha: {self.config.alpha}")
            print(f"  Threshold: {self.threshold:.4f}")
            print(f"  Reference samples: {len(reference_confidences)}")
            print(f"  Reference confidence - Mean: {np.mean(reference_confidences):.4f}, "
                  f"Std: {np.std(reference_confidences):.4f}")

        self._is_fitted = True
        return self

    def detect(
        self,
        df: pd.DataFrame,
        batch_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect OOD samples in a DataFrame.

        Args:
            df: DataFrame with prediction data
                Must contain configured text_column, confidence_column, and predicted_label_column
            batch_id: Optional batch identifier (used if batch_id_column is not configured)

        Returns:
            DataFrame with OOD detection results. Only returns rows detected as OOD with columns:
            - All original columns from input df
            - batch_id: Batch identifier
            - ood_confidence: Confidence score of the OOD sample
            - threshold: Threshold used for detection
            - ood_severity: Severity level for this detection run
            - timestamp: When detection was performed
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection. Call fit() first.")

        # Validate required columns
        required_cols = [
            self.config.text_column,
            self.config.confidence_column,
            self.config.predicted_label_column
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        timestamp = datetime.utcnow()

        # Get batch_id column or use parameter
        if self.config.batch_id_column and self.config.batch_id_column in df.columns:
            batch_ids = df[self.config.batch_id_column].values
        elif batch_id:
            batch_ids = [batch_id] * len(df)
        else:
            raise ValueError(
                "Either batch_id parameter must be provided or batch_id_column must be configured and present in DataFrame"
            )

        # Extract confidence scores
        confidences = df[self.config.confidence_column].values

        # Identify samples below threshold as OOD
        ood_mask = confidences < self.threshold
        ood_indices = np.where(ood_mask)[0]

        # Calculate metrics
        total_samples = len(df)
        ood_count = len(ood_indices)
        ood_ratio = ood_count / total_samples if total_samples > 0 else 0

        # Determine severity based on proportion of OOD samples
        if ood_ratio == 0:
            severity = OODSeverity.NONE
        elif ood_ratio < 0.1:  # Less than 10%
            severity = OODSeverity.LOW
        elif ood_ratio < 0.3:  # Less than 30%
            severity = OODSeverity.MEDIUM
        else:  # 30% or more
            severity = OODSeverity.HIGH

        if self.config.verbose:
            print(f"\nOOD Detection Results:")
            print(f"  Batch ID: {batch_id if batch_id else 'Multiple'}")
            print(f"  Total samples: {total_samples}")
            print(f"  OOD samples: {ood_count} ({ood_ratio*100:.2f}%)")
            print(f"  Severity: {severity.value}")
            print(f"  Threshold: {self.threshold:.4f}")
            if len(confidences) > 0:
                print(f"  Current batch - Mean confidence: {np.mean(confidences):.4f}, "
                      f"Min: {np.min(confidences):.4f}, Max: {np.max(confidences):.4f}")

        # Create output DataFrame with only OOD samples
        if ood_count == 0:
            # Return empty DataFrame with expected columns
            ood_df = pd.DataFrame(columns=list(df.columns) + [
                'batch_id', 'ood_confidence', 'threshold', 'ood_severity', 'timestamp'
            ])
        else:
            # Filter to only OOD samples
            ood_df = df.iloc[ood_indices].copy()

            # Add OOD-specific columns
            ood_df['batch_id'] = [batch_ids[i] for i in ood_indices]
            ood_df['ood_confidence'] = confidences[ood_indices]
            ood_df['threshold'] = self.threshold
            ood_df['ood_severity'] = severity.value
            ood_df['timestamp'] = timestamp.isoformat()

        # Store detection stats as metadata (accessible via attributes)
        ood_df.attrs['detection_stats'] = {
            'total_samples': total_samples,
            'ood_count': ood_count,
            'ood_ratio': ood_ratio,
            'ood_severity': severity.value,
            'threshold': self.threshold,
            'alpha': self.config.alpha,
            'mean_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
            'min_confidence': float(np.min(confidences)) if len(confidences) > 0 else 0.0,
            'max_confidence': float(np.max(confidences)) if len(confidences) > 0 else 0.0,
            'timestamp': timestamp.isoformat()
        }

        return ood_df

    def get_detection_summary(self, ood_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics from OOD detection results.

        Args:
            ood_df: Output DataFrame from detect() method

        Returns:
            Dictionary with detection statistics
        """
        if hasattr(ood_df, 'attrs') and 'detection_stats' in ood_df.attrs:
            return ood_df.attrs['detection_stats']
        else:
            return {
                'error': 'No detection stats available. DataFrame may not be from detect() method.'
            }
