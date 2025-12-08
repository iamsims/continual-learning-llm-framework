"""
Embedding Drift Detection
=========================
Semantic embedding-based drift detection using Evidently + Sentence Transformers.

How it works:
- Converts texts to dense semantic vectors using pre-trained models
- Compares embedding distributions using statistical tests (MMD, etc.)
- More semantically aware than TF-IDF (e.g., "great" and "excellent" are similar)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from evidently.legacy.report import Report
from evidently.legacy.metrics.data_drift.embeddings_drift import EmbeddingsDriftMetric
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    MMDDriftMethod,
    DistanceDriftMethod,
    RatioDriftMethod
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping

from sentence_transformers import SentenceTransformer

from .structures import MethodInput, MethodResult


class EmbeddingDriftDetector:
    """
    Embedding-based drift detection.

    Uses sentence transformers to create semantic embeddings and
    statistical tests to detect distribution shifts.
    """

    def __init__(
        self,
        column_name: str = 'text',
        report_dir: str = 'drift_reports',
        model_name: str = 'all-MiniLM-L6-v2',
        drift_method: str = 'mmd',
        use_bootstrap: bool = True,
        quantile_probability: float = 0.01,
        verbose: bool = True
    ):
        """
        Initialize embedding drift detector.

        Args:
            column_name: Name of the text column in dataframes
            report_dir: Directory to save HTML reports
            model_name: Sentence transformer model to use
                - 'all-MiniLM-L6-v2': Fast, lightweight (default)
                - 'all-mpnet-base-v2': More accurate but slower
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual support
            drift_method: Method to detect drift ('mmd', 'euclidean', 'cosine', 'ratio')
            use_bootstrap: Whether to use bootstrap for threshold calculation (default: True)
                - True: Calculate threshold from reference data (95th percentile)
                - False: Use hardcoded threshold (0.015 for MMD, 0.2 for distance)
            quantile_probability: Probability for bootstrap quantile (default: 0.05 = 95th percentile)
                Only used when use_bootstrap=True
            verbose: Whether to print progress messages
        """
        self.column_name = column_name
        self.report_dir = Path(report_dir)
        self.model_name = model_name
        self.drift_method = drift_method
        self.use_bootstrap = use_bootstrap
        self.quantile_probability = quantile_probability
        self.verbose = verbose

        # Create report directory
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Cache for embedding model
        self._model: Optional[SentenceTransformer] = None

        # Cache for reference data
        self._reference_texts: Optional[List[str]] = None
        self._reference_embeddings: Optional[np.ndarray] = None

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _get_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if self._model is None:
            self._log(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, reference_texts: List[str]) -> 'EmbeddingDriftDetector':
        """
        Fit detector on reference data by pre-computing embeddings.

        Args:
            reference_texts: Reference dataset texts

        Returns:
            self (for method chaining)
        """
        self._log("Fitting embedding detector on reference data...")

        # Store reference texts
        self._reference_texts = reference_texts

        # Pre-compute embeddings
        model = self._get_model()
        self._log("Pre-computing embeddings for reference data...")
        self._reference_embeddings = model.encode(
            reference_texts,
            show_progress_bar=self.verbose,
            convert_to_numpy=True
        )
        self._log(f"✓ Cached embeddings shape: {self._reference_embeddings.shape}")
        self._log("✓ Fitting complete!")

        return self

    def detect(
        self, 
        input: MethodInput,
    ) -> Dict:
        """
        Detect drift using embeddings.

        Args:
            current_texts: Current dataset texts
            reference_texts: Reference dataset texts (optional if fit() was called)
            report_filename: Filename for HTML report

        Returns:
            Dictionary with drift results:
                - drift_detected: bool
                - drift_score: float
                - method: str
                - model: str
                - embedding_dim: int
                - report_path: str
                - used_cache: bool
        """
        model = self._get_model()


        ref_embeddings = self._reference_embeddings
        reference_texts = self._reference_texts
        
        current_texts = input.current_data
        report_filename = input.report_filename

        # Generate embeddings for current data
        self._log("Generating embeddings for current data...")
        curr_embeddings = model.encode(
            current_texts,
            show_progress_bar=self.verbose,
            convert_to_numpy=True
        )

        self._log(f"Reference embeddings shape: {ref_embeddings.shape}")
        self._log(f"Current embeddings shape: {curr_embeddings.shape}")

        # Create dataframes with embeddings as separate columns
        emb_dim = ref_embeddings.shape[1]
        embedding_column_names = [f'{self.column_name}_emb_{i}' for i in range(emb_dim)]

        ref_df_emb = pd.DataFrame(ref_embeddings, columns=embedding_column_names)
        ref_df_emb[self.column_name] = reference_texts

        curr_df_emb = pd.DataFrame(curr_embeddings, columns=embedding_column_names)
        curr_df_emb[self.column_name] = current_texts

        # Select drift method with bootstrap configuration
        if self.drift_method.lower() == 'mmd':
            drift_method_obj = MMDDriftMethod(
                bootstrap=self.use_bootstrap,
                quantile_probability=self.quantile_probability
            )
            method_name = 'MMD (Maximum Mean Discrepancy)'
        elif self.drift_method.lower() in ['euclidean', 'cosine']:
            drift_method_obj = DistanceDriftMethod(
                distance=self.drift_method.lower(),
                bootstrap=self.use_bootstrap,
                quantile_probability=self.quantile_probability
            )
            method_name = f'Distance-based ({self.drift_method})'
        elif self.drift_method.lower() == 'ratio':
            drift_method_obj = RatioDriftMethod(
                bootstrap=self.use_bootstrap,
                quantile_probability=self.quantile_probability
            )
            method_name = 'Ratio-based'
        else:
            drift_method_obj = MMDDriftMethod(
                bootstrap=self.use_bootstrap,
                quantile_probability=self.quantile_probability
            )
            method_name = 'MMD (Maximum Mean Discrepancy) [default]'

        bootstrap_info = f" (bootstrap: {self.use_bootstrap})" if self.use_bootstrap else " (static threshold)"
        self._log(f"Calculating embedding drift using {method_name}{bootstrap_info}...")

        # Create column mapping
        column_mapping = ColumnMapping(
            embeddings={self.column_name: embedding_column_names}
        )

        report = Report(metrics=[
            EmbeddingsDriftMetric(
                embeddings_name=self.column_name,
                drift_method=drift_method_obj
            ),
        ])

        report.run(
            reference_data=ref_df_emb,
            current_data=curr_df_emb,
            column_mapping=column_mapping
        )

        report_path = self.report_dir / report_filename
        report.save_html(str(report_path))
        self._log(f"✓ Report saved to {report_path}")

        # Extract results
        result = report.as_dict()['metrics'][0]['result']

        return MethodResult(
            method="embedding",
            drift_detected=result.get('drift_detected', False),
            details=report.as_dict()
        )
    

