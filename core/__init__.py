"""
Core module for thesis project.

Contains:
- inference: Inference engine for BERT-based text classification
- active_learning: LLM-based active learning for sample selection
- drift_detection: Text drift detection methods
- structures: Common data structures
"""

from . import inference
from . import active_learning
from . import drift_detection

__all__ = ['inference', 'active_learning', 'drift_detection']
