from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List
from dataclasses import field

class DriftMethod(Enum):
    """Available drift detection methods"""
    DOMAIN_CLASSIFIER = "domain_classifier"
    EMBEDDING = "embedding"


class DriftSeverity(Enum):
    """Drift severity levels based on multi-method aggregation"""
    NONE = "none"           # No drift detected by any method
    LOW = "low"             # Drift detected by only 1 method
    MEDIUM = "medium"       # Drift detected by majority of methods
    HIGH = "high"           # Drift detected by all methods

