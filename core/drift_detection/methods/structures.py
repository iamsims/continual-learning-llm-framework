from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


@dataclass
class MethodInput:
    current_data: List[str]
    report_filename: str 
    

@dataclass
class MethodResult:
    """
    Result from a single drift detection method.

    Attributes:
        method: Name of the detection method
        drift_detected: Whether drift was detected
        drift_score: Numeric drift score
        severity: Method-specific severity
        details: Additional method-specific information
    """
    method: str
    drift_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)
