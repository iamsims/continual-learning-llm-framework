from enum import Enum


class OODSeverity(Enum):
    """OOD severity levels"""
    NONE = "none"           # No OOD samples detected
    LOW = "low"             # Small proportion of OOD samples
    MEDIUM = "medium"       # Moderate proportion of OOD samples
    HIGH = "high"           # Large proportion of OOD samples
