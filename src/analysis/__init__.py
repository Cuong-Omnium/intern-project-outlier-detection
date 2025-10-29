"""
Analysis module for K-Fold cross-validation.
"""

from .config import KFoldConfig
from .kfold import KFoldAnalyzer
from .results import FoldResult, KFoldResult

__all__ = [
    "KFoldConfig",
    "KFoldResult",
    "FoldResult",
    "KFoldAnalyzer",
]
