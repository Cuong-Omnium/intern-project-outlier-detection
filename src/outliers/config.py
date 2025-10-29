"""
Configuration for outlier detection.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OutlierConfig:
    """
    Configuration for outlier detection algorithms.

    Attributes:
        contamination: Expected proportion of outliers (for Isolation Forest)
        k_for_eps: K value for k-distance plot (DBSCAN eps selection)
        eps_percentile: Percentile of k-distances to use as eps
        dbscan_min_samples: Minimum samples for DBSCAN core point
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN
        hdbscan_min_samples: Minimum samples for HDBSCAN core point
        use_dbcv: Whether to use DBCV for density method selection
        show_kdist_plot: Whether to show k-distance plot
        random_state: Random seed for reproducibility

    Example:
        >>> config = OutlierConfig(
        ...     contamination=0.10,
        ...     eps_percentile=90,
        ...     use_dbcv=True
        ... )
    """

    contamination: float = 0.10
    k_for_eps: int = 5
    eps_percentile: int = 90
    dbscan_min_samples: int = 5
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    use_dbcv: bool = True
    show_kdist_plot: bool = False
    random_state: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.contamination < 1:
            raise ValueError("contamination must be between 0 and 1")

        if self.k_for_eps < 2:
            raise ValueError("k_for_eps must be at least 2")

        if not 0 < self.eps_percentile <= 100:
            raise ValueError("eps_percentile must be between 0 and 100")

        if self.dbscan_min_samples < 1:
            raise ValueError("dbscan_min_samples must be at least 1")

        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size must be at least 2")

        logger.debug(f"Created OutlierConfig: {self}")
