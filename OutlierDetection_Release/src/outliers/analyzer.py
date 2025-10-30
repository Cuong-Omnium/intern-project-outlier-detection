"""
Main outlier analyzer that orchestrates the detection pipeline.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .aggregator import ResidualAggregator
from .config import OutlierConfig
from .detectors import (
    DBSCANDetector,
    DensityEnsemble,
    HDBSCANDetector,
    IsolationForestDetector,
)
from .results import OutlierResult

logger = logging.getLogger(__name__)


class OutlierAnalyzer:
    """
    Main class for detecting outlier accounts based on residuals.

    This analyzer:
    1. Aggregates residuals by account
    2. Runs multiple outlier detection algorithms
    3. Combines results using ensemble method
    4. Returns structured results

    Example:
        >>> config = OutlierConfig(contamination=0.10)
        >>> analyzer = OutlierAnalyzer(config)
        >>> result = analyzer.detect(accounts, residuals, weights)
        >>> print(f"Found {result.n_outliers} outliers")
    """

    def __init__(self, config: OutlierConfig):
        """
        Initialize analyzer with configuration.

        Args:
            config: OutlierConfig specifying detection parameters
        """
        self.config = config
        self.aggregator = ResidualAggregator()
        logger.info(f"Initialized OutlierAnalyzer with config: {config}")

    def detect(
        self,
        accounts: np.ndarray,
        residuals: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> OutlierResult:
        """
        Detect outlier accounts based on residuals.

        Args:
            accounts: Array of account identifiers
            residuals: Array of residual values from regression
            weights: Optional weights (e.g., sales volume)

        Returns:
            OutlierResult with complete analysis

        Example:
            >>> result = analyzer.detect(
            ...     accounts=np.array(['A1', 'A1', 'A2']),
            ...     residuals=np.array([0.1, 0.2, 5.0]),
            ...     weights=np.array([100, 200, 150])
            ... )
        """
        logger.info(
            f"Starting outlier detection on {len(np.unique(accounts))} accounts..."
        )

        # Step 1: Aggregate residuals by account
        logger.info("Step 1/4: Aggregating residuals by account")
        agg_df = self.aggregator.aggregate(accounts, residuals, weights)
        logger.info(f"Aggregated to {len(agg_df)} unique accounts")

        # Step 2: Extract feature matrix
        logger.info("Step 2/4: Extracting feature matrix")
        X = self.aggregator.get_feature_matrix(agg_df)
        logger.info(f"Feature matrix shape: {X.shape}")

        # Step 3: Run outlier detection algorithms
        logger.info("Step 3/4: Running outlier detection algorithms")

        # Density-based methods (DBSCAN/HDBSCAN)
        dbscan_detector = DBSCANDetector(
            k=self.config.k_for_eps,
            eps_percentile=self.config.eps_percentile,
            min_samples=self.config.dbscan_min_samples,
            show_plot=self.config.show_kdist_plot,
        )

        hdbscan_detector = HDBSCANDetector(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
        )

        density_ensemble = DensityEnsemble(
            dbscan_detector=dbscan_detector,
            hdbscan_detector=hdbscan_detector,
            use_dbcv=self.config.use_dbcv,
        )

        density_outliers, density_metadata = density_ensemble.fit_predict(X)

        # Isolation Forest
        iforest_detector = IsolationForestDetector(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )

        iforest_outliers = iforest_detector.fit_predict(X)

        # Step 4: Combine results
        logger.info("Step 4/4: Combining results")
        outlier_flags, outlier_types = self._combine_outliers(
            density_outliers, iforest_outliers, density_metadata["chosen_method"]
        )

        # Add outlier information to aggregated data
        agg_df["outlier_flag"] = outlier_flags
        agg_df["outlier_type"] = outlier_types

        # Create metadata
        metadata = {
            "density_method": density_metadata["chosen_method"],
            "dbscan_outliers": density_metadata["dbscan_outliers"].sum(),
            "hdbscan_outliers": density_metadata["hdbscan_outliers"].sum(),
            "iforest_outliers": iforest_outliers.sum(),
            "combined_outliers": outlier_flags.sum(),
            "dbcv_scores": density_metadata["dbcv_scores"],
            "contamination": self.config.contamination,
            "feature_names": [
                "w_mean_residual",
                "z_mean_residual",
                "w_std_residual",
                "w_skew_residual",
            ],
        }

        # Create result object
        result = OutlierResult(
            agg_data=agg_df,
            outlier_flags=outlier_flags,
            outlier_types=outlier_types,
            density_method=density_metadata["chosen_method"],
            metadata=metadata,
        )

        logger.info(
            f"Detection complete: {result.n_outliers} outliers "
            f"({result.outlier_rate*100:.1f}%)"
        )

        return result

    def _combine_outliers(
        self,
        density_outliers: np.ndarray,
        iforest_outliers: np.ndarray,
        density_method: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine outliers from multiple methods.

        Args:
            density_outliers: Outliers from density method
            iforest_outliers: Outliers from Isolation Forest
            density_method: Name of density method used

        Returns:
            Tuple of (combined_flags, outlier_types)
        """
        # Combined: flagged by any method
        combined_flags = ((density_outliers == 1) | (iforest_outliers == 1)).astype(int)

        # Classify outlier types
        outlier_types = np.where(
            (density_outliers == 1) & (iforest_outliers == 1),
            "both",
            np.where(
                density_outliers == 1,
                f"{density_method.lower()}_only",
                np.where(iforest_outliers == 1, "iforest_only", "normal"),
            ),
        )

        # Log breakdown
        type_counts = pd.Series(outlier_types).value_counts()
        logger.info(f"Outlier type breakdown:\n{type_counts}")

        return combined_flags, outlier_types
