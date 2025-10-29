"""
Residual aggregation for outlier detection.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)


class ResidualAggregator:
    """
    Aggregates residuals by account with weighted statistics.

    Calculates per-account summary statistics from residuals:
    - Weighted mean
    - Weighted standard deviation
    - Weighted skewness
    - Median absolute deviation (MAD)
    - Z-score of mean

    Example:
        >>> aggregator = ResidualAggregator()
        >>> agg_df = aggregator.aggregate(accounts, residuals, weights)
        >>> print(agg_df.head())
    """

    def __init__(self):
        """Initialize aggregator."""
        pass

    def aggregate(
        self,
        accounts: np.ndarray,
        residuals: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Aggregate residuals by account.

        Args:
            accounts: Array of account identifiers
            residuals: Array of residual values
            weights: Optional array of weights (e.g., sales volume)

        Returns:
            DataFrame with one row per account and columns:
                - Account
                - w_mean_residual: Weighted mean
                - w_std_residual: Weighted standard deviation
                - w_skew_residual: Weighted skewness
                - mad_residual: Median absolute deviation
                - z_mean_residual: Z-score of weighted mean
                - count: Number of observations

        Example:
            >>> agg = aggregator.aggregate(
            ...     accounts=['A1', 'A1', 'A2'],
            ...     residuals=[0.1, 0.2, 0.5],
            ...     weights=[100, 200, 150]
            ... )
        """

        df = pd.DataFrame(
            {
                "Account": accounts,
                "Residual": residuals,
                "Weight": weights if weights is not None else np.ones_like(residuals),
            }
        )

        df["Weight"] = df["Weight"].fillna(1.0)

        df_grouped = df.groupby("Account")

        agg_df = df_grouped.apply(
            self._weighted_stats, include_groups=False
        ).reset_index()

        mean_of_means = agg_df["w_mean_residual"].mean()
        std_of_means = agg_df["w_mean_residual"].std(ddof=0)
        agg_df["z_mean_residual"] = (
            (agg_df["w_mean_residual"] - mean_of_means) / std_of_means
            if std_of_means > 0
            else 0
        )

        return agg_df

    def _weighted_stats(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted statistics for a single account.

        Args:
            group: DataFrame with columns: Residual, Weight

        Returns:
            Series with calculated statistics
        """
        residuals = group["Residual"].values
        weights = group["Weight"].values

        w_mean = np.average(residuals, weights=weights)

        w_var = np.average((residuals - w_mean) ** 2, weights=weights)
        w_std = np.sqrt(w_var)

        w_skew = (
            np.average((residuals - w_mean) ** 3, weights=weights) / (w_std**3)
            if w_std > 0
            else 0
        )

        mad = median_abs_deviation(residuals)

        return pd.Series(
            {
                "w_mean_residual": w_mean,
                "w_std_residual": w_std,
                "w_skew_residual": w_skew,
                "mad_residual": mad,
                "count": len(residuals),
            }
        )

    def get_feature_matrix(self, agg_df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix for outlier detection.

        Args:
            agg_df: Aggregated DataFrame from aggregate()

        Returns:
            Standardized feature matrix (numpy array)

        Features used:
            - w_mean_residual
            - z_mean_residual
            - w_std_residual
            - w_skew_residual

        Example:
            >>> X = aggregator.get_feature_matrix(agg_df)
            >>> X.shape
            (100, 4)  # 100 accounts, 4 features
        """

        feature_cols = [
            "w_mean_residual",
            "z_mean_residual",
            "w_std_residual",
            "w_skew_residual",
        ]

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(agg_df[feature_cols])

        return np.array(X)
