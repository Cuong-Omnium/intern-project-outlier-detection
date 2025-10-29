"""
Result classes for outlier detection.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OutlierResult:
    """
    Complete results from outlier detection.

    Attributes:
        agg_data: DataFrame with aggregated statistics per account
        outlier_flags: Array of outlier flags (1=outlier, 0=normal)
        outlier_types: Array of outlier type labels
        density_method: Which density method was chosen ('DBSCAN' or 'HDBSCAN')
        metadata: Dictionary with additional information

    Example:
        >>> result = OutlierResult(...)
        >>> print(f"Found {result.n_outliers} outliers")
        >>> outliers = result.get_outlier_accounts()
    """

    agg_data: pd.DataFrame
    outlier_flags: np.ndarray
    outlier_types: np.ndarray
    density_method: str
    metadata: dict

    @property
    def n_outliers(self) -> int:
        """Total number of outliers detected."""

        return np.sum(self.outlier_flags)

    @property
    def n_accounts(self) -> int:
        """Total number of accounts analyzed."""

        return self.agg_data.shape[0]

    @property
    def outlier_rate(self) -> float:
        """Proportion of accounts flagged as outliers."""

        return self.n_outliers / self.n_accounts if self.n_accounts > 0 else 0.0

    def get_outlier_accounts(self, outlier_type: Optional[str] = None) -> list:
        """
        Get list of outlier account identifiers.

        Args:
            outlier_type: Filter by type ('both', 'dbscan_hdbscan_only',
                         'iforest_only'). If None, return all outliers.

        Returns:
            List of account names
        """

        if outlier_type is not None:
            outliers = self.agg_data[self.outlier_types == outlier_type]["Account"]
        else:
            outliers = self.agg_data[self.outlier_flags == 1]["Account"]

        return outliers.tolist()

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of outliers with their statistics.

        Returns:
            DataFrame with outliers and their key metrics
        """

        outlier_df = self.agg_data[self.outlier_flags == 1].copy()

        outlier_df["Outlier_Type"] = self.outlier_types[self.outlier_flags == 1]

        outlier_df = outlier_df.sort_values(
            by="w_mean_residual", key=abs, ascending=False
        )
        return outlier_df

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return self.__dict__
