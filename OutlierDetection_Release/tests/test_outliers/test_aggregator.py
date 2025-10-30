"""
Tests for residual aggregator.
"""

import numpy as np
import pandas as pd
import pytest

from src.outliers.aggregator import ResidualAggregator


class TestResidualAggregator:
    """Tests for ResidualAggregator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample residuals."""
        return {
            "accounts": np.array(["A1", "A1", "A1", "A2", "A2", "A3"]),
            "residuals": np.array([0.1, 0.2, 0.15, 0.5, 0.6, -0.3]),
            "weights": np.array([100, 200, 150, 300, 200, 100]),
        }

    def test_aggregate_with_weights(self, sample_data):
        """Test aggregation with weights."""

        aggregator = ResidualAggregator()

        agg_df = aggregator.aggregate(
            accounts=sample_data["accounts"],
            residuals=sample_data["residuals"],
            weights=sample_data["weights"],
        )

        assert agg_df.shape[0] == 3  # 3 unique accounts

        expected_columns = [
            "Account",
            "w_mean_residual",
            "w_std_residual",
            "w_skew_residual",
            "mad_residual",
            "count",
            "z_mean_residual",
        ]
        assert all(col in agg_df.columns for col in expected_columns)

        assert set(agg_df["Account"]) == {"A1", "A2", "A3"}
        pass

    def test_aggregate_without_weights(self, sample_data):
        """Test aggregation without weights (uniform)."""

        aggregator = ResidualAggregator()
        agg_df = aggregator.aggregate(
            accounts=sample_data["accounts"],
            residuals=sample_data["residuals"],
            weights=None,
        )

        assert agg_df.shape[0] == 3  # 3 unique accounts
        assert set(agg_df["Account"]) == {"A1", "A2", "A3"}
        expected_columns = [
            "Account",
            "w_mean_residual",
            "w_std_residual",
            "w_skew_residual",
            "mad_residual",
            "count",
            "z_mean_residual",
        ]
        assert all(col in agg_df.columns for col in expected_columns)
        pass

    def test_weighted_mean_calculation(self, sample_data):
        """Test weighted mean is calculated correctly."""

        aggregator = ResidualAggregator()
        agg_df = aggregator.aggregate(
            accounts=sample_data["accounts"],
            residuals=sample_data["residuals"],
            weights=sample_data["weights"],
        )

        A1_weighted_mean = np.average(
            sample_data["residuals"][:3], weights=sample_data["weights"][:3]
        )

        assert agg_df[agg_df["Account"] == "A1"]["w_mean_residual"].values[
            0
        ] == pytest.approx(A1_weighted_mean)
        pass

    def test_get_feature_matrix(self, sample_data):
        """Test feature matrix extraction."""

        aggregator = ResidualAggregator()
        agg_df = aggregator.aggregate(
            accounts=sample_data["accounts"],
            residuals=sample_data["residuals"],
            weights=sample_data["weights"],
        )

        feature_matrix = aggregator.get_feature_matrix(agg_df)

        assert feature_matrix.shape == (3, 4)

        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        assert all(abs(m) < 1e-6 for m in means)
        assert all(abs(s - 1) < 1e-6 for s in stds)
        pass
