"""
Tests for outlier analyzer.
"""

import numpy as np
import pytest

from src.outliers.analyzer import OutlierAnalyzer
from src.outliers.config import OutlierConfig
from src.outliers.results import OutlierResult


class TestOutlierAnalyzer:
    """Tests for OutlierAnalyzer."""

    @pytest.fixture
    def sample_residuals(self):
        """Create sample residual data."""
        np.random.seed(42)
        n_accounts = 50
        obs_per_account = 20

        accounts = np.repeat([f"A{i}" for i in range(n_accounts)], obs_per_account)

        # Most accounts have small residuals
        residuals = np.random.normal(0, 0.1, n_accounts * obs_per_account)

        # Add a few outlier accounts (large residuals)
        outlier_indices = np.random.choice(n_accounts, 5, replace=False)
        for idx in outlier_indices:
            start = idx * obs_per_account
            end = start + obs_per_account
            residuals[start:end] += np.random.choice([3, -3])  # Large bias

        weights = np.random.randint(10, 100, n_accounts * obs_per_account)

        return accounts, residuals, weights

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        config = OutlierConfig(contamination=0.10)
        analyzer = OutlierAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.aggregator is not None

    def test_detect_finds_outliers(self, sample_residuals):
        """Test that detect method finds outliers."""
        accounts, residuals, weights = sample_residuals

        config = OutlierConfig(
            contamination=0.10,
            use_dbcv=False,  # Faster for testing
            show_kdist_plot=False,
        )
        analyzer = OutlierAnalyzer(config)

        result = analyzer.detect(accounts, residuals, weights)

        assert isinstance(result, OutlierResult)
        assert result.n_outliers > 0
        assert result.n_outliers < result.n_accounts
        assert 0 <= result.outlier_rate <= 1

    def test_detect_without_weights(self, sample_residuals):
        """Test detection works without weights."""
        accounts, residuals, _ = sample_residuals

        config = OutlierConfig(contamination=0.10, use_dbcv=False)
        analyzer = OutlierAnalyzer(config)

        result = analyzer.detect(accounts, residuals, weights=None)

        assert isinstance(result, OutlierResult)
        assert result.n_accounts == 50

    def test_result_structure(self, sample_residuals):
        """Test result has expected structure."""
        accounts, residuals, weights = sample_residuals

        config = OutlierConfig(contamination=0.10, use_dbcv=False)
        analyzer = OutlierAnalyzer(config)

        result = analyzer.detect(accounts, residuals, weights)

        # Check result properties
        assert hasattr(result, "agg_data")
        assert hasattr(result, "outlier_flags")
        assert hasattr(result, "outlier_types")
        assert hasattr(result, "density_method")
        assert hasattr(result, "metadata")

        # Check agg_data has required columns
        assert "Account" in result.agg_data.columns
        assert "outlier_flag" in result.agg_data.columns
        assert "outlier_type" in result.agg_data.columns

    def test_get_outlier_accounts(self, sample_residuals):
        """Test retrieving outlier account list."""
        accounts, residuals, weights = sample_residuals

        config = OutlierConfig(contamination=0.15, use_dbcv=False)
        analyzer = OutlierAnalyzer(config)

        result = analyzer.detect(accounts, residuals, weights)

        outlier_accounts = result.get_outlier_accounts()

        assert isinstance(outlier_accounts, list)
        assert len(outlier_accounts) == result.n_outliers
        assert all(isinstance(acc, str) for acc in outlier_accounts)

    def test_get_outlier_summary(self, sample_residuals):
        """Test outlier summary DataFrame."""
        accounts, residuals, weights = sample_residuals

        config = OutlierConfig(contamination=0.10, use_dbcv=False)
        analyzer = OutlierAnalyzer(config)

        result = analyzer.detect(accounts, residuals, weights)

        summary = result.get_outlier_summary()

        assert len(summary) == result.n_outliers
        assert "Outlier_Type" in summary.columns
        assert "w_mean_residual" in summary.columns
