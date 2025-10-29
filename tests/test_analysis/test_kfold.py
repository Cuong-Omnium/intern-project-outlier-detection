"""
Tests for K-Fold cross-validation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.config import KFoldConfig
from src.analysis.kfold import KFoldAnalyzer
from src.analysis.results import KFoldResult


class TestKFoldConfig:
    """Tests for KFoldConfig."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = KFoldConfig(
            n_splits=5, group_column="Period", account_column="Account"
        )

        assert config.n_splits == 5
        assert config.shuffle is True
        assert config.random_state == 42

    def test_invalid_n_splits(self):
        """Test that n_splits < 2 raises error."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            KFoldConfig(n_splits=1, group_column="Period", account_column="Account")

    def test_empty_group_column(self):
        """Test that empty group_column raises error."""
        with pytest.raises(ValueError, match="group_column cannot be empty"):
            KFoldConfig(n_splits=5, group_column="", account_column="Account")

    def test_validate_data_success(self):
        """Test data validation with valid data."""
        config = KFoldConfig(
            n_splits=3,
            group_column="Period",
            account_column="Account",
            weight_column="Weight",
        )

        data = pd.DataFrame(
            {
                "Period": [1, 1, 2, 2, 3, 3],
                "Account": ["A1", "A2", "A1", "A2", "A1", "A2"],
                "Weight": [10, 20, 30, 40, 50, 60],
            }
        )

        # Should not raise
        config.validate_data(data)

    def test_validate_data_missing_columns(self):
        """Test validation fails with missing columns."""
        config = KFoldConfig(
            n_splits=3, group_column="Period", account_column="Account"
        )

        data = pd.DataFrame({"Period": [1, 2, 3]})  # Missing 'Account'

        with pytest.raises(ValueError, match="Missing required columns"):
            config.validate_data(data)

    def test_validate_data_not_enough_groups(self):
        """Test validation fails with insufficient groups."""
        config = KFoldConfig(
            n_splits=5, group_column="Period", account_column="Account"
        )

        # Only 3 groups, but need 5 for 5-fold
        data = pd.DataFrame(
            {
                "Period": [1, 1, 2, 2, 3, 3],
                "Account": ["A1", "A2", "A1", "A2", "A1", "A2"],
            }
        )

        with pytest.raises(ValueError, match="Not enough groups"):
            config.validate_data(data)


class TestKFoldAnalyzer:
    """Tests for KFoldAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 120

        return pd.DataFrame(
            {
                "Period": np.repeat(range(1, 7), 20),  # 6 periods, 20 samples each
                "Account": np.tile(["A1", "A2", "A3", "A4"], 30),
                "Weight": np.random.randint(10, 100, n_samples),
                "X1": np.random.randn(n_samples),
                "X2": np.random.randn(n_samples),
                "y": np.random.randn(n_samples),
            }
        )

    def test_run_kfold_basic(self, sample_data):
        """Test basic K-Fold execution."""
        config = KFoldConfig(
            n_splits=3, group_column="Period", account_column="Account"
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1", "X2"]]
        y = sample_data["y"]
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

        results = analyzer.run(sample_data, X, y, pipeline)

        assert isinstance(results, KFoldResult)
        assert results.n_folds == 3
        assert len(results.fold_results) == 3
        assert results.mean_mse > 0
        assert -1 <= results.mean_r2 <= 1

    def test_run_kfold_with_weights(self, sample_data):
        """Test K-Fold with weight column."""
        config = KFoldConfig(
            n_splits=3,
            group_column="Period",
            account_column="Account",
            weight_column="Weight",
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1", "X2"]]
        y = sample_data["y"]
        pipeline = Pipeline([("regressor", LinearRegression())])

        results = analyzer.run(sample_data, X, y, pipeline)

        # Check that weights were captured
        assert results.get_all_weights() is not None
        assert len(results.get_all_weights()) == len(sample_data)

    def test_fold_results_structure(self, sample_data):
        """Test that fold results have correct structure."""
        config = KFoldConfig(
            n_splits=3, group_column="Period", account_column="Account"
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1", "X2"]]
        y = sample_data["y"]
        pipeline = Pipeline([("regressor", LinearRegression())])

        results = analyzer.run(sample_data, X, y, pipeline)

        fold = results.fold_results[0]
        assert fold.fold_number == 1
        assert len(fold.predictions) == len(fold.actuals)
        assert len(fold.residuals) == len(fold.actuals)
        assert fold.n_train > 0
        assert fold.n_test > 0

    def test_get_fold_summary(self, sample_data):
        """Test fold summary DataFrame generation."""
        config = KFoldConfig(
            n_splits=3, group_column="Period", account_column="Account"
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1", "X2"]]
        y = sample_data["y"]
        pipeline = Pipeline([("regressor", LinearRegression())])

        results = analyzer.run(sample_data, X, y, pipeline)
        summary = results.get_fold_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        assert "fold" in summary.columns
        assert "mse" in summary.columns
        assert "r2" in summary.columns

    def test_to_dataframe(self, sample_data):
        """Test converting results to DataFrame."""
        config = KFoldConfig(
            n_splits=3,
            group_column="Period",
            account_column="Account",
            weight_column="Weight",
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1", "X2"]]
        y = sample_data["y"]
        pipeline = Pipeline([("regressor", LinearRegression())])

        results = analyzer.run(sample_data, X, y, pipeline)
        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
        assert "fold" in df.columns
        assert "account" in df.columns
        assert "actual" in df.columns
        assert "predicted" in df.columns
        assert "residual" in df.columns
        assert "weight" in df.columns


class TestFindOptimalK:
    """Tests for find_optimal_k method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset with many groups."""
        np.random.seed(42)
        n_samples = 200

        return pd.DataFrame(
            {
                "Period": np.repeat(range(1, 11), 20),  # 10 periods
                "Account": np.tile(["A1", "A2"], 100),
                "X1": np.random.randn(n_samples),
                "y": np.random.randn(n_samples),
            }
        )

    def test_find_optimal_k(self, sample_data):
        """Test finding optimal k value."""
        config = KFoldConfig(
            n_splits=5,  # Initial value (will be tested)
            group_column="Period",
            account_column="Account",
            verbose=False,
        )

        analyzer = KFoldAnalyzer(config)

        X = sample_data[["X1"]]
        y = sample_data["y"]
        pipeline = Pipeline([("regressor", LinearRegression())])

        optimal_k, results_df = analyzer.find_optimal_k(
            sample_data, X, y, pipeline, k_range=range(2, 6)
        )

        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= 5
        assert isinstance(results_df, pd.DataFrame)
        assert "k" in results_df.columns
        assert "mean_mse" in results_df.columns
        assert len(results_df) <= 4  # Max 4 values in range(2, 6)
