"""
Tests for regression pipeline builder.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.config import RegressionConfig
from src.models.regression import RegressionPipelineBuilder


class TestRegressionPipelineBuilder:
    """Tests for RegressionPipelineBuilder."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Sales": np.random.randint(100, 1000, 100),
                "Units": np.random.randint(10, 100, 100),
                "Price": np.random.uniform(1, 10, 100),
                "Account": np.random.choice(["A1", "A2", "A3"], 100),
                "COT": np.random.choice(["Food", "Beverage"], 100),
            }
        )

    def test_build_pipeline(self, sample_data):
        """Test that pipeline is built successfully."""
        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units", "Price"],
            categorical_vars=["Account"],
        )

        builder = RegressionPipelineBuilder(config)
        pipeline = builder.build()

        assert isinstance(pipeline, Pipeline)
        assert "preprocessing" in pipeline.named_steps
        assert "regression" in pipeline.named_steps

    def test_prepare_data(self, sample_data):
        """Test data preparation (X/y split and transform)."""
        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units"],
            categorical_vars=["Account"],
        )

        builder = RegressionPipelineBuilder(config)
        X, y = builder.prepare_data(sample_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert list(X.columns) == ["Units", "Account"]
        assert y.name == "Log_Sales"

    def test_fit_predict_pipeline(self, sample_data):
        """Test end-to-end pipeline fitting and prediction."""
        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units", "Price"],
            categorical_vars=["Account"],
        )

        builder = RegressionPipelineBuilder(config)
        X, y = builder.prepare_data(sample_data)

        pipeline = builder.build()
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

    def test_handle_missing_values(self):
        """Test that missing values are handled appropriately."""
        data = pd.DataFrame(
            {
                "Sales": [100, 200, np.nan, 400],
                "Units": [10, np.nan, 30, 40],
                "Account": ["A1", "A2", "A3", "A1"],
            }
        )

        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units"],
            categorical_vars=["Account"],
            drop_na=True,
        )

        builder = RegressionPipelineBuilder(config)
        X, y = builder.prepare_data(data)

        # Should have dropped rows with NaN
        assert len(X) < 4
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_negative_dependent_var_raises_error(self):
        """Test that negative dependent variable raises error."""
        data = pd.DataFrame(
            {
                "Sales": [-100, 200, 300],
                "Units": [10, 20, 30],
                "Account": ["A1", "A2", "A3"],
            }
        )

        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units"],
            categorical_vars=["Account"],
        )

        builder = RegressionPipelineBuilder(config)

        with pytest.raises(ValueError, match="contains negative values"):
            builder.prepare_data(data)
