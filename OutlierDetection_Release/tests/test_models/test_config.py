"""
Tests for regression configuration.
"""

import pandas as pd
import pytest

from src.models.config import RegressionConfig


class TestRegressionConfig:
    """Tests for RegressionConfig dataclass."""

    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = RegressionConfig(
            dependent_var="Unit_Sales",
            continuous_vars=["Auto_Base_Units", "ACV"],
            categorical_vars=["Account", "COT"],
        )

        assert config.dependent_var == "Unit_Sales"
        assert len(config.continuous_vars) == 2
        assert len(config.categorical_vars) == 2
        assert config.log_transform is True  # Default

    def test_empty_dependent_var_raises_error(self):
        """Test that empty dependent variable raises ValueError."""
        with pytest.raises(ValueError, match="dependent_var cannot be empty"):
            RegressionConfig(dependent_var="", continuous_vars=["X1"])

    def test_no_independent_vars_raises_error(self):
        """Test that no independent variables raises ValueError."""
        with pytest.raises(
            ValueError, match="Must specify at least one independent variable"
        ):
            RegressionConfig(dependent_var="Y")

    def test_duplicate_variables_raises_error(self):
        """Test that duplicate variables raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate variables"):
            RegressionConfig(
                dependent_var="Sales",
                continuous_vars=["Sales", "Units"],  # 'Sales' appears twice
                categorical_vars=["Account"],
            )

    def test_invalid_handle_zeros_raises_error(self):
        """Test that invalid handle_zeros strategy raises ValueError."""
        with pytest.raises(ValueError, match="handle_zeros must be"):
            RegressionConfig(
                dependent_var="Y",
                continuous_vars=["X"],
                handle_zeros="invalid_strategy",
            )

    def test_all_independent_vars_property(self):
        """Test the all_independent_vars property."""
        config = RegressionConfig(
            dependent_var="Y", continuous_vars=["X1", "X2"], categorical_vars=["Cat1"]
        )

        assert config.all_independent_vars == ["X1", "X2", "Cat1"]

    def test_all_vars_property(self):
        """Test the all_vars property."""
        config = RegressionConfig(
            dependent_var="Y", continuous_vars=["X1"], categorical_vars=["Cat1"]
        )

        assert config.all_vars == ["Y", "X1", "Cat1"]

    def test_validate_data_success(self):
        """Test that validation passes with correct data."""
        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units"],
            categorical_vars=["Account"],
        )

        df = pd.DataFrame(
            {"Sales": [100, 200], "Units": [10, 20], "Account": ["A1", "A2"]}
        )

        # Should not raise
        config.validate_data(df)

    def test_validate_data_missing_columns(self):
        """Test that validation fails with missing columns."""
        config = RegressionConfig(
            dependent_var="Sales",
            continuous_vars=["Units"],
            categorical_vars=["Account"],
        )

        df = pd.DataFrame(
            {
                "Sales": [100, 200],
                # Missing 'Units' and 'Account'
            }
        )

        with pytest.raises(ValueError, match="Missing required variables"):
            config.validate_data(df)
