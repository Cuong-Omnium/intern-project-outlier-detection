"""
Unit tests for data filtering module.
"""

import pandas as pd
import pytest

from src.data.filters import FilterError, apply_filters, get_filter_summary


class TestApplyFilters:
    """Tests for apply_filters function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "Account": ["A1", "A2", "A3", "A4", "A5"],
                "Lower": [1, 1, 0, 1, 1],
                "COT": ["Food", "Food", "Beverage", "Food", "Food"],
                "ACV > 10": [1, 1, 1, 0, 1],
                "Auto_Promo_Code": [0, 0, 0, 1, 0],
                "Unit_Sales": [100, 200, 0, 150, 300],
            }
        )

    def test_no_filters_returns_copy(self, sample_data):
        """Test that with no filters, a copy of data is returned."""
        # TODO: Call apply_filters with no parameters except data
        # TODO: Assert result has same length as input
        # TODO: Assert it's a different object (not same reference)
        pass

    def test_lower_filter(self, sample_data):
        """Test filtering by Lower column."""
        # TODO: Apply filter with lower=1
        # TODO: Assert only rows with Lower==1 remain
        # TODO: Assert correct number of rows (should be 4)
        pass

    def test_cot_filter(self, sample_data):
        """Test filtering by COT column."""
        # TODO: Apply filter with cot="Food"
        # TODO: Assert only "Food" rows remain
        pass

    def test_multiple_filters(self, sample_data):
        """Test combining multiple filters."""
        # TODO: Apply filters: lower=1, cot="Food", auto_promo_code=0
        # TODO: Assert correct rows remain
        # Hint: Should be A1, A2, A5 (3 rows)
        pass

    def test_exclude_zero_sales(self, sample_data):
        """Test excluding zero sales."""
        # TODO: Apply with exclude_zero_sales=True
        # TODO: Assert no rows with Unit_Sales==0
        pass

    def test_all_filters_combined(self, sample_data):
        """Test all filters together."""
        # TODO: Apply all filters
        # TODO: Should result in specific subset
        pass

    def test_missing_column_raises_error(self, sample_data):
        """Test that filtering on missing column raises FilterError."""
        # TODO: Drop a column
        # TODO: Try to filter on that column
        # TODO: Assert FilterError is raised
        pass

    def test_all_data_filtered_raises_error(self, sample_data):
        """Test that filtering out all data raises FilterError."""
        # TODO: Apply impossible filter (e.g., Lower=999)
        # TODO: Assert FilterError with "No data remains"
        pass


class TestGetFilterSummary:
    """Tests for get_filter_summary function."""

    def test_filter_summary_structure(self):
        """Test that summary has correct structure."""
        # TODO: Create original data (10 rows)
        # TODO: Create filtered data (5 rows)
        # TODO: Call get_filter_summary
        # TODO: Assert summary has expected keys and values
        pass
