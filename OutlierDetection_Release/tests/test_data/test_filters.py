"""
Unit tests for data filtering module.
"""

import re

import pandas as pd
import pytest

from src.data.filters import FilterError, apply_filters, get_filter_summary


class TestFlexibleColumnNames:
    """Tests for flexible column name matching."""

    def test_underscores_in_column_names(self):
        """Test filtering works with underscore column names."""
        data = pd.DataFrame(
            {
                "Account": ["A1", "A2", "A3"],
                "Lower": [1, 0, 1],
                "Unit_Sales": [100, 0, 200],  # Note: underscore
            }
        )

        result = apply_filters(data, lower=1, exclude_zero_sales=True)
        assert len(result) == 2
        assert all(result["Unit_Sales"] != 0)

    def test_dots_in_column_names(self):
        """Test filtering works with dot column names."""
        data = pd.DataFrame(
            {
                "Account": ["A1", "A2", "A3"],
                "Lower": [1, 1, 0],
                "Auto.Promo.Code": [0, 1, 0],  # Note: dots
            }
        )

        result = apply_filters(data, auto_promo_code=0)
        assert len(result) == 2

    def test_spaces_in_column_names(self):
        """Test filtering works with space column names."""
        data = pd.DataFrame(
            {
                "Account": ["A1", "A2", "A3"],
                "Lower": [1, 1, 1],
                "Unit Sales": [100, 0, 200],  # Note: space
            }
        )

        result = apply_filters(data, exclude_zero_sales=True)
        assert len(result) == 2
        assert all(result["Unit Sales"] != 0)

    def test_mixed_column_formats(self):
        """Test filtering works with mixed naming conventions."""
        data = pd.DataFrame(
            {
                "Account": ["A1", "A2", "A3"],
                "Lower": [1, 1, 0],
                "COT": ["Food", "Beverage", "Food"],
                "ACV > 10": [1, 1, 0],
                "Auto.Promo.Code": [0, 0, 1],
                "Unit Sales": [100, 200, 0],
            }
        )

        result = apply_filters(
            data,
            lower=1,
            cot="Food",
            acv_threshold=1,
            auto_promo_code=0,
            exclude_zero_sales=True,
        )
        assert len(result) == 1
        assert result.iloc[0]["Account"] == "A1"

    def test_case_insensitive_matching(self):
        """Test that column matching is case-insensitive."""
        data = pd.DataFrame(
            {
                "Account": ["A1", "A2"],
                "lower": [1, 0],  # Note: lowercase
                "cot": ["Food", "Beverage"],  # Note: lowercase
            }
        )

        result = apply_filters(data, lower=1, cot="Food")
        assert len(result) == 1


class TestColumnFindingHelpers:
    """Tests for helper functions."""

    def test_normalize_column_name(self):
        """Test column name normalization."""
        from src.data.filters import normalize_column_name

        assert normalize_column_name("Unit Sales") == "unitsales"
        assert normalize_column_name("Unit_Sales") == "unitsales"
        assert normalize_column_name("Unit.Sales") == "unitsales"
        assert normalize_column_name("UNIT SALES") == "unitsales"
        assert normalize_column_name("ACV > 10") == "acv10"

    def test_build_column_map(self):
        """Test column map building."""
        from src.data.filters import build_column_map

        df = pd.DataFrame({"Unit Sales": [1], "Account Name": [2], "ACV > 10": [3]})

        col_map = build_column_map(df)
        assert col_map["unitsales"] == "Unit Sales"
        assert col_map["accountname"] == "Account Name"
        assert col_map["acv10"] == "ACV > 10"

    def test_find_column_success(self):
        """Test finding column with multiple name variations."""
        from src.data.filters import build_column_map, find_column

        df = pd.DataFrame({"Unit Sales": [1, 2, 3]})
        col_map = build_column_map(df)

        # Should find it with any variation
        assert find_column(df, col_map, "Unit_Sales") == "Unit Sales"
        assert find_column(df, col_map, "Unit.Sales") == "Unit Sales"
        assert find_column(df, col_map, "UnitSales") == "Unit Sales"

    def test_find_column_not_found(self):
        """Test that find_column raises error when column doesn't exist."""
        from src.data.filters import FilterError, build_column_map, find_column

        df = pd.DataFrame({"Account": [1, 2, 3]})
        col_map = build_column_map(df)

        with pytest.raises(FilterError, match="Column not found"):
            find_column(df, col_map, "NonExistent", "AlsoMissing")


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
                "ACV Weighted Distribution": [1, 1, 1, 0, 1],
                "Auto.Promo.Code": [0, 0, 0, 1, 0],
                "Unit Sales": [100, 200, 0, 150, 300],
            }
        )

    def test_no_filters_returns_copy(self, sample_data):
        """Test that with no filters, a copy of data is returned."""
        filtered_data = apply_filters(sample_data)
        assert filtered_data is not sample_data
        assert len(filtered_data) == len(sample_data)
        pass

    def test_lower_filter(self, sample_data):
        """Test filtering by Lower column."""
        filtered_data = apply_filters(sample_data, lower=1)
        assert all(filtered_data["Lower"] == 1)
        assert len(filtered_data) == 4
        pass

    def test_cot_filter(self, sample_data):
        """Test filtering by COT column."""
        filtered_data = apply_filters(sample_data, cot="Food")
        assert all(filtered_data["COT"] == "Food")
        pass

    def test_multiple_filters(self, sample_data):
        """Test combining multiple filters."""
        filtered_data = apply_filters(
            sample_data, lower=1, cot="Food", auto_promo_code=0
        )
        assert len(filtered_data) == 3
        assert set(filtered_data["Account"]) == {"A1", "A2", "A5"}
        pass

    def test_exclude_zero_sales(self, sample_data):
        """Test excluding zero sales."""
        filtered_data = apply_filters(sample_data, exclude_zero_sales=True)
        assert all(filtered_data["Unit Sales"] != 0)
        pass

    def test_all_filters_combined(self, sample_data):
        """Test all filters together."""
        filtered_data = apply_filters(
            sample_data,
            lower=1,
            cot="Food",
            acv_threshold=1,
            auto_promo_code=0,
            exclude_zero_sales=True,
        )
        assert len(filtered_data) == 3
        assert set(filtered_data["Account"]) == {"A1", "A2", "A5"}
        pass

    def test_missing_column_raises_error(self, sample_data):
        """Test that filtering on missing column raises FilterError."""
        dropped_data = sample_data.drop(columns=["COT"])
        with pytest.raises(
            FilterError,
            match=re.compile(
                re.escape(
                    "Column not found. Searched for: ('COT', 'Channel of Trade'). "
                    "Available columns: ['Account', 'Lower', 'ACV Weighted Distribution', 'Auto.Promo.Code', 'Unit Sales']"
                )
            ),
        ):
            apply_filters(dropped_data, cot="Food")
        pass

    def test_all_data_filtered_raises_error(self, sample_data):
        """Test that filtering out all data raises FilterError."""
        with pytest.raises(FilterError, match="No data remains after applying filters"):
            apply_filters(sample_data, lower=999)
        pass


class TestGetFilterSummary:
    """Tests for get_filter_summary function."""

    def test_filter_summary_structure(self):
        """Test that summary has correct structure."""
        # TODO: Create original data (10 rows)
        # TODO: Create filtered data (5 rows)
        # TODO: Call get_filter_summary
        # TODO: Assert summary has expected keys and values
        original_data = pd.DataFrame(
            {
                "Account": [
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "A6",
                    "A7",
                    "A8",
                    "A9",
                    "A10",
                ],
                "Lower": [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                "COT": [
                    "Food",
                    "Food",
                    "Beverage",
                    "Food",
                    "Food",
                    "Beverage",
                    "Food",
                    "Food",
                    "Beverage",
                    "Food",
                ],
                "ACV Weighted Distribution": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                "Auto.Promo.code": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                "Unit Sales": [100, 200, 0, 150, 300, 0, 250, 0, 400, 500],
            }
        )

        filtered_data = apply_filters(original_data, lower=1, cot="Food")
        summary = get_filter_summary(original_data, filtered_data)
        assert summary["rows_original"] == 10
        assert summary["rows_filtered"] == 5
        assert summary["rows_removed"] == 5
        assert summary["percent_removed"] == 50.0
        pass
