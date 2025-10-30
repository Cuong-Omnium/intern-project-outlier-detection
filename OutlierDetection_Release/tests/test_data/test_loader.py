"""
Unit tests for data loading module.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    DataLoadError,
    get_column_info,
    load_data,
    validate_required_columns,
)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_csv_success(self, tmp_path):
        """Test loading a valid CSV file."""
        # Arrange: Create a temporary CSV file
        csv_file = tmp_path / "test.csv"
        test_data = pd.DataFrame(
            {
                "Account": ["A1", "A2"],
                "Date": ["2024-01-01", "2024-01-02"],
                "Unit_Sales": [100, 200],
            }
        )
        test_data.to_csv(csv_file, index=False)

        # Act: Load the file
        result = load_data(csv_file)

        # Assert: Check it loaded correctly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["Account", "Date", "Unit_Sales"]
        assert result["Unit_Sales"].sum() == 300

    def test_load_excel_success(self, tmp_path):
        """Test loading a valid Excel file."""
        # Arrange
        excel_file = tmp_path / "test.xlsx"
        test_data = pd.DataFrame({"Account": ["A1", "A2"], "Sales": [100, 200]})
        test_data.to_excel(excel_file, index=False)

        # Act
        result = load_data(excel_file)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_file_not_found(self):
        """Test that missing file raises DataLoadError."""
        # Arrange
        fake_path = Path("nonexistent_file.csv")

        # Act & Assert
        with pytest.raises(DataLoadError, match="File not found"):
            load_data(fake_path)

    def test_load_unsupported_file_type(self, tmp_path):
        """Test that unsupported file type raises DataLoadError."""
        # Arrange
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some data")

        # Act & Assert
        with pytest.raises(DataLoadError, match="Unsupported file type"):
            load_data(txt_file)

    def test_load_empty_file(self, tmp_path):
        """Test that empty CSV raises DataLoadError."""
        # Arrange
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        # Act & Assert
        with pytest.raises(DataLoadError):
            load_data(empty_csv)


class TestValidateRequiredColumns:
    """Tests for validate_required_columns function."""

    def test_all_columns_present(self):
        """Test validation passes when all columns present."""
        # Arrange
        df = pd.DataFrame({"Account": [1], "Date": ["2024-01-01"], "Sales": [100]})
        required = ["Account", "Date", "Sales"]

        # Act & Assert (should not raise)
        validate_required_columns(df, required)

    def test_missing_columns_raises_error(self):
        """Test validation fails when columns missing."""
        # Arrange
        df = pd.DataFrame({"Account": [1]})
        required = ["Account", "Date", "Sales"]

        # Act & Assert
        with pytest.raises(DataLoadError, match="Missing required columns"):
            validate_required_columns(df, required)

    def test_empty_required_list(self):
        """Test validation passes with no requirements."""
        # Arrange
        df = pd.DataFrame({"Account": [1]})

        # Act & Assert (should not raise)
        validate_required_columns(df, [])


class TestGetColumnInfo:
    """Tests for get_column_info function."""

    def test_column_info_structure(self):
        """Test that column info has correct structure."""
        # Arrange
        df = pd.DataFrame({"Account": ["A1", "A2", None], "Sales": [100, 200, 300]})

        # Act
        info = get_column_info(df)

        # Assert
        assert "Account" in info
        assert "Sales" in info

        # Check Account column (has nulls)
        assert info["Account"]["dtype"] == "object"
        assert info["Account"]["null_count"] == 1
        assert info["Account"]["null_pct"] == pytest.approx(33.33, rel=0.1)
        assert info["Account"]["unique_count"] == 2  # A1, A2 (None doesn't count)

        # Check Sales column (no nulls)
        assert info["Sales"]["null_count"] == 0
        assert info["Sales"]["unique_count"] == 3
