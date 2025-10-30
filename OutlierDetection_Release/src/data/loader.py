"""
Data loading module for account outlier detection.

This module handles loading CSV/Excel files and performing basic validation.
"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

# Set up module-level logger
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data cannot be loaded or validated."""

    pass


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV or Excel file with validation.

    Args:
        filepath: Path to the data file (CSV or Excel)

    Returns:
        DataFrame containing the loaded data

    Raises:
        DataLoadError: If file cannot be loaded or is invalid

    Example:
        >>> df = load_data("data/sales.csv")
        >>> print(df.shape)
        (1000, 43)
    """
    # Convert string to Path object for better path handling
    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        error_msg = f"File not found: {filepath}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

    # Load based on file extension
    try:
        if filepath.suffix.lower() == ".csv":
            logger.info(f"Loading CSV file: {filepath}")
            data = pd.read_csv(filepath)
        elif filepath.suffix.lower() in [".xlsx", ".xls"]:
            logger.info(f"Loading Excel file: {filepath}")
            data = pd.read_excel(filepath)
        else:
            error_msg = f"Unsupported file type: {filepath.suffix}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)

    except pd.errors.EmptyDataError:
        error_msg = f"File is empty: {filepath}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    except Exception as e:
        error_msg = f"Failed to parse file: {e}"
        logger.error(error_msg)
        raise DataLoadError(error_msg) from e

    # Basic validation
    if data.empty:
        error_msg = "Loaded data is empty"
        logger.warning(error_msg)
        raise DataLoadError(error_msg)

    logger.info(f"Successfully loaded {len(data)} rows, {len(data.columns)} columns")
    return data


def validate_required_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that required columns exist in the dataframe.

    Args:
        data: DataFrame to validate
        required_columns: List of column names that must be present

    Raises:
        DataLoadError: If any required columns are missing

    Example:
        >>> validate_required_columns(df, ['Account', 'Date', 'Unit_Sales'])
    """
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        error_msg = f"Missing required columns: {sorted(missing_columns)}"
        logger.error(error_msg)
        logger.info(f"Available columns: {sorted(data.columns)}")
        raise DataLoadError(error_msg)

    logger.debug(f"All required columns present: {required_columns}")


def get_column_info(data: pd.DataFrame) -> dict:
    """
    Get summary information about dataframe columns.

    Args:
        data: DataFrame to analyze

    Returns:
        Dictionary with column information including types and null counts

    Example:
        >>> info = get_column_info(df)
        >>> print(info['Account'])
        {'dtype': 'object', 'null_count': 0, 'null_pct': 0.0}
    """
    column_info = {}

    for col in data.columns:
        null_count = data[col].isnull().sum()
        column_info[col] = {
            "dtype": str(data[col].dtype),
            "null_count": int(null_count),
            "null_pct": float(null_count / len(data) * 100),
            "unique_count": int(data[col].nunique()),
        }

    return column_info
