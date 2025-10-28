"""
Data filtering module for account outlier detection.

This module provides functions to apply business logic filters to sales data.
"""

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FilterError(Exception):
    """Raised when filtering operations fail."""

    pass


def normalize_column_name(name: str) -> str:
    """
    Normalize column name by converting to lowercase and removing special characters.

    This allows flexible matching of columns like:
    - "Unit Sales" → "unitsales"
    - "Unit_Sales" → "unitsales"
    - "Unit.Sales" → "unitsales"

    Args:
        name: Original column name

    Returns:
        Normalized column name (lowercase, alphanumeric only)

    Example:
        >>> normalize_column_name("Unit Sales")
        'unitsales'
        >>> normalize_column_name("ACV > 10")
        'acv10'
    """
    # Convert to lowercase and keep only alphanumeric characters
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized


def build_column_map(data: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping from normalized column names to actual column names.

    Args:
        data: DataFrame to map columns for

    Returns:
        Dictionary mapping normalized names to actual column names

    Example:
        >>> df = pd.DataFrame({"Unit Sales": [1], "Account Name": [2]})
        >>> build_column_map(df)
        {'unitsales': 'Unit Sales', 'accountname': 'Account Name'}
    """
    column_map = {}
    for col in data.columns:
        normalized = normalize_column_name(col)
        column_map[normalized] = col
    return column_map


def find_column(
    data: pd.DataFrame, column_map: dict[str, str], *possible_names: str
) -> str:
    """
    Find actual column name from possible variations.

    Args:
        data: DataFrame to search
        column_map: Mapping from normalized to actual column names
        possible_names: Possible column name variations

    Returns:
        Actual column name found in the DataFrame

    Raises:
        FilterError: If column not found

    Example:
        >>> find_column(df, col_map, "Unit Sales", "Unit_Sales", "UnitSales")
        'Unit Sales'  # Returns the actual column name in df
    """
    for name in possible_names:
        normalized = normalize_column_name(name)
        if normalized in column_map:
            actual_name = column_map[normalized]
            logger.debug(
                f"Found column '{actual_name}' (searched for: {possible_names})"
            )
            return actual_name

    # Not found - raise helpful error
    available_cols = list(data.columns)
    error_msg = (
        f"Column not found. Searched for: {possible_names}. "
        f"Available columns: {available_cols}"
    )
    logger.error(error_msg)
    raise FilterError(error_msg)


def apply_filters(
    data: pd.DataFrame,
    lower: Optional[int] = None,
    cot: Optional[str] = None,
    acv_threshold: Optional[int] = None,
    auto_promo_code: Optional[int] = None,
    exclude_zero_sales: bool = False,
) -> pd.DataFrame:
    """
    Apply business logic filters to sales data.

    Column names are matched flexibly (case-insensitive, ignoring spaces/special chars).
    For example, "Unit Sales", "Unit_Sales", and "UnitSales" are all recognized.

    Args:
        data: Input DataFrame with sales data
        lower: Value to filter 'Lower' column (e.g., 1). If None, no filter applied.
        cot: Channel of Trade value to filter on (e.g., "Food"). If None, no filter applied.
        acv_threshold: Filter for 'ACV' column. If None, no filter applied.
        auto_promo_code: Value to filter 'Auto_Promo_Code' column. If None, no filter applied.
        exclude_zero_sales: If True, exclude rows where 'Unit_Sales' == 0

    Returns:
        Filtered DataFrame

    Raises:
        FilterError: If required columns don't exist or filtering fails

    Example:
        >>> df = apply_filters(
        ...     data,
        ...     lower=1,
        ...     cot="Food",
        ...     acv_threshold=10,
        ...     auto_promo_code=0
        ... )
        >>> print(f"Filtered to {len(df)} rows")
    """
    # Make a copy to avoid modifying original
    filtered_data = data.copy()

    # Build column name mapping for flexible matching
    column_map = build_column_map(filtered_data)

    initial_count = len(filtered_data)
    logger.info(f"Initial row count: {initial_count}")

    # Apply 'Lower' filter
    if lower is not None:
        col_name = find_column(filtered_data, column_map, "Lower")
        filtered_data = filtered_data[filtered_data[col_name] == lower]
        logger.debug(f"Row count after '{col_name}' filter: {len(filtered_data)}")

    # Apply 'COT' (Channel of Trade) filter
    if cot is not None:
        col_name = find_column(filtered_data, column_map, "COT", "Channel of Trade")
        filtered_data = filtered_data[filtered_data[col_name] == cot]
        logger.debug(f"Row count after '{col_name}' filter: {len(filtered_data)}")

    # Apply 'ACV' filter
    if acv_threshold is not None:
        col_name = find_column(
            filtered_data,
            column_map,
            "ACV > 10",
            "ACV Weighted Distribution",
            "ACV_Weighted_Distribution",
            "ACV",
        )
        filtered_data = filtered_data[filtered_data[col_name] >= acv_threshold]
        logger.debug(f"Row count after '{col_name}' filter: {len(filtered_data)}")

    # Apply 'Auto_Promo_Code' filter
    if auto_promo_code is not None:
        col_name = find_column(
            filtered_data,
            column_map,
            "Auto_Promo_Code",
            "Auto Promo Code",
            "Auto.Promo.Code",
        )
        filtered_data = filtered_data[filtered_data[col_name] == auto_promo_code]
        logger.debug(f"Row count after '{col_name}' filter: {len(filtered_data)}")

    # Apply 'Unit_Sales' exclusion
    if exclude_zero_sales:
        col_name = find_column(
            filtered_data,
            column_map,
            "Unit_Sales",
            "Unit Sales",
            "Unit.Sales",
            "UnitSales",
        )
        filtered_data = filtered_data[filtered_data[col_name] != 0]
        logger.debug(
            f"Row count after excluding zero '{col_name}': {len(filtered_data)}"
        )

    # Check if any data remains
    if filtered_data.empty:
        raise FilterError("No data remains after applying filters")

    final_count = len(filtered_data)
    rows_removed = initial_count - final_count
    percent_removed = (rows_removed / initial_count * 100) if initial_count > 0 else 0
    logger.info(
        f"Filtering complete: {final_count} rows remain "
        f"({rows_removed} removed, {percent_removed:.1f}%)"
    )

    return filtered_data


def get_filter_summary(data: pd.DataFrame, filtered_data: pd.DataFrame) -> dict:
    """
    Generate summary statistics comparing original and filtered data.

    Args:
        data: Original DataFrame before filtering
        filtered_data: DataFrame after filtering

    Returns:
        Dictionary with summary statistics

    Example:
        >>> summary = get_filter_summary(original_df, filtered_df)
        >>> print(summary['rows_removed'])
        500
    """
    rows_original = len(data)
    rows_filtered = len(filtered_data)
    rows_removed = rows_original - rows_filtered
    percent_removed = (rows_removed / rows_original * 100) if rows_original > 0 else 0.0

    return {
        "rows_original": rows_original,
        "rows_filtered": rows_filtered,
        "rows_removed": rows_removed,
        "percent_removed": round(percent_removed, 2),
    }
