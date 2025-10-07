"""
Data filtering module for account outlier detection.

This module provides functions to apply business logic filters to sales data.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FilterError(Exception):
    """Raised when filtering operations fail."""

    pass


def apply_filters(
    data: pd.DataFrame,
    lower: Optional[int] = None,
    cot: Optional[str] = None,
    acv_threshold: Optional[int] = None,
    auto_promo_code: Optional[int] = None,
    exclude_zero_sales: bool = True,
) -> pd.DataFrame:
    """
    Apply business logic filters to sales data.

    Args:
        data: Input DataFrame with sales data
        lower: Value to filter 'Lower' column (e.g., 1). If None, no filter applied.
        cot: Channel of Trade value to filter on (e.g., "Food"). If None, no filter applied.
        acv_threshold: Filter for 'ACV Weighted Distribution' column. If None, no filter applied.
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
        ...     acv_threshold=1,
        ...     auto_promo_code=0
        ... )
        >>> print(f"Filtered to {len(df)} rows")
    """
    # TODO: Make a copy of the data to avoid modifying the original
    filtered_data = data.copy()

    # TODO: Log initial row count
    initial_count = len(filtered_data)
    logger.info(f"Initial row count: {initial_count}")

    # TODO: Apply 'Lower' filter if provided
    #   - Check if 'Lower' column exists
    #   - Filter if lower parameter is not None
    #   - Log how many rows remain

    if lower is not None:
        if "Lower" not in filtered_data.columns:
            raise FilterError("'Lower' column not found in data")
        filtered_data = filtered_data[filtered_data["Lower"] == lower]
        logger.info(f"Row count after 'Lower' filter: {len(filtered_data)}")

    # TODO: Apply 'COT' filter if provided
    #   - Check if 'COT' column exists
    #   - Filter if cot parameter is not None
    #   - Log how many rows remain

    if cot is not None:
        if "COT" not in filtered_data.columns:
            raise FilterError("'COT' column not found in data")
        filtered_data = filtered_data[filtered_data["COT"] == cot]
        logger.info(f"Row count after 'COT' filter: {len(filtered_data)}")

    # TODO: Apply 'ACV > 10' filter if provided
    #   - Check if column exists
    #   - Filter if acv_threshold parameter is not None
    #   - Log how many rows remain

    if acv_threshold is not None:
        if "ACV Weighted Distribution" not in filtered_data.columns:
            raise FilterError("'ACV Weighted Distribution' column not found in data")
        filtered_data = filtered_data[
            filtered_data["ACV Weighted Distribution"] >= acv_threshold
        ]
        logger.info(
            f"Row count after 'ACV Weighted Distribution' filter: {len(filtered_data)}"
        )

    # TODO: Apply 'Auto_Promo_Code' filter if provided
    #   - Check if column exists
    #   - Filter if auto_promo_code parameter is not None
    #   - Log how many rows remain

    if auto_promo_code is not None:
        if "Auto.Promo.Code" not in filtered_data.columns:
            raise FilterError("'Auto.Promo.Code' column not found in data")
        filtered_data = filtered_data[
            filtered_data["Auto.Promo.Code"] == auto_promo_code
        ]
        logger.info(f"Row count after 'Auto.Promo.Code' filter: {len(filtered_data)}")

    # TODO: Apply Unit_Sales exclusion if exclude_zero_sales is True
    #   - Check if 'Unit_Sales' column exists
    #   - Filter out zeros
    #   - Log how many rows remain

    if exclude_zero_sales is not None:
        if "Unit Sales" not in filtered_data.columns:
            raise FilterError("'Unit Sales' column not found in data")
        if exclude_zero_sales:
            filtered_data = filtered_data[filtered_data["Unit Sales"] != 0]
        logger.info(f"Row count after 'Unit Sales' filter: {len(filtered_data)}")

    # TODO: Check if any data remains after filtering
    #   - Raise FilterError if empty
    if filtered_data.empty:
        raise FilterError("No data remains after applying filters")

    # TODO: Log final row count
    final_count = len(filtered_data)
    logger.info(f"Final row count after all filters: {final_count}")

    # TODO: Return filtered data
    # Ensure the return type is always a DataFrame
    if not isinstance(filtered_data, pd.DataFrame):
        filtered_data = pd.DataFrame(filtered_data)
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
    # TODO: Calculate statistics
    #   - rows_original
    #   - rows_filtered
    #   - rows_removed
    #   - percent_removed
    rows_original = len(data)
    rows_filtered = len(filtered_data)
    rows_removed = rows_original - rows_filtered
    percent_removed = (rows_removed / rows_original * 100) if rows_original > 0 else 0.0
    return {
        "rows_original": rows_original,
        "rows_filtered": rows_filtered,
        "rows_removed": rows_removed,
        "percent_removed": percent_removed,
    }
