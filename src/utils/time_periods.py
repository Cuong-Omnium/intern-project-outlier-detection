"""
Utility functions for time period segmentation.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_time_periods(
    data: pd.DataFrame,
    date_column: str = "Date",
    period_weeks: int = 13,
    period_column_name: str = "Time_Period",
) -> pd.DataFrame:
    """
    Create time period segments from date column.

    Divides the date range into equal-length periods (e.g., 4-week, 13-week).
    Each period is a continuous group of dates.

    Args:
        data: DataFrame with date column
        date_column: Name of the date column
        period_weeks: Number of weeks per period (4, 8, 12, or 13)
        period_column_name: Name for the new period column

    Returns:
        DataFrame with new period column added

    Example:
        >>> df = create_time_periods(df, period_weeks=13)
        >>> df['Time_Period'].unique()
        array([1, 2, 3, 4])  # 4 periods of 13 weeks each
    """
    df = data.copy()

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    total_days = (max_date - min_date).days

    logger.info(
        f"Date range: {min_date.date()} to {max_date.date()} ({total_days} days)"
    )

    # Calculate period boundaries
    period_days = period_weeks * 7

    # Create period labels based on days from start
    df["_days_from_start"] = (df[date_column] - min_date).dt.days
    df[period_column_name] = (df["_days_from_start"] // period_days) + 1

    # Clean up temporary column
    df = df.drop(columns=["_days_from_start"])

    # Get statistics
    n_periods = df[period_column_name].nunique()
    periods_info = df.groupby(period_column_name)[date_column].agg(
        ["min", "max", "count"]
    )

    logger.info(f"Created {n_periods} periods of {period_weeks} weeks each")
    logger.info(f"Period distribution:\n{periods_info}")

    return df


def create_equal_periods(
    data: pd.DataFrame,
    date_column: str = "Date",
    period_weeks: int = 13,
    period_column_name: str = "Time_Period",
) -> pd.DataFrame:
    """
    Create time periods with EXACT equal lengths by assigning dates to period buckets.

    This approach ensures each period has exactly N weeks, even if there are data gaps.

    Args:
        data: DataFrame with date column
        date_column: Name of the date column
        period_weeks: Number of weeks per period (4, 8, 12, or 13)
        period_column_name: Name for the new period column

    Returns:
        DataFrame with new period column added
    """
    df = data.copy()

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")

    # Calculate period length in days
    period_days = period_weeks * 7

    # Calculate which period each date belongs to
    # Formula: period = floor((days_since_start) / period_length) + 1
    days_since_start = (df[date_column] - min_date).dt.days
    df[period_column_name] = (days_since_start // period_days) + 1

    # Convert to integer
    df[period_column_name] = df[period_column_name].astype(int)

    # Handle any potential NaNs (shouldn't happen, but safety check)
    if df[period_column_name].isna().any():
        logger.warning(
            f"Found {df[period_column_name].isna().sum()} NaN periods, filling with 1"
        )
        df[period_column_name] = df[period_column_name].fillna(1).astype(int)

    # Get statistics
    n_periods = df[period_column_name].nunique()

    logger.info(f"Created {n_periods} periods of {period_weeks} weeks each")

    # Log period distribution with actual weeks
    period_stats = []
    for period in sorted(df[period_column_name].unique()):
        period_data = df[df[period_column_name] == period]
        period_min = period_data[date_column].min()
        period_max = period_data[date_column].max()
        days = (period_max - period_min).days + 1
        weeks = days / 7

        period_stats.append(
            {
                "period": period,
                "start": period_min.date(),
                "end": period_max.date(),
                "days": days,
                "weeks": round(weeks, 1),
                "records": len(period_data),
            }
        )

        logger.info(
            f"Period {period}: {period_min.date()} to {period_max.date()} "
            f"({days} days = {weeks:.1f} weeks, {len(period_data)} records)"
        )

    # Check if last period is shorter and warn user
    if period_stats:
        last_period = period_stats[-1]
        if last_period["weeks"] < period_weeks * 0.8:  # Less than 80% of target
            logger.warning(
                f"Last period is short: {last_period['weeks']:.1f} weeks "
                f"(expected {period_weeks} weeks)"
            )

    return df


def validate_period_coverage(
    data: pd.DataFrame,
    date_column: str = "Date",
    period_weeks: int = 13,
    min_periods_required: int = 3,
) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validate that the data has sufficient coverage for the chosen period length.

    Args:
        data: DataFrame with date column
        date_column: Name of the date column
        period_weeks: Number of weeks per period
        min_periods_required: Minimum number of periods needed

    Returns:
        Tuple of (is_valid, message, period_summary_df)
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    min_date = df[date_column].min()
    max_date = df[date_column].max()
    total_days = (max_date - min_date).days
    total_weeks = total_days / 7

    expected_periods = int(total_weeks / period_weeks)

    if expected_periods < min_periods_required:
        message = (
            f"⚠️ Insufficient data for {period_weeks}-week periods. "
            f"You have {total_weeks:.1f} weeks ({expected_periods} periods), "
            f"but need at least {min_periods_required} periods."
        )
        return False, message, None

    # Create temporary periods to check distribution
    temp_df = create_time_periods(df, date_column, period_weeks, "_temp_period")

    # Calculate actual weeks per period based on the date range within each period
    period_stats = []
    for period_num in sorted(temp_df["_temp_period"].unique()):
        period_data = temp_df[temp_df["_temp_period"] == period_num]

        period_min = period_data[date_column].min()
        period_max = period_data[date_column].max()

        # Calculate expected vs actual
        actual_days = (
            period_max - period_min
        ).days + 1  # +1 to include both start and end dates
        actual_weeks = actual_days / 7

        period_stats.append(
            {
                "Period": period_num,
                "Start_Date": period_min,
                "End_Date": period_max,
                "Num_Records": len(period_data),
                "Days": actual_days,
                "Weeks": round(actual_weeks, 1),
                "Expected_Weeks": period_weeks,
            }
        )

    period_summary = pd.DataFrame(period_stats)

    message = (
        f"✅ Valid: {expected_periods} periods of ~{period_weeks} weeks "
        f"(Total: {total_weeks:.1f} weeks)"
    )

    return True, message, period_summary


def get_recommended_period_lengths(
    data: pd.DataFrame, date_column: str = "Date"
) -> list[dict]:
    """
    Get recommended period lengths based on data coverage.

    Returns list of viable options with statistics.

    Args:
        data: DataFrame with date column
        date_column: Name of the date column

    Returns:
        List of dictionaries with period length options
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    total_days = (df[date_column].max() - df[date_column].min()).days
    total_weeks = total_days / 7

    options = []

    for weeks in [4, 8, 12, 13]:
        num_periods = int(total_weeks / weeks)

        if num_periods >= 3:  # Need at least 3 periods for k-fold
            is_valid, message, summary = validate_period_coverage(
                df, date_column, weeks, min_periods_required=3
            )

            options.append(
                {
                    "weeks": weeks,
                    "num_periods": num_periods,
                    "is_valid": is_valid,
                    "message": message,
                    "recommended": (
                        num_periods >= 4 and num_periods <= 10
                    ),  # Sweet spot
                }
            )

    return options
