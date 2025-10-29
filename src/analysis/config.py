"""
Configuration for K-Fold cross-validation analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KFoldConfig:
    """
    Configuration for K-Fold cross-validation.

    Attributes:
        n_splits: Number of folds for cross-validation
        group_column: Column name to group by (e.g., time periods)
        account_column: Column name for accounts
        weight_column: Optional column for weighting (e.g., 'Auto_Base_Units')
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
        verbose: Whether to print progress

    Example:
        >>> config = KFoldConfig(
        ...     n_splits=5,
        ...     group_column='13_Week_Periods',
        ...     account_column='Account',
        ...     weight_column='Auto_Base_Units'
        ... )
    """

    n_splits: int
    group_column: str
    account_column: str
    weight_column: Optional[str] = None
    shuffle: bool = True
    random_state: Optional[int] = 42
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if not self.group_column:
            raise ValueError("group_column cannot be empty")

        if not self.account_column:
            raise ValueError("account_column cannot be empty")

        logger.debug(f"Created KFoldConfig: {self}")

    def validate_data(self, data) -> None:
        """
        Validate that data contains required columns.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required = [self.group_column, self.account_column]
        if self.weight_column:
            required.append(self.weight_column)

        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. "
                f"Available: {sorted(data.columns)}"
            )

        # Check that we have enough groups for n_splits
        n_groups = data[self.group_column].nunique()
        if n_groups < self.n_splits:
            raise ValueError(
                f"Not enough groups for {self.n_splits} splits. "
                f"Found only {n_groups} unique values in '{self.group_column}'"
            )

        logger.debug(
            f"Validation passed: {n_groups} groups available for {self.n_splits} splits"
        )
