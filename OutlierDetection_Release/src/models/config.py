"""
Configuration classes for regression models.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RegressionConfig:
    """
    Configuration for regression model.

    Attributes:
        dependent_var: Name of the dependent variable (y)
        continuous_vars: List of continuous independent variables (will be log-transformed)
        categorical_vars: List of categorical independent variables (will be one-hot encoded)
        log_transform: Whether to log-transform continuous variables
        handle_zeros: How to handle zeros before log transform ('clip' or 'add_constant')
        clip_value: Value to clip to when handle_zeros='clip'
        drop_na: Whether to drop rows with NaN values

    Example:
        >>> config = RegressionConfig(
        ...     dependent_var='Unit_Sales',
        ...     continuous_vars=['Auto_Base_Units', 'ACV_Weighted_Distribution'],
        ...     categorical_vars=['Account', 'COT']
        ... )
    """

    dependent_var: str
    continuous_vars: list[str] = field(default_factory=list)
    categorical_vars: list[str] = field(default_factory=list)
    log_transform: bool = True
    handle_zeros: str = "clip"  # 'clip' or 'add_constant'
    clip_value: float = 1e-6
    drop_na: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.dependent_var:
            raise ValueError("dependent_var cannot be empty")

        if not self.continuous_vars and not self.categorical_vars:
            raise ValueError("Must specify at least one independent variable")

        if self.handle_zeros not in ["clip", "add_constant"]:
            raise ValueError("handle_zeros must be 'clip' or 'add_constant'")

        # Check for duplicates
        all_vars = [self.dependent_var] + self.continuous_vars + self.categorical_vars
        if len(all_vars) != len(set(all_vars)):
            raise ValueError("Duplicate variables found in configuration")

        logger.debug(f"Created RegressionConfig: {self}")

    @property
    def all_independent_vars(self) -> list[str]:
        """Get all independent variables (continuous + categorical)."""
        return self.continuous_vars + self.categorical_vars

    @property
    def all_vars(self) -> list[str]:
        """Get all variables (dependent + independent)."""
        return [self.dependent_var] + self.all_independent_vars

    def validate_data(self, data) -> None:
        """
        Validate that data contains all required variables.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If required variables are missing
        """
        missing = set(self.all_vars) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required variables in data: {sorted(missing)}. "
                f"Available columns: {sorted(data.columns)}"
            )
        logger.debug(f"Data validation passed for {len(self.all_vars)} variables")
