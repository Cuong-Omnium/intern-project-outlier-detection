"""
Regression pipeline builder for log-linear models.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RegressionConfig
from .transformers import LogTransformer

logger = logging.getLogger(__name__)


class RegressionPipelineBuilder:
    """
    Builds scikit-learn pipelines for log-linear regression.

    This class handles the construction of preprocessing and modeling pipelines
    based on a RegressionConfig specification.

    Attributes:
        config: RegressionConfig specifying model structure

    Example:
        >>> config = RegressionConfig(
        ...     dependent_var='Unit_Sales',
        ...     continuous_vars=['Auto_Base_Units'],
        ...     categorical_vars=['Account']
        ... )
        >>> builder = RegressionPipelineBuilder(config)
        >>> pipeline = builder.build()
        >>> pipeline.fit(X, y)
    """

    def __init__(self, config: RegressionConfig):
        """
        Initialize the pipeline builder.

        Args:
            config: Regression configuration
        """
        self.config = config
        logger.info(f"Initialized RegressionPipelineBuilder with config: {config}")

    def build(self) -> Pipeline:
        """
        Build the complete scikit-learn pipeline.

        Returns:
            Fitted Pipeline object ready for training

        Example:
            >>> pipeline = builder.build()
            >>> pipeline.fit(X_train, y_train)
            >>> predictions = pipeline.predict(X_test)
        """
        logger.info("Building regression pipeline...")

        # Build preprocessing pipeline
        preprocessor = self._build_preprocessor()

        # Build complete pipeline with model
        pipeline = Pipeline(
            [("preprocessing", preprocessor), ("regression", LinearRegression())]
        )

        logger.info("Pipeline built successfully")
        return pipeline

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the preprocessing ColumnTransformer.

        This handles:
        - Log transformation of continuous variables
        - One-hot encoding of categorical variables
        - Imputation of missing values

        Returns:
            ColumnTransformer for preprocessing
        """
        transformers = []

        # Continuous variables: log transform + impute
        if self.config.continuous_vars:
            continuous_pipeline = Pipeline(
                [
                    (
                        "log",
                        LogTransformer(
                            handle_zeros=self.config.handle_zeros,
                            clip_value=self.config.clip_value,
                        ),
                    ),
                    ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ]
            )

            transformers.append(
                ("continuous", continuous_pipeline, self.config.continuous_vars)
            )

            logger.debug(
                f"Added continuous transformer for {len(self.config.continuous_vars)} variables"
            )

        # Categorical variables: one-hot encode
        if self.config.categorical_vars:
            categorical_pipeline = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="first",  # Avoid multicollinearity
                            handle_unknown="ignore",  # Handle new categories gracefully
                            sparse_output=False,
                        ),
                    )
                ]
            )

            transformers.append(
                ("categorical", categorical_pipeline, self.config.categorical_vars)
            )

            logger.debug(
                f"Added categorical transformer for {len(self.config.categorical_vars)} variables"
            )

        # Build ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any columns not specified
        )

        return preprocessor

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling (separate X and y, apply transformations).

        Args:
            data: Input DataFrame with all variables

        Returns:
            Tuple of (X, y) where:
                - X: Independent variables (DataFrame)
                - y: Log-transformed dependent variable (Series)

        Raises:
            ValueError: If required variables missing or data invalid

        Example:
            >>> X, y = builder.prepare_data(df)
            >>> X.shape, y.shape
            ((1000, 5), (1000,))
        """
        # Validate data has required columns
        self.config.validate_data(data)

        # Make a copy to avoid modifying original
        data = data.copy()

        # Log-transform dependent variable
        if self.config.log_transform:
            logger.info(
                f"Log-transforming dependent variable: {self.config.dependent_var}"
            )

            dep_values = data[self.config.dependent_var].values

            # Check for negative values
            if np.any(np.array(dep_values) < 0):
                raise ValueError(
                    f"Dependent variable '{self.config.dependent_var}' contains negative values. "
                    "Cannot log-transform."
                )

            # Handle zeros
            if self.config.handle_zeros == "clip":
                dep_values = np.clip(dep_values, self.config.clip_value, None)
            elif self.config.handle_zeros == "add_constant":
                dep_values = dep_values + 1.0

            y = pd.Series(
                np.log(dep_values),
                index=data.index,
                name=f"Log_{self.config.dependent_var}",
            )
        else:
            y = data[self.config.dependent_var]

        # Extract independent variables
        X = data[self.config.all_independent_vars].copy()

        # Drop NaN if configured
        if self.config.drop_na:
            # Combine X and y for joint dropping
            combined = pd.concat([X, y], axis=1)
            initial_len = len(combined)
            combined = combined.dropna()
            dropped = initial_len - len(combined)

            if dropped > 0:
                logger.warning(
                    f"Dropped {dropped} rows with NaN values ({dropped/initial_len*100:.1f}%)"
                )

            X = combined[self.config.all_independent_vars]
            y = combined[y.name]

        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")

        return X, y
