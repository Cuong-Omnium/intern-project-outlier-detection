"""
Custom scikit-learn transformers for regression preprocessing.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log-transform numerical features with handling for zeros and negatives.

    This transformer applies natural log transformation and handles edge cases:
    - Zeros: Either clip to small positive value or add constant
    - Negatives: Raise error (log undefined for negative numbers)

    Parameters:
        handle_zeros: Strategy for handling zeros ('clip' or 'add_constant')
        clip_value: Value to clip to when handle_zeros='clip' (default: 1e-6)
        add_constant: Value to add when handle_zeros='add_constant' (default: 1.0)

    Attributes:
        feature_names_in_: Feature names seen during fit
        n_features_in_: Number of features seen during fit

    Example:
        >>> transformer = LogTransformer(handle_zeros='clip')
        >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        handle_zeros: str = "clip",
        clip_value: float = 1e-6,
        add_constant: float = 1.0,
    ):
        self.handle_zeros = handle_zeros
        self.clip_value = clip_value
        self.add_constant = add_constant

    def fit(self, X, y=None):
        """
        Fit the transformer (learns feature names).

        Args:
            X: Input features (array-like or DataFrame)
            y: Target variable (ignored, for compatibility)

        Returns:
            self
        """
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        self.n_features_in_ = len(self.feature_names_in_)

        logger.debug(
            f"LogTransformer fitted with {self.n_features_in_} features: "
            f"{self.feature_names_in_}"
        )
        return self

    def transform(self, X):
        """
        Apply log transformation to features.

        Args:
            X: Input features to transform

        Returns:
            Transformed features (same type as input)

        Raises:
            ValueError: If negative values found
        """
        # Convert to array for computation
        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else np.array(X)

        # Check for negative values
        if np.any(X_array < 0):
            raise ValueError(
                "Cannot log-transform negative values. "
                f"Found {np.sum(X_array < 0)} negative values."
            )

        # Handle zeros
        if self.handle_zeros == "clip":
            X_array = np.clip(X_array, self.clip_value, None)
            logger.debug(
                f"Clipped {np.sum(X_array == self.clip_value)} zero values to {self.clip_value}"
            )
        elif self.handle_zeros == "add_constant":
            X_array = X_array + self.add_constant
            logger.debug(f"Added constant {self.add_constant} before log transform")

        # Apply log transform
        X_transformed = np.log(X_array)

        # Return in same format as input
        if is_dataframe:
            # Prefix column names with 'Log_'
            new_columns = [f"Log_{col}" for col in X.columns]
            return pd.DataFrame(X_transformed, index=X.index, columns=new_columns)
        else:
            return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Args:
            input_features: Input feature names (if None, uses fitted names)

        Returns:
            Array of output feature names
        """
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array([f"Log_{name}" for name in input_features])


class SafeDropNA(BaseEstimator, TransformerMixin):
    """
    Drop rows with NaN values, tracking how many were removed.

    This is a scikit-learn compatible transformer for removing NaN values
    while logging the operation.

    Attributes:
        rows_dropped_: Number of rows dropped during last transform

    Example:
        >>> dropper = SafeDropNA()
        >>> X_clean = dropper.fit_transform(X)
        >>> print(f"Dropped {dropper.rows_dropped_} rows")
    """

    def __init__(self):
        self.rows_dropped_ = 0

    def fit(self, X, y=None):
        """Fit the transformer (no-op, for compatibility)."""
        return self

    def transform(self, X):
        """
        Remove rows with any NaN values.

        Args:
            X: Input features

        Returns:
            Features with NaN rows removed
        """
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            initial_rows = len(X)
            X_clean = X.dropna()
            self.rows_dropped_ = initial_rows - len(X_clean)

            if self.rows_dropped_ > 0:
                logger.warning(
                    f"Dropped {self.rows_dropped_} rows with NaN values "
                    f"({self.rows_dropped_/initial_rows*100:.1f}%)"
                )

            return X_clean
        else:
            # For arrays, use numpy
            initial_rows = X.shape[0]
            mask = ~np.isnan(X).any(axis=1)
            X_clean = X[mask]
            self.rows_dropped_ = initial_rows - X_clean.shape[0]

            if self.rows_dropped_ > 0:
                logger.warning(f"Dropped {self.rows_dropped_} rows with NaN values")

            return X_clean
