"""
K-Fold cross-validation analyzer.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .config import KFoldConfig
from .results import FoldResult, KFoldResult

logger = logging.getLogger(__name__)


class KFoldAnalyzer:
    """
    Performs K-Fold cross-validation with group-based splitting.

    This analyzer handles time-series grouped cross-validation, where splits
    are made based on time periods to prevent data leakage.

    Attributes:
        config: KFoldConfig specifying analysis parameters

    Example:
        >>> config = KFoldConfig(
        ...     n_splits=5,
        ...     group_column='13_Week_Periods',
        ...     account_column='Account'
        ... )
        >>> analyzer = KFoldAnalyzer(config)
        >>> results = analyzer.run(data, X, y, pipeline)
        >>> print(f"Average MSE: {results.mean_mse:.4f}")
    """

    def __init__(self, config: KFoldConfig):
        """
        Initialize the analyzer.

        Args:
            config: K-Fold configuration
        """
        self.config = config
        logger.info(f"Initialized KFoldAnalyzer: {config}")

    def run(
        self, data: pd.DataFrame, X: pd.DataFrame, y: pd.Series, pipeline
    ) -> KFoldResult:
        """
        Run K-Fold cross-validation.

        Args:
            data: Full dataset (for accessing accounts, weights, groups)
            X: Features for modeling
            y: Target variable
            pipeline: sklearn Pipeline to train and evaluate

        Returns:
            KFoldResult with complete analysis results

        Raises:
            ValueError: If data validation fails

        Example:
            >>> results = analyzer.run(data, X, y, pipeline)
            >>> print(results.get_fold_summary())
        """
        # Validate data
        self.config.validate_data(data)

        # Extract groups for splitting
        groups = data[self.config.group_column].values

        # Initialize splitter
        gkf = GroupKFold(n_splits=self.config.n_splits)

        # Run cross-validation
        fold_results = []

        logger.info(f"Starting {self.config.n_splits}-fold cross-validation...")

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            if self.config.verbose:
                logger.info(f"Processing fold {fold_idx}/{self.config.n_splits}...")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Clone pipeline to avoid state contamination
            fold_pipeline = clone(pipeline)

            # Train model
            fold_pipeline.fit(X_train, y_train)

            # Predict on test set
            y_pred = fold_pipeline.predict(X_test)

            # Calculate residuals
            residuals = y_test.values - y_pred

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Extract accounts
            accounts = data.iloc[test_idx][self.config.account_column].values

            # Extract weights if specified
            weights = None
            if self.config.weight_column:
                weights = data.iloc[test_idx][self.config.weight_column].values

            # Create fold result
            fold_result = FoldResult(
                fold_number=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                predictions=y_pred,
                actuals=y_test.values,
                residuals=residuals,
                mse=mse,
                r2=r2,
                accounts=accounts,
                weights=weights,
            )

            fold_results.append(fold_result)

            if self.config.verbose:
                logger.info(f"Fold {fold_idx}: MSE = {mse:.4f}, R² = {r2:.4f}")

        # Create complete result
        result = KFoldResult(fold_results=fold_results, config=self.config)

        return result

    def find_optimal_k(
        self,
        data: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline,
        k_range: range = range(2, 11),
    ) -> tuple[int, pd.DataFrame]:
        """
        Find optimal number of folds by trying different values.

        Args:
            data: Full dataset
            X: Features
            y: Target
            pipeline: Model pipeline
            k_range: Range of k values to try (default: 2-10)

        Returns:
            Tuple of (optimal_k, results_df) where results_df contains
            MSE and R² for each k value

        Example:
            >>> optimal_k, results = analyzer.find_optimal_k(data, X, y, pipeline)
            >>> print(f"Best k: {optimal_k}")
            >>> print(results)
        """
        logger.info(f"Searching for optimal k in range {list(k_range)}...")

        results = []

        for k in k_range:
            # Check if we have enough groups
            n_groups = data[self.config.group_column].nunique()
            if k > n_groups:
                logger.warning(f"Skipping k={k} (only {n_groups} groups available)")
                continue

            # Create temporary config with this k
            temp_config = KFoldConfig(
                n_splits=k,
                group_column=self.config.group_column,
                account_column=self.config.account_column,
                weight_column=self.config.weight_column,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
                verbose=False,  # Suppress individual fold logging
            )

            # Run analysis
            temp_analyzer = KFoldAnalyzer(temp_config)
            kfold_result = temp_analyzer.run(data, X, y, pipeline)

            # Store results
            results.append(
                {
                    "k": k,
                    "mean_mse": kfold_result.mean_mse,
                    "std_mse": kfold_result.std_mse,
                    "mean_r2": kfold_result.mean_r2,
                    "std_r2": kfold_result.std_r2,
                }
            )

            logger.info(
                f"k={k}: MSE={kfold_result.mean_mse:.4f}, R²={kfold_result.mean_r2:.4f}"
            )

        results_df = pd.DataFrame(results)

        # Find optimal k (lowest mean MSE)
        optimal_k = int(results_df.loc[results_df["mean_mse"].idxmin(), "k"])

        logger.info(f"Optimal k found: {optimal_k}")

        return optimal_k, results_df
