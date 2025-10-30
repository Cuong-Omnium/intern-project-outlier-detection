"""
Result classes for K-Fold analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """
    Results from a single fold of cross-validation.

    Attributes:
        fold_number: Fold index (1-based)
        train_indices: Indices used for training
        test_indices: Indices used for testing
        predictions: Model predictions on test set
        actuals: Actual values on test set
        residuals: Prediction residuals (actual - predicted)
        mse: Mean squared error for this fold
        r2: R-squared score for this fold
        accounts: Account identifiers from test set
        weights: Optional weights from test set
    """

    fold_number: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    mse: float
    r2: float
    accounts: np.ndarray
    weights: Optional[np.ndarray] = None

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.test_indices)

    def __repr__(self) -> str:
        return (
            f"FoldResult(fold={self.fold_number}, "
            f"train={self.n_train}, test={self.n_test}, "
            f"MSE={self.mse:.4f}, R²={self.r2:.4f})"
        )


@dataclass
class KFoldResult:
    """
    Complete results from K-Fold cross-validation.

    Attributes:
        fold_results: List of individual fold results
        config: Configuration used for analysis
        mean_mse: Average MSE across folds
        std_mse: Standard deviation of MSE across folds
        mean_r2: Average R² across folds
        std_r2: Standard deviation of R² across folds
    """

    fold_results: list[FoldResult]
    config: "KFoldConfig"

    def __post_init__(self):
        """Calculate aggregate metrics."""
        mse_scores = [fold.mse for fold in self.fold_results]
        r2_scores = [fold.r2 for fold in self.fold_results]

        self.mean_mse = float(np.mean(mse_scores))
        self.std_mse = float(np.std(mse_scores))
        self.mean_r2 = float(np.mean(r2_scores))
        self.std_r2 = float(np.std(r2_scores))

        logger.info(
            f"K-Fold complete: MSE = {self.mean_mse:.4f} ± {self.std_mse:.4f}, "
            f"R² = {self.mean_r2:.4f} ± {self.std_r2:.4f}"
        )

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.fold_results)

    def get_all_residuals(self) -> np.ndarray:
        """Get residuals from all folds concatenated."""
        return np.concatenate([fold.residuals for fold in self.fold_results])

    def get_all_accounts(self) -> np.ndarray:
        """Get accounts from all folds concatenated."""
        return np.concatenate([fold.accounts for fold in self.fold_results])

    def get_all_weights(self) -> Optional[np.ndarray]:
        """Get weights from all folds concatenated (if available)."""
        if self.fold_results[0].weights is None:
            return None
        return np.concatenate([fold.weights for fold in self.fold_results])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all fold results to a DataFrame.

        Returns:
            DataFrame with columns: fold, account, actual, predicted, residual, weight
        """
        rows = []
        for fold in self.fold_results:
            fold_df = pd.DataFrame(
                {
                    "fold": fold.fold_number,
                    "account": fold.accounts,
                    "actual": fold.actuals,
                    "predicted": fold.predictions,
                    "residual": fold.residuals,
                }
            )

            if fold.weights is not None:
                fold_df["weight"] = fold.weights

            rows.append(fold_df)

        return pd.concat(rows, ignore_index=True)

    def get_fold_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each fold.

        Returns:
            DataFrame with fold-level metrics
        """
        return pd.DataFrame(
            [
                {
                    "fold": fold.fold_number,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "mse": fold.mse,
                    "r2": fold.r2,
                    "mean_residual": float(np.mean(fold.residuals)),
                    "std_residual": float(np.std(fold.residuals)),
                }
                for fold in self.fold_results
            ]
        )

    def __repr__(self) -> str:
        return (
            f"KFoldResult(n_folds={self.n_folds}, "
            f"mean_MSE={self.mean_mse:.4f}, mean_R²={self.mean_r2:.4f})"
        )
