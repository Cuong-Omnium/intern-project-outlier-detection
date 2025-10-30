"""
Tests for custom transformers.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.transformers import LogTransformer, SafeDropNA


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_fit_transform_with_dataframe(self):
        """Test log transformation with DataFrame input."""
        X = pd.DataFrame({"A": [1, 10, 100], "B": [2, 20, 200]})

        transformer = LogTransformer()
        X_transformed = transformer.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert list(X_transformed.columns) == ["Log_A", "Log_B"]
        assert np.allclose(X_transformed["Log_A"], np.log([1, 10, 100]))

    def test_fit_transform_with_array(self):
        """Test log transformation with array input."""
        X = np.array([[1, 2], [10, 20], [100, 200]])

        transformer = LogTransformer()
        X_transformed = transformer.fit_transform(X)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == X.shape
        assert np.allclose(X_transformed[:, 0], np.log([1, 10, 100]))

    def test_handle_zeros_with_clip(self):
        """Test that zeros are clipped to small positive value."""
        X = pd.DataFrame({"A": [0, 1, 10]})

        transformer = LogTransformer(handle_zeros="clip", clip_value=1e-6)
        X_transformed = transformer.fit_transform(X)

        # First value should be log(1e-6), not -inf
        assert np.isfinite(X_transformed["Log_A"].iloc[0])
        assert X_transformed["Log_A"].iloc[0] == np.log(1e-6)

    def test_handle_zeros_with_add_constant(self):
        """Test that constant is added before log."""
        X = pd.DataFrame({"A": [0, 1, 10]})

        transformer = LogTransformer(handle_zeros="add_constant", add_constant=1.0)
        X_transformed = transformer.fit_transform(X)

        # Should be log(0+1), log(1+1), log(10+1)
        expected = np.log([1, 2, 11])
        assert np.allclose(X_transformed["Log_A"], expected)

    def test_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        X = pd.DataFrame({"A": [-1, 1, 10]})

        transformer = LogTransformer()

        with pytest.raises(ValueError, match="Cannot log-transform negative values"):
            transformer.fit_transform(X)

    def test_get_feature_names_out(self):
        """Test feature name generation."""
        X = pd.DataFrame({"Sales": [1, 2], "Units": [3, 4]})

        transformer = LogTransformer()
        transformer.fit(X)

        names = transformer.get_feature_names_out()
        assert list(names) == ["Log_Sales", "Log_Units"]


class TestSafeDropNA:
    """Tests for SafeDropNA transformer."""

    def test_drop_na_rows(self):
        """Test that rows with NaN are dropped."""
        X = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8]})

        dropper = SafeDropNA()
        X_clean = dropper.fit_transform(X)

        assert len(X_clean) == 2  # Only rows 0 and 3 have no NaN
        assert dropper.rows_dropped_ == 2

    def test_no_na_rows(self):
        """Test with data that has no NaN values."""
        X = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        dropper = SafeDropNA()
        X_clean = dropper.fit_transform(X)

        assert len(X_clean) == 3
        assert dropper.rows_dropped_ == 0

    def test_with_array_input(self):
        """Test SafeDropNA with numpy array."""
        X = np.array([[1, 2], [3, np.nan], [5, 6]])

        dropper = SafeDropNA()
        X_clean = dropper.fit_transform(X)

        assert X_clean.shape[0] == 2
        assert dropper.rows_dropped_ == 1
