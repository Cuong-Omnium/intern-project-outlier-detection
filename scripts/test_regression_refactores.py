"""
Test refactored regression pipeline with real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.data.filters import apply_filters
from src.data.loader import load_data
from src.models.config import RegressionConfig
from src.models.regression import RegressionPipelineBuilder


def main():
    # TODO: Load your data
    data = load_data(
        r"G:\.shortcut-targets-by-id\1bL9_9iTX8zmohHv_oK1Igik8jgfLvBCP\Cuong\9. Project\Intern Project - Summer 2025\250725 - Coke Pratyush.csv"
    )

    # TODO: Apply filters
    filtered_data = apply_filters(
        data,
        lower=1,
        cot="Food",
        acv_threshold=10,
        auto_promo_code=0,
        exclude_zero_sales=True,
    )

    # TODO: Create regression config
    config = RegressionConfig(
        dependent_var="Auto_Base_Units",
        continuous_vars=["ACV_Weighted_Distribution"],
        categorical_vars=["Price_Bucket_Quarter", "Account"],
    )

    # TODO: Build pipeline
    builder = RegressionPipelineBuilder(config)
    X, y = builder.prepare_data(filtered_data)
    pipeline = builder.build()

    # TODO: Fit the model
    pipeline.fit(X, y)

    # TODO: Make predictions
    y_pred = pipeline.predict(X)

    # TODO: Evaluate
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    pass


if __name__ == "__main__":
    main()
