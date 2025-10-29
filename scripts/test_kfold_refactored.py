"""
Test refactored K-Fold analysis.
"""

from pathlib import Path

from src.analysis.config import KFoldConfig
from src.analysis.kfold import KFoldAnalyzer
from src.data.filters import apply_filters
from src.data.loader import load_data
from src.models.config import RegressionConfig
from src.models.regression import RegressionPipelineBuilder


def main():

    data = load_data(
        r"G:\.shortcut-targets-by-id\1bL9_9iTX8zmohHv_oK1Igik8jgfLvBCP\Cuong\9. Project\Intern Project - Summer 2025\250725 - Coke Pratyush.csv"
    )

    filtered_data = apply_filters(
        data,
        lower=1,
        cot="Food",
        acv_threshold=10,
        auto_promo_code=0,
        exclude_zero_sales=True,
    )

    config = RegressionConfig(
        dependent_var="Auto_Base_Units",
        continuous_vars=["ACV_Weighted_Distribution"],
        categorical_vars=["Price_Bucket_Quarter", "Account"],
    )

    builder = RegressionPipelineBuilder(config)
    X, y = builder.prepare_data(filtered_data)
    pipeline = builder.build()

    config = KFoldConfig(
        n_splits=5,
        group_column="13_Week_Periods",
        account_column="Account",
        weight_column="Auto_Base_Units",
    )

    analyzer = KFoldAnalyzer(config)
    results = analyzer.run(filtered_data, X, y, pipeline)

    print(results.get_fold_summary())
    print(f"\nOverall: MSE = {results.mean_mse:.4f} ± {results.std_mse:.4f}")

    optimal_k, k_results = analyzer.find_optimal_k(filtered_data, X, y, pipeline)
    print(f"\nOptimal k: {optimal_k}")
    print(k_results)

    pass


if __name__ == "__main__":
    main()
