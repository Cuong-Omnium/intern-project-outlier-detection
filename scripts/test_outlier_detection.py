"""
End-to-end test of outlier detection pipeline.
"""

import logging
from pathlib import Path

from src.analysis.config import KFoldConfig
from src.analysis.kfold import KFoldAnalyzer
from src.data.filters import apply_filters
from src.data.loader import load_data
from src.models.config import RegressionConfig
from src.models.regression import RegressionPipelineBuilder
from src.outliers.analyzer import OutlierAnalyzer
from src.outliers.config import OutlierConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Run complete outlier detection pipeline."""

    logger.info("=" * 80)
    logger.info("STARTING COMPLETE OUTLIER DETECTION PIPELINE")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)
    data_path = Path(
        r"G:\.shortcut-targets-by-id\1bL9_9iTX8zmohHv_oK1Igik8jgfLvBCP\Cuong\9. Project\Intern Project - Summer 2025\250725 - Coke Pratyush.csv"
    )
    data = load_data(data_path)
    logger.info(f"Loaded {len(data)} rows")

    # 2. Apply filters
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: APPLYING FILTERS")
    logger.info("=" * 80)
    filtered_data = apply_filters(
        data,
        lower=1,
        cot="Food",
        acv_threshold=10,
        auto_promo_code=0,
        exclude_zero_sales=True,
    )
    logger.info(f"Filtered to {len(filtered_data)} rows")

    # 3. Configure and build regression
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: BUILDING REGRESSION MODEL")
    logger.info("=" * 80)
    reg_config = RegressionConfig(
        dependent_var="Auto_Base_Units",
        continuous_vars=["ACV_Weighted_Distribution"],
        categorical_vars=["Price_Bucket_Quarter", "Account"],
    )

    builder = RegressionPipelineBuilder(reg_config)
    X, y = builder.prepare_data(filtered_data)
    pipeline = builder.build()
    logger.info(f"Prepared data: X={X.shape}, y={y.shape}")

    # 4. Run K-Fold cross-validation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: RUNNING K-FOLD CROSS-VALIDATION")
    logger.info("=" * 80)
    kfold_config = KFoldConfig(
        n_splits=5,
        group_column="13_Week_Periods",
        account_column="Account",
        weight_column="Auto_Base_Units",
        verbose=True,
    )

    kfold_analyzer = KFoldAnalyzer(kfold_config)
    kfold_results = kfold_analyzer.run(filtered_data, X, y, pipeline)

    logger.info(f"\nK-Fold Results:")
    logger.info(
        f"  Mean MSE: {kfold_results.mean_mse:.4f} ± {kfold_results.std_mse:.4f}"
    )
    logger.info(f"  Mean R²: {kfold_results.mean_r2:.4f} ± {kfold_results.std_r2:.4f}")

    # 5. Detect outliers
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: DETECTING OUTLIER ACCOUNTS")
    logger.info("=" * 80)
    outlier_config = OutlierConfig(
        contamination=0.10, eps_percentile=90, use_dbcv=True, show_kdist_plot=False
    )

    outlier_analyzer = OutlierAnalyzer(outlier_config)
    outlier_results = outlier_analyzer.detect(
        accounts=kfold_results.get_all_accounts(),
        residuals=kfold_results.get_all_residuals(),
        weights=kfold_results.get_all_weights(),
    )

    # 6. Display results
    logger.info("\n" + "=" * 80)
    logger.info("OUTLIER DETECTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total accounts analyzed: {outlier_results.n_accounts}")
    logger.info(
        f"Outliers detected: {outlier_results.n_outliers} ({outlier_results.outlier_rate*100:.1f}%)"
    )
    logger.info(f"Density method used: {outlier_results.density_method}")

    # Show outlier summary
    logger.info("\n" + "-" * 80)
    logger.info("OUTLIER SUMMARY (Top 10 by mean residual magnitude)")
    logger.info("-" * 80)
    summary = outlier_results.get_outlier_summary()
    display_cols = [
        "Account",
        "Outlier_Type",
        "w_mean_residual",
        "w_std_residual",
        "count",
    ]
    print(summary[display_cols].head(10).to_string(index=False))

    # Breakdown by type
    logger.info("\n" + "-" * 80)
    logger.info("OUTLIER TYPE BREAKDOWN")
    logger.info("-" * 80)
    type_counts = summary["Outlier_Type"].value_counts()
    for outlier_type, count in type_counts.items():
        accounts = outlier_results.get_outlier_accounts(outlier_type)
        logger.info(f"{outlier_type}: {count} accounts")
        logger.info(f"  Examples: {', '.join(accounts[:5])}")

    # 7. Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 80)

    from src.visualization.plots import (
        create_fold_mse_comparison,
        create_interactive_3d_plot,
    )

    # 3D interactive plot
    fig_3d = create_interactive_3d_plot(outlier_results.agg_data)
    logger.info("3D plot created - opening in browser...")
    fig_3d.show()

    # Fold MSE comparison
    fig_mse = create_fold_mse_comparison(kfold_results)
    logger.info("MSE comparison plot created - opening in browser...")
    fig_mse.show()

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)

    return outlier_results


if __name__ == "__main__":
    results = main()
