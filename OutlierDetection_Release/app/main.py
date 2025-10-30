"""
Streamlit web application for Account Outlier Detection.
Enhanced with loading animations, config management, and improved charts.
"""

# Path setup
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Configure page
st.set_page_config(
    page_title="Account Outlier Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import modules
from src.analysis.config import KFoldConfig
from src.analysis.kfold import KFoldAnalyzer
from src.config.manager import ConfigManager
from src.data.filters import apply_filters, get_filter_summary
from src.data.loader import DataLoadError, load_data, validate_required_columns
from src.models.config import RegressionConfig
from src.models.regression import RegressionPipelineBuilder
from src.outliers.analyzer import OutlierAnalyzer
from src.outliers.config import OutlierConfig
from src.visualization.plots import (
    create_account_time_series,
    create_all_account_charts,
    create_fold_mse_comparison,
    create_interactive_3d_plot,
    create_optimal_k_plot,
)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None
if "outlier_results" not in st.session_state:
    st.session_state.outlier_results = None
if "kfold_results" not in st.session_state:
    st.session_state.kfold_results = None
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager()
if "optimal_k" not in st.session_state:
    st.session_state.optimal_k = None


def show_loading_animation(message: str, duration: float = 0.5):
    """Show a loading animation with message."""
    with st.spinner(message):
        time.sleep(duration)


def main():
    """Main Streamlit app."""

    # Custom CSS
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
            background-color: #008080;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            margin: 1rem 0;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            margin: 1rem 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Title
    st.title("🔍 Account Outlier Detection")
    st.markdown("**Identify problematic accounts in CPG linear regression models**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📍 Navigation")
        page = st.radio(
            "Go to",
            [
                "1️⃣ Upload Data",
                "2️⃣ Configure Analysis",
                "3️⃣ Run Analysis",
                "4️⃣ View Results",
                "5️⃣ Account Charts",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Configuration management
        st.subheader("💾 Configurations")

        configs = st.session_state.config_manager.list_configs()

        if configs:
            config_names = [cfg.stem for cfg in configs]
            selected_config = st.selectbox("Saved configs", [""] + config_names)

            col1, col2 = st.columns(2)
            with col1:
                if selected_config and st.button("📥 Load", use_container_width=True):
                    try:
                        config_path = (
                            st.session_state.config_manager.config_dir
                            / f"{selected_config}.yaml"
                        )
                        loaded_config = st.session_state.config_manager.load_config(
                            config_path
                        )

                        st.session_state.reg_config = loaded_config["regression"]
                        st.session_state.kfold_config = loaded_config["kfold"]
                        st.session_state.outlier_config = loaded_config["outlier"]
                        st.session_state.filter_config = loaded_config["filters"]

                        st.success(f"✅ Loaded!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            with col2:
                if selected_config and st.button("🗑️ Delete", use_container_width=True):
                    try:
                        config_path = (
                            st.session_state.config_manager.config_dir
                            / f"{selected_config}.yaml"
                        )
                        st.session_state.config_manager.delete_config(config_path)
                        st.success("Deleted!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No saved configurations")

    # Route to pages
    if page == "1️⃣ Upload Data":
        page_upload_data()
    elif page == "2️⃣ Configure Analysis":
        page_configure_analysis()
    elif page == "3️⃣ Run Analysis":
        page_run_analysis()
    elif page == "4️⃣ View Results":
        page_view_results()
    elif page == "5️⃣ Account Charts":
        page_account_charts()


def page_upload_data():
    """Page 1: Upload and preview data."""
    st.header("📁 Step 1: Upload Data")

    st.markdown(
        """
    Upload your sales data file (CSV or Excel format).

    **Required columns:**
    - Account identifier
    - Date/time period
    - Sales metrics (units, dollars, etc.)
    - Categorical variables (COT, price bucket, etc.)
    """
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV or Excel file with sales data",
    )

    if uploaded_file is not None:
        try:
            temp_path = Path("temp_upload")
            temp_path.mkdir(exist_ok=True)
            file_path = temp_path / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            show_loading_animation("📊 Loading data...", 0.5)

            data = load_data(file_path)
            st.session_state.data = data

            st.markdown(
                '<div class="success-box">✅ Successfully loaded data!</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{len(data):,}")
            with col2:
                st.metric("📋 Columns", len(data.columns))
            with col3:
                if "Account" in data.columns:
                    st.metric("🏢 Unique Accounts", data["Account"].nunique())

            st.subheader("👀 Data Preview")
            st.dataframe(data.head(20), use_container_width=True)

            with st.expander("📊 Column Information"):
                col_info = pd.DataFrame(
                    {
                        "Column": data.columns,
                        "Type": data.dtypes.astype(str),
                        "Non-Null": data.count(),
                        "Null %": (data.isnull().sum() / len(data) * 100).round(2),
                    }
                )
                st.dataframe(col_info, use_container_width=True, height=400)

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.exception(e)
    else:
        st.info("👆 Upload a file to get started")


def page_configure_analysis():
    """Page 2: Configure filters and model."""
    st.header("⚙️ Step 2: Configure Analysis")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first (Step 1)")
        return

    data = st.session_state.data
    filter_config = st.session_state.get("filter_config", {})

    # 1. FILTERS
    st.subheader("1️⃣ Data Filters")

    with st.expander("🔍 Configure Filters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            if "Lower" in data.columns:
                use_lower = st.checkbox(
                    "Filter by Lower", value=filter_config.get("use_lower", True)
                )
                lower_value = (
                    st.number_input(
                        "Lower value",
                        value=filter_config.get("lower_value", 1),
                        disabled=not use_lower,
                    )
                    if use_lower
                    else None
                )
            else:
                lower_value = None

            if any("ACV" in col for col in data.columns):
                use_acv = st.checkbox(
                    "Filter by ACV threshold", value=filter_config.get("use_acv", True)
                )
                acv_value = (
                    st.number_input(
                        "ACV threshold",
                        value=filter_config.get("acv_value", 10),
                        disabled=not use_acv,
                    )
                    if use_acv
                    else None
                )
            else:
                acv_value = None

        with col2:
            if "COT" in data.columns:
                use_cot = st.checkbox(
                    "Filter by COT", value=filter_config.get("use_cot", True)
                )
                default_cot = filter_config.get("cot_value", data["COT"].unique()[0])
                cot_value = (
                    st.selectbox(
                        "COT value",
                        options=data["COT"].unique(),
                        index=(
                            list(data["COT"].unique()).index(default_cot)
                            if default_cot in data["COT"].unique()
                            else 0
                        ),
                        disabled=not use_cot,
                    )
                    if use_cot
                    else None
                )
            else:
                cot_value = None

            if "Auto_Promo_Code" in data.columns:
                use_promo = st.checkbox(
                    "Filter by Auto Promo Code",
                    value=filter_config.get("use_promo", True),
                )
                promo_value = (
                    st.number_input(
                        "Promo code",
                        value=filter_config.get("promo_value", 0),
                        disabled=not use_promo,
                    )
                    if use_promo
                    else None
                )
            else:
                promo_value = None

        exclude_zeros = st.checkbox(
            "Exclude zero unit sales", value=filter_config.get("exclude_zeros", True)
        )

    if st.button("✅ Apply Filters", type="primary"):
        show_loading_animation("🔄 Applying filters...", 0.3)

        try:
            filtered = apply_filters(
                data,
                lower=lower_value,
                cot=cot_value,
                acv_threshold=acv_value,
                auto_promo_code=promo_value,
                exclude_zero_sales=exclude_zeros,
            )
            st.session_state.filtered_data = filtered

            st.session_state.filter_config = {
                "use_lower": locals().get("use_lower", False),
                "lower_value": lower_value,
                "use_acv": locals().get("use_acv", False),
                "acv_value": acv_value,
                "use_cot": locals().get("use_cot", False),
                "cot_value": cot_value,
                "use_promo": locals().get("use_promo", False),
                "promo_value": promo_value,
                "exclude_zeros": exclude_zeros,
            }

            summary = get_filter_summary(data, filtered)
            st.markdown(
                f'<div class="success-box">✅ Filtered to {len(filtered):,} rows ({summary["percent_removed"]:.1f}% removed)</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Original", f"{summary['rows_original']:,}")
            with col2:
                st.metric("✅ Filtered", f"{summary['rows_filtered']:,}")
            with col3:
                st.metric("🗑️ Removed", f"{summary['rows_removed']:,}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # 2. MODEL CONFIGURATION
    if st.session_state.filtered_data is not None:
        filtered = st.session_state.filtered_data

        st.subheader("2️⃣ Regression Model")

        with st.expander("📈 Configure Variables", expanded=True):
            reg_config = st.session_state.get("reg_config", {})

            dep_var = st.selectbox(
                "Dependent Variable (Y)",
                options=[
                    col
                    for col in filtered.columns
                    if filtered[col].dtype in ["int64", "float64"]
                ],
                index=0,
            )

            continuous_options = [
                col
                for col in filtered.columns
                if filtered[col].dtype in ["int64", "float64"] and col != dep_var
            ]
            continuous_vars = st.multiselect(
                "Continuous Variables (log-transformed)",
                options=continuous_options,
                default=reg_config.get(
                    "continuous_vars",
                    continuous_options[:1] if continuous_options else [],
                ),
            )

            categorical_options = [
                col for col in filtered.columns if filtered[col].dtype == "object"
            ]
            categorical_vars = st.multiselect(
                "Categorical Variables (one-hot encoded)",
                options=categorical_options,
                default=reg_config.get(
                    "categorical_vars",
                    ["Account"] if "Account" in categorical_options else [],
                ),
            )

            if not continuous_vars and not categorical_vars:
                st.warning("⚠️ Select at least one independent variable")

        st.session_state.reg_config = {
            "dependent_var": dep_var,
            "continuous_vars": continuous_vars,
            "categorical_vars": categorical_vars,
        }

        # 3. K-FOLD CONFIG
        st.subheader("3️⃣ K-Fold Cross-Validation")

        with st.expander("🔄 Configure K-Fold", expanded=True):
            kfold_config = st.session_state.get("kfold_config", {})

            col1, col2 = st.columns(2)

            with col1:
                group_col = st.selectbox(
                    "Time period column",
                    options=[
                        col
                        for col in filtered.columns
                        if "period" in col.lower() or "week" in col.lower()
                    ],
                    index=0,
                )

                account_col = st.selectbox(
                    "Account column",
                    options=categorical_options,
                    index=0 if "Account" in categorical_options else 0,
                )

            with col2:
                # Show optimal k suggestion
                if st.session_state.optimal_k is not None:
                    st.markdown(
                        f'<div class="info-box">💡 Suggested optimal k: <b>{st.session_state.optimal_k}</b></div>',
                        unsafe_allow_html=True,
                    )

                n_splits = st.slider(
                    "Number of folds (k)",
                    min_value=2,
                    max_value=10,
                    value=(
                        st.session_state.optimal_k
                        if st.session_state.optimal_k
                        else kfold_config.get("n_splits", 5)
                    ),
                )

                weight_col = st.selectbox(
                    "Weight column (optional)",
                    options=["None"]
                    + [
                        col
                        for col in filtered.columns
                        if filtered[col].dtype in ["int64", "float64"]
                    ],
                    index=0,
                )

            # Find optimal k button
            if st.button("🎯 Find Optimal K"):
                with st.spinner("🔍 Testing different k values..."):
                    try:
                        # Build temporary model
                        temp_config = RegressionConfig(
                            dependent_var=dep_var,
                            continuous_vars=continuous_vars,
                            categorical_vars=categorical_vars,
                        )
                        builder = RegressionPipelineBuilder(temp_config)
                        X, y = builder.prepare_data(filtered)
                        pipeline = builder.build()

                        # Test different k values
                        temp_kfold_config = KFoldConfig(
                            n_splits=5,  # Will be overridden
                            group_column=group_col,
                            account_column=account_col,
                            weight_column=None if weight_col == "None" else weight_col,
                            verbose=False,
                        )

                        analyzer = KFoldAnalyzer(temp_kfold_config)
                        optimal_k, k_results = analyzer.find_optimal_k(
                            filtered,
                            X,
                            y,
                            pipeline,
                            k_range=range(
                                2, min(11, filtered[group_col].nunique() + 1)
                            ),
                        )

                        st.session_state.optimal_k = optimal_k

                        st.success(f"✅ Optimal k found: {optimal_k}")

                        # Show chart
                        fig = create_optimal_k_plot(k_results)
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Error finding optimal k: {e}")

        st.session_state.kfold_config = {
            "n_splits": n_splits,
            "group_column": group_col,
            "account_column": account_col,
            "weight_column": None if weight_col == "None" else weight_col,
        }

        # 4. OUTLIER CONFIG
        st.subheader("4️⃣ Outlier Detection")

        with st.expander("🎯 Configure Detection", expanded=True):
            outlier_config = st.session_state.get("outlier_config", {})

            col1, col2 = st.columns(2)

            with col1:
                contamination = (
                    st.slider(
                        "Expected outlier %",
                        min_value=1,
                        max_value=30,
                        value=int(outlier_config.get("contamination", 10)),
                        help="Expected percentage of outliers",
                    )
                    / 100
                )

                eps_percentile = st.slider(
                    "DBSCAN eps percentile",
                    min_value=50,
                    max_value=99,
                    value=outlier_config.get("eps_percentile", 90),
                )

            with col2:
                dbscan_min_samples = st.number_input(
                    "DBSCAN min samples",
                    min_value=2,
                    value=outlier_config.get("dbscan_min_samples", 5),
                )

                hdbscan_min_cluster = st.number_input(
                    "HDBSCAN min cluster size",
                    min_value=2,
                    value=outlier_config.get("hdbscan_min_cluster_size", 5),
                )

                use_dbcv = st.checkbox(
                    "Use DBCV", value=outlier_config.get("use_dbcv", True)
                )

        st.session_state.outlier_config = {
            "contamination": contamination,
            "eps_percentile": eps_percentile,
            "dbscan_min_samples": dbscan_min_samples,
            "hdbscan_min_cluster_size": hdbscan_min_cluster,
            "use_dbcv": use_dbcv,
        }

        # SAVE CONFIGURATION
        st.markdown("---")
        st.subheader("💾 Save Configuration")

        col1, col2 = st.columns([3, 1])
        with col1:
            config_name = st.text_input(
                "Configuration name", placeholder="e.g., Food_Channel_Analysis"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save Config", type="secondary", use_container_width=True):
                if config_name:
                    try:
                        st.session_state.config_manager.save_config(
                            name=config_name,
                            filters=st.session_state.filter_config,
                            regression=st.session_state.reg_config,
                            kfold=st.session_state.kfold_config,
                            outlier=st.session_state.outlier_config,
                            description=f"Saved on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        st.success(f"✅ Saved: {config_name}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Enter a configuration name")

        st.markdown(
            '<div class="success-box">✅ Configuration complete! Go to Step 3 to run analysis.</div>',
            unsafe_allow_html=True,
        )


def page_run_analysis():
    """Page 3: Run the analysis."""
    st.header("▶️ Step 3: Run Analysis")

    if st.session_state.filtered_data is None:
        st.warning("⚠️ Please complete Steps 1 and 2 first")
        return

    if "reg_config" not in st.session_state:
        st.warning("⚠️ Please configure the model in Step 2")
        return

    st.markdown(
        '<div class="info-box">Click the button below to run the complete analysis pipeline</div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0, text="Starting analysis...")

            # Step 1: Build model
            progress_bar.progress(10, text="📊 Step 1/3: Building regression model...")
            time.sleep(0.3)

            reg_config = RegressionConfig(**st.session_state.reg_config)
            builder = RegressionPipelineBuilder(reg_config)
            X, y = builder.prepare_data(st.session_state.filtered_data)
            pipeline = builder.build()

            progress_bar.progress(25, text="✅ Model built successfully")
            time.sleep(0.2)

            # Step 2: K-Fold
            progress_bar.progress(
                30, text="🔄 Step 2/3: Running K-Fold cross-validation..."
            )

            kfold_config = KFoldConfig(**st.session_state.kfold_config)
            kfold_analyzer = KFoldAnalyzer(kfold_config)
            kfold_results = kfold_analyzer.run(
                st.session_state.filtered_data, X, y, pipeline
            )

            st.session_state.kfold_results = kfold_results
            progress_bar.progress(60, text="✅ K-Fold complete")
            time.sleep(0.2)

            # Step 3: Outlier detection
            progress_bar.progress(65, text="🎯 Step 3/3: Detecting outliers...")

            outlier_config = OutlierConfig(**st.session_state.outlier_config)
            outlier_analyzer = OutlierAnalyzer(outlier_config)
            outlier_results = outlier_analyzer.detect(
                accounts=kfold_results.get_all_accounts(),
                residuals=kfold_results.get_all_residuals(),
                weights=kfold_results.get_all_weights(),
            )

            st.session_state.outlier_results = outlier_results
            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.3)

            st.markdown(
                '<div class="success-box">🎉 Analysis Complete!</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Mean R²", f"{kfold_results.mean_r2:.4f}")
            with col2:
                st.metric("📉 Mean MSE", f"{kfold_results.mean_mse:.4f}")
            with col3:
                st.metric("🎯 Outliers", outlier_results.n_outliers)

            st.info("👉 Go to Step 4 to view detailed results")

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.exception(e)


def page_view_results():
    """Page 4: View and export results."""
    st.header("📊 Step 4: View Results")

    if st.session_state.outlier_results is None:
        st.warning("⚠️ Please run the analysis first (Step 3)")
        return

    results = st.session_state.outlier_results
    kfold_results = st.session_state.kfold_results

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Overview", "🎯 Outliers", "📊 3D Plot", "💾 Export"]
    )

    with tab1:
        st.subheader("Analysis Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏢 Total Accounts", results.n_accounts)
        with col2:
            st.metric("🎯 Outliers", results.n_outliers)
        with col3:
            st.metric("📊 Outlier Rate", f"{results.outlier_rate*100:.1f}%")
        with col4:
            st.metric("🔍 Method", results.density_method)

        st.subheader("K-Fold Results")
        fold_summary = kfold_results.get_fold_summary()
        st.dataframe(fold_summary, use_container_width=True)

        fig_mse = create_fold_mse_comparison(kfold_results)
        st.plotly_chart(fig_mse, use_container_width=True)

    with tab2:
        st.subheader("Detected Outlier Accounts")

        outlier_type_filter = st.selectbox(
            "Filter by type",
            options=["All"]
            + list(
                results.agg_data[results.agg_data["outlier_flag"] == 1][
                    "outlier_type"
                ].unique()
            ),
        )

        summary = results.get_outlier_summary()

        if outlier_type_filter != "All":
            summary = summary[summary["Outlier_Type"] == outlier_type_filter]

        st.dataframe(summary, use_container_width=True, height=400)

        # Account drill-down
        st.subheader("📋 Account Details")
        selected_account = st.selectbox(
            "Select account", options=[""] + list(summary["Account"].unique())
        )

        if selected_account:
            account_stats = summary[summary["Account"] == selected_account].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Type", account_stats["Outlier_Type"])
            with col2:
                st.metric("Mean Residual", f"{account_stats['w_mean_residual']:.4f}")
            with col3:
                st.metric("Std Residual", f"{account_stats['w_std_residual']:.4f}")
            with col4:
                st.metric("Observations", int(account_stats["count"]))

            st.info("💡 View detailed charts in Step 5: Account Charts")

        # Type breakdown
        st.subheader("📊 Outlier Type Breakdown")
        type_counts = summary["Outlier_Type"].value_counts()
        st.bar_chart(type_counts)

    with tab3:
        st.subheader("🎨 3D Interactive Visualization")
        st.info("💡 Use the dropdowns in the plot to change axes")

        fig_3d = create_interactive_3d_plot(results.agg_data)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab4:
        st.subheader("💾 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = results.get_outlier_summary().to_csv(index=False)
            st.download_button(
                label="📥 Outlier Summary",
                data=csv,
                file_name="outlier_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            csv_full = results.agg_data.to_csv(index=False)
            st.download_button(
                label="📥 Full Results",
                data=csv_full,
                file_name="full_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            csv_kfold = kfold_results.to_dataframe().to_csv(index=False)
            st.download_button(
                label="📥 K-Fold Results",
                data=csv_kfold,
                file_name="kfold_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


def page_account_charts():
    """Page 5: View account time series charts."""
    st.header("📈 Step 5: Account Charts")

    if st.session_state.filtered_data is None:
        st.warning("⚠️ Please complete Steps 1-3 first")
        return

    data = st.session_state.filtered_data

    # Get account list
    all_accounts = sorted(data["Account"].unique())

    # Filter options
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.outlier_results is not None:
            show_option = st.radio(
                "Show",
                ["All Accounts", "Outliers Only", "Normal Accounts Only"],
                horizontal=True,
            )

            if show_option == "Outliers Only":
                accounts_to_show = (
                    st.session_state.outlier_results.get_outlier_accounts()
                )
            elif show_option == "Normal Accounts Only":
                outlier_accounts = (
                    st.session_state.outlier_results.get_outlier_accounts()
                )
                accounts_to_show = [
                    acc for acc in all_accounts if acc not in outlier_accounts
                ]
            else:
                accounts_to_show = all_accounts
        else:
            accounts_to_show = all_accounts
            st.info("Run analysis first to filter by outliers")

    with col2:
        max_charts = st.number_input(
            "Max charts to display",
            min_value=1,
            max_value=100,
            value=min(20, len(accounts_to_show)),
            help="Limit to avoid performance issues",
        )

    st.markdown(
        f"**Showing {min(max_charts, len(accounts_to_show))} of {len(accounts_to_show)} accounts**"
    )

    # Get date range for consistency
    data["Date"] = pd.to_datetime(data["Date"])
    date_range = (data["Date"].min(), data["Date"].max())

    # Generate charts
    if st.button("📊 Generate Charts", type="primary"):
        with st.spinner(
            f"🎨 Creating {min(max_charts, len(accounts_to_show))} charts..."
        ):
            for i, account in enumerate(accounts_to_show[:max_charts]):
                with st.container():
                    # Show outlier badge if applicable
                    if st.session_state.outlier_results is not None:
                        outlier_accounts = (
                            st.session_state.outlier_results.get_outlier_accounts()
                        )
                        if account in outlier_accounts:
                            outlier_data = st.session_state.outlier_results.agg_data[
                                st.session_state.outlier_results.agg_data["Account"]
                                == account
                            ]
                            outlier_type = outlier_data["outlier_type"].values[0]
                            st.markdown(
                                f"### 🚨 {account} - **OUTLIER** ({outlier_type})"
                            )
                        else:
                            st.markdown(f"### ✅ {account}")
                    else:
                        st.markdown(f"### {account}")

                    fig = create_account_time_series(data, account, date_range)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")


if __name__ == "__main__":
    main()
