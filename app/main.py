"""
Streamlit web application for Account Outlier Detection.
Enhanced with loading animations, config management, optimal k-fold, and improved charts.
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
from src.utils.time_periods import (
    create_equal_periods,  # ADD THIS LINE
    create_time_periods,
    get_recommended_period_lengths,
    validate_period_coverage,
)
from src.visualization.plots import (
    create_account_time_series,
    create_all_account_charts,
    create_fold_mse_comparison,
    create_interactive_3d_plot,
    create_optimal_k_plot,
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "data": None,
        "filtered_data": None,
        "outlier_results": None,
        "kfold_results": None,
        "optimal_k": None,
        "config_manager": ConfigManager(),
        "current_page": "1️⃣ Upload Data",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


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

    # Header
    st.title("🔍 Account Outlier Detection")
    st.markdown("**Identify problematic accounts in CPG linear regression models**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select page",
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
        st.markdown("### 💾 Configuration")

        configs = st.session_state.config_manager.list_configs()

        if configs:
            config_names = [cfg.stem for cfg in configs]
            selected_config = st.selectbox("Saved configurations", [""] + config_names)

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
                        time.sleep(1)
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
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No saved configurations")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
        **Version:** 1.0.0
        **Author:** Cuong Bui
        **Contact:** [cuong@omniumcpg.com](mailto:cuong@omniumcpg.com)
        """
        )

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
    - Sales metrics (Unit_Sales, Dollar_Sales, etc.)
    - Base metrics (Auto_Base_Units, Auto_Base_Dollars)
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

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", f"{len(data):,}")
            with col2:
                st.metric("📋 Columns", len(data.columns))
            with col3:
                if "Account" in data.columns:
                    st.metric("🏢 Accounts", data["Account"].nunique())
            with col4:
                if "Date" in data.columns:
                    data["Date"] = pd.to_datetime(data["Date"])
                    date_range = (data["Date"].max() - data["Date"].min()).days
                    st.metric("📅 Date Range", f"{date_range} days")

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

    # SECTION 1: FILTERS
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

    # TIME PERIOD SELECTION (BEFORE FILTERING)
    st.subheader("⏰ Time Period Segmentation")

    st.info(
        "🗓️ Time periods are created on the full dataset before filtering to ensure consistent calendar alignment."
    )

    period_options = get_recommended_period_lengths(data, date_column="Date")

    if not period_options:
        st.error("❌ Insufficient data for time-based analysis")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Available Period Lengths:**")
            for opt in period_options:
                icon = "⭐" if opt["recommended"] else "✓" if opt["is_valid"] else "❌"
                st.write(
                    f"{icon} **{opt['weeks']} weeks** → {opt['num_periods']} periods"
                )

        with col2:
            valid_weeks = [opt["weeks"] for opt in period_options if opt["is_valid"]]

            if valid_weeks:
                default_weeks = st.session_state.get(
                    "period_weeks", 13 if 13 in valid_weeks else valid_weeks[0]
                )

                period_weeks = st.selectbox(
                    "Select period length",
                    options=valid_weeks,
                    index=(
                        valid_weeks.index(default_weeks)
                        if default_weeks in valid_weeks
                        else 0
                    ),
                    format_func=lambda x: f"{x} weeks",
                    key="period_weeks_selector",
                )

                st.session_state.period_weeks = period_weeks
            else:
                st.error("No valid period lengths")
                st.session_state.period_weeks = 13

        # Show selected period info
        selected_option = next(
            opt for opt in period_options if opt["weeks"] == period_weeks
        )
        st.success(f"📊 {selected_option['message']}")

        # Preview: Show CALENDAR BOUNDARIES (not data distribution)
        with st.expander("📅 Period Calendar Boundaries"):
            st.markdown(
                f"""
            The date range will be divided into **{period_weeks}-week periods** based on calendar time.

            Each period contains exactly **{period_weeks * 7} days** ({period_weeks} weeks).
            """
            )

            # Calculate boundaries
            min_date = pd.to_datetime(data["Date"].min())
            max_date = pd.to_datetime(data["Date"].max())
            period_days = period_weeks * 7

            boundaries = []
            current_date = min_date
            period_num = 1

            while current_date <= max_date:
                next_date = current_date + pd.Timedelta(days=period_days)
                end_date = min(next_date - pd.Timedelta(days=1), max_date)

                actual_days = (end_date - current_date).days + 1

                boundaries.append(
                    {
                        "Period": period_num,
                        "Start_Date": current_date.strftime("%Y-%m-%d"),
                        "End_Date": end_date.strftime("%Y-%m-%d"),
                        "Days": actual_days,
                        "Weeks": round(actual_days / 7, 1),
                    }
                )

                current_date = next_date
                period_num += 1

                # Safety: don't create infinite loops
                if period_num > 100:
                    break

            boundary_df = pd.DataFrame(boundaries)
            st.dataframe(boundary_df, use_container_width=True, height=300)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Periods", len(boundary_df))
            with col2:
                st.metric(
                    "Full Periods", (boundary_df["Weeks"] >= period_weeks * 0.95).sum()
                )
            with col3:
                last_period_weeks = boundary_df.iloc[-1]["Weeks"]
                st.metric("Last Period", f"{last_period_weeks} weeks")

            if last_period_weeks < period_weeks * 0.8:
                st.info(
                    f"ℹ️ The last period is shorter ({last_period_weeks} weeks) because your date range doesn't divide evenly by {period_weeks} weeks. This is normal."
                )

    # APPLY TIME PERIODS & FILTERS BUTTON
    if st.button(
        "✅ Apply Time Periods & Filters", type="primary", key="apply_filters_btn"
    ):

        with st.spinner("🔄 Processing..."):
            try:
                # STEP 1: Create periods on FULL data
                st.info("📅 Step 1/2: Creating time periods on full dataset...")

                data_with_periods = create_equal_periods(
                    data,
                    date_column="Date",
                    period_weeks=st.session_state.period_weeks,
                    period_column_name="_Auto_Time_Period",
                )

                n_periods = data_with_periods["_Auto_Time_Period"].nunique()
                st.success(
                    f"✅ Created {n_periods} time periods of {st.session_state.period_weeks} weeks"
                )

                # STEP 2: Apply filters
                st.info("🔍 Step 2/2: Applying filters...")

                filtered = apply_filters(
                    data_with_periods,
                    lower=lower_value,
                    cot=cot_value,
                    acv_threshold=acv_value,
                    auto_promo_code=promo_value,
                    exclude_zero_sales=exclude_zeros,
                )

                st.session_state.filtered_data = filtered

                # Save filter config
                st.session_state.filter_config = {
                    "use_lower": use_lower if "use_lower" in locals() else False,
                    "lower_value": lower_value,
                    "use_acv": use_acv if "use_acv" in locals() else False,
                    "acv_value": acv_value,
                    "use_cot": use_cot if "use_cot" in locals() else False,
                    "cot_value": cot_value,
                    "use_promo": use_promo if "use_promo" in locals() else False,
                    "promo_value": promo_value,
                    "exclude_zeros": exclude_zeros,
                }

                # Show summary
                summary = get_filter_summary(data_with_periods, filtered)

                st.markdown(
                    f'<div class="success-box">✅ Filtered to {len(filtered):,} rows ({summary["percent_removed"]:.1f}% removed)</div>',
                    unsafe_allow_html=True,
                )

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Original Rows", f"{summary['rows_original']:,}")
                with col2:
                    st.metric("✅ Filtered Rows", f"{summary['rows_filtered']:,}")
                with col3:
                    st.metric(
                        "📅 Periods Remaining", filtered["_Auto_Time_Period"].nunique()
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

    # SECTION 2: REGRESSION MODEL
    if st.session_state.filtered_data is not None:
        st.subheader("2️⃣ Regression Model")

        filtered = st.session_state.filtered_data
        reg_config = st.session_state.get("reg_config", {})

        with st.expander("📈 Configure Model Variables", expanded=True):
            numeric_cols = [
                col
                for col in filtered.columns
                if filtered[col].dtype in ["int64", "float64"]
            ]
            categorical_cols = [
                col for col in filtered.columns if filtered[col].dtype == "object"
            ]

            dep_var = st.selectbox(
                "Dependent Variable (Y)",
                options=numeric_cols,
                index=(
                    numeric_cols.index(reg_config.get("dependent_var", numeric_cols[0]))
                    if reg_config.get("dependent_var") in numeric_cols
                    else 0
                ),
            )

            continuous_vars = st.multiselect(
                "Continuous Variables (log-transformed)",
                options=[col for col in numeric_cols if col != dep_var],
                default=reg_config.get(
                    "continuous_vars",
                    [numeric_cols[1]] if len(numeric_cols) > 1 else [],
                ),
            )

            categorical_vars = st.multiselect(
                "Categorical Variables (one-hot encoded)",
                options=categorical_cols,
                default=reg_config.get(
                    "categorical_vars",
                    ["Account"] if "Account" in categorical_cols else [],
                ),
            )

            if not continuous_vars and not categorical_vars:
                st.warning("⚠️ Select at least one independent variable")

        st.session_state.reg_config = {
            "dependent_var": dep_var,
            "continuous_vars": continuous_vars,
            "categorical_vars": categorical_vars,
        }

        # SECTION 3: K-FOLD (UPDATED)
        st.subheader("3️⃣ K-Fold Cross-Validation")

        kfold_config = st.session_state.get("kfold_config", {})

        with st.expander("🔄 Configure K-Fold", expanded=True):

            # # TIME PERIOD SEGMENTATION
            # st.markdown("#### 📅 Time Period Segmentation")

            # period_options = get_recommended_period_lengths(
            #     filtered, date_column="Date"
            # )

            # if not period_options:
            #     st.error("❌ Insufficient data for time-based k-fold analysis")
            # else:
            #     # Show options
            #     col1, col2 = st.columns([2, 1])

            #     with col1:
            #         st.markdown("**Available Period Lengths:**")

            #         for opt in period_options:
            #             icon = (
            #                 "⭐"
            #                 if opt["recommended"]
            #                 else "✓" if opt["is_valid"] else "❌"
            #             )
            #             st.write(
            #                 f"{icon} **{opt['weeks']} weeks** → {opt['num_periods']} periods"
            #             )

            #     with col2:
            #         valid_weeks = [
            #             opt["weeks"] for opt in period_options if opt["is_valid"]
            #         ]

            #         if valid_weeks:
            #             default_weeks = kfold_config.get(
            #                 "period_weeks", 13 if 13 in valid_weeks else valid_weeks[0]
            #             )

            #             period_weeks = st.selectbox(
            #                 "Select period length",
            #                 options=valid_weeks,
            #                 index=(
            #                     valid_weeks.index(default_weeks)
            #                     if default_weeks in valid_weeks
            #                     else 0
            #                 ),
            #                 format_func=lambda x: f"{x} weeks",
            #             )

            #             # IMPORTANT: Always store this, even if user doesn't click anything
            #             st.session_state.period_weeks = period_weeks
            #         else:
            #             st.error("No valid period lengths available")
            #             # Set a default
            #             st.session_state.period_weeks = 13
            #             return

            #     # Show selected period info
            #     selected_option = next(
            #         opt for opt in period_options if opt["weeks"] == period_weeks
            #     )
            #     st.info(f"📊 {selected_option['message']}")

            #     # Preview period distribution
            #     with st.expander("👁️ Preview Period Distribution"):
            #         # Use create_equal_periods for the preview
            #         preview_data = create_equal_periods(
            #             filtered,
            #             date_column="Date",
            #             period_weeks=period_weeks,
            #             period_column_name="_preview_period",
            #         )

            #         # Create summary
            #         period_summary = []
            #         for period in sorted(preview_data["_preview_period"].unique()):
            #             period_data = preview_data[
            #                 preview_data["_preview_period"] == period
            #             ]

            #             period_min = period_data["Date"].min()
            #             period_max = period_data["Date"].max()

            #             # Calculate exact weeks
            #             days_span = (period_max - period_min).days + 1
            #             weeks_span = days_span / 7

            #             period_summary.append(
            #                 {
            #                     "Period": period,
            #                     "Start_Date": period_min,
            #                     "End_Date": period_max,
            #                     "Num_Records": len(period_data),
            #                     "Days": days_span,
            #                     "Weeks": round(weeks_span, 1),
            #                 }
            #             )

            #         summary_df = pd.DataFrame(period_summary)
            #         st.dataframe(summary_df, use_container_width=True)

            #         # Show statistics
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             st.metric("Total Periods", len(summary_df))
            #         with col2:
            #             st.metric(
            #                 "Avg Weeks/Period", f"{summary_df['Weeks'].mean():.1f}"
            #             )
            #         with col3:
            #             st.metric("Target Weeks", period_weeks)

            # Optimal K finder
            st.markdown("#### 🎯 Optimal K Selection")

            st.info("💡 The system can automatically find the optimal number of folds.")

            if st.button("🔍 Find Optimal K", use_container_width=True):
                if continuous_vars or categorical_vars:
                    # Create time periods BEFORE finding optimal k
                    filtered_with_periods = create_equal_periods(
                        filtered,
                        date_column="Date",
                        period_weeks=st.session_state.period_weeks,
                        period_column_name="_Auto_Time_Period",
                    )
                    st.session_state.filtered_data_with_periods = filtered_with_periods

                    find_optimal_k(
                        filtered_with_periods,
                        dep_var,
                        continuous_vars,
                        categorical_vars,
                    )
                else:
                    st.error("Configure model variables first!")

        # SHOW OPTIMAL K RESULTS OUTSIDE EXPANDER
        if st.session_state.optimal_k is not None:
            st.success(f"✅ Suggested optimal K: **{st.session_state.optimal_k}**")

            fig = create_optimal_k_plot(st.session_state.optimal_k_results)
            st.plotly_chart(fig, use_container_width=True)

        # K-FOLD SETTINGS
        with st.expander("🔄 Configure K-Fold", expanded=True):
            st.markdown("#### ⚙️ K-Fold Settings")

            col1, col2 = st.columns(2)

            with col1:
                # Get the number of periods available
                if "period_weeks" in st.session_state:
                    period_opts = get_recommended_period_lengths(filtered, "Date")
                    selected_opt = next(
                        opt
                        for opt in period_opts
                        if opt["weeks"] == st.session_state.period_weeks
                    )
                    max_k = selected_opt["num_periods"]
                else:
                    max_k = 10

                default_k = (
                    st.session_state.optimal_k
                    if st.session_state.optimal_k
                    else kfold_config.get("n_splits", 5)
                )
                default_k = min(default_k, max_k)  # Don't exceed available periods

                n_splits = st.slider(
                    "Number of folds (K)",
                    min_value=2,
                    max_value=max_k,
                    value=default_k,
                    help=f"Max K = {max_k} (based on {st.session_state.get('period_weeks', 13)}-week periods)",
                )

                account_col = st.selectbox(
                    "Account column",
                    options=categorical_cols,
                    index=(
                        categorical_cols.index("Account")
                        if "Account" in categorical_cols
                        else 0
                    ),
                )

            with col2:
                weight_options = ["None"] + [col for col in numeric_cols]
                default_weight = kfold_config.get("weight_column", "Auto_Base_Units")
                weight_col = st.selectbox(
                    "Weight column (optional)",
                    options=weight_options,
                    index=(
                        weight_options.index(default_weight)
                        if default_weight in weight_options
                        else 0
                    ),
                )

                st.info(
                    f"📊 Using auto-generated {st.session_state.get('period_weeks', 13)}-week periods"
                )

        st.session_state.kfold_config = {
            "n_splits": n_splits,
            "period_weeks": st.session_state.get(
                "period_weeks", 13
            ),  # Always include this
            "group_column": "_Auto_Time_Period",  # Use auto-generated column
            "account_column": account_col,
            "weight_column": None if weight_col == "None" else weight_col,
        }

        # SECTION 4: OUTLIER DETECTION
        st.subheader("4️⃣ Outlier Detection")

        outlier_config = st.session_state.get("outlier_config", {})

        with st.expander("🎯 Configure Outlier Detection", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                contamination = st.slider(
                    "Expected outlier proportion",
                    min_value=0.01,
                    max_value=0.30,
                    value=outlier_config.get("contamination", 0.10),
                    step=0.01,
                    help="Expected % of outliers (Isolation Forest)",
                )

                eps_percentile = st.slider(
                    "DBSCAN eps percentile",
                    min_value=50,
                    max_value=99,
                    value=outlier_config.get("eps_percentile", 90),
                    help="Higher = larger neighborhood",
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
                    "Use DBCV for method selection",
                    value=outlier_config.get("use_dbcv", True),
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
                "Configuration name",
                value="",
                placeholder="e.g., Walmart_Food_Analysis",
            )
            config_desc = st.text_input(
                "Description (optional)",
                value="",
                placeholder="e.g., Standard analysis for Walmart food channel",
            )

        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save Config", type="secondary", use_container_width=True):
                if config_name:
                    try:
                        st.session_state.config_manager.save_config(
                            name=config_name,
                            description=config_desc,
                            filters=st.session_state.filter_config,
                            regression=st.session_state.reg_config,
                            kfold=st.session_state.kfold_config,
                            outlier=st.session_state.outlier_config,
                        )
                        st.success(f"✅ Saved: {config_name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Enter a configuration name")

        st.markdown(
            '<div class="info-box">✅ Configuration complete! Go to Step 3 to run analysis.</div>',
            unsafe_allow_html=True,
        )


def find_optimal_k(data, dep_var, continuous_vars, categorical_vars):
    """Find optimal K using k-fold analysis."""

    # Verify periods exist
    if "_Auto_Time_Period" not in data.columns:
        st.error("❌ Time periods not found. Please apply filters first.")
        return

    with st.spinner("🔍 Finding optimal K (testing k=2 to k=10)..."):
        try:
            # Build model
            reg_config = RegressionConfig(
                dependent_var=dep_var,
                continuous_vars=continuous_vars,
                categorical_vars=categorical_vars,
            )

            builder = RegressionPipelineBuilder(reg_config)
            X, y = builder.prepare_data(data)
            pipeline = builder.build()

            # Get config
            account_col = st.session_state.kfold_config.get("account_column", "Account")
            weight_col = st.session_state.kfold_config.get("weight_column")

            # Determine max K based on periods in filtered data
            max_k = data["_Auto_Time_Period"].nunique()
            k_range = range(2, min(11, max_k + 1))

            # Test different K values
            temp_config = KFoldConfig(
                n_splits=5,
                group_column="_Auto_Time_Period",
                account_column=account_col,
                weight_column=weight_col,
                verbose=False,
            )

            analyzer = KFoldAnalyzer(temp_config)
            optimal_k, k_results = analyzer.find_optimal_k(
                data, X, y, pipeline, k_range=k_range
            )

            # Store results
            st.session_state.optimal_k = optimal_k
            st.session_state.optimal_k_results = k_results

            # Force rerun
            st.rerun()

        except Exception as e:
            st.error(f"Error finding optimal K: {e}")
            st.exception(e)


def page_run_analysis():
    """Page 3: Run the analysis."""
    st.header("▶️ Step 3: Run Analysis")

    # Check prerequisites
    if st.session_state.filtered_data is None:
        st.warning("⚠️ Complete Steps 1 and 2 first")
        return

    if "reg_config" not in st.session_state:
        st.warning("⚠️ Configure the model in Step 2")
        return

    st.markdown(
        '<div class="info-box">📊 Ready to run complete analysis pipeline</div>',
        unsafe_allow_html=True,
    )

    # Show configuration summary
    with st.expander("📋 Configuration Summary"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data:**")
            st.write(f"- Rows: {len(st.session_state.filtered_data):,}")
            st.write(
                f"- Accounts: {st.session_state.filtered_data['Account'].nunique()}"
            )

            st.markdown("**Model:**")
            st.write(f"- Dependent: {st.session_state.reg_config['dependent_var']}")
            st.write(
                f"- Continuous: {len(st.session_state.reg_config['continuous_vars'])}"
            )
            st.write(
                f"- Categorical: {len(st.session_state.reg_config['categorical_vars'])}"
            )

        with col2:
            st.markdown("**K-Fold:**")
            st.write(f"- Folds: {st.session_state.kfold_config['n_splits']}")
            st.write(f"- Group by: {st.session_state.kfold_config['group_column']}")

            st.markdown("**Outlier Detection:**")
            st.write(
                f"- Contamination: {st.session_state.outlier_config['contamination']:.0%}"
            )
            st.write(
                f"- DBSCAN eps: p{st.session_state.outlier_config['eps_percentile']}"
            )

    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        run_complete_analysis()


def run_complete_analysis():
    """Execute the complete analysis pipeline with progress tracking."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Verify periods exist
        if "_Auto_Time_Period" not in st.session_state.filtered_data.columns:
            st.error(
                "❌ Time periods not found. Please go back to Configure Analysis and apply filters."
            )
            return

        filtered_with_periods = st.session_state.filtered_data
        period_weeks = st.session_state.get("period_weeks", 13)
        n_periods = filtered_with_periods["_Auto_Time_Period"].nunique()

        status_text.markdown(
            f"### ✅ Using {n_periods} periods of {period_weeks} weeks"
        )
        progress_bar.progress(10)
        time.sleep(0.5)

        # STEP 1: Build Model
        status_text.markdown("### 🔨 Step 1/3: Building regression model...")
        progress_bar.progress(15)
        time.sleep(0.3)

        reg_config = RegressionConfig(**st.session_state.reg_config)
        builder = RegressionPipelineBuilder(reg_config)
        X, y = builder.prepare_data(filtered_with_periods)
        pipeline = builder.build()

        progress_bar.progress(30)
        status_text.markdown("✅ Model built successfully")
        time.sleep(0.5)

        # STEP 2: K-Fold
        status_text.markdown("### 🔄 Step 2/3: Running K-Fold cross-validation...")
        progress_bar.progress(35)

        # Create KFold config
        kfold_config_dict = {
            "n_splits": st.session_state.kfold_config["n_splits"],
            "group_column": "_Auto_Time_Period",
            "account_column": st.session_state.kfold_config["account_column"],
            "weight_column": st.session_state.kfold_config["weight_column"],
            "shuffle": st.session_state.kfold_config.get("shuffle", True),
            "random_state": st.session_state.kfold_config.get("random_state", 42),
            "verbose": st.session_state.kfold_config.get("verbose", True),
        }

        kfold_config = KFoldConfig(**kfold_config_dict)
        kfold_analyzer = KFoldAnalyzer(kfold_config)

        with st.spinner(f"Running {kfold_config.n_splits}-fold cross-validation..."):
            kfold_results = kfold_analyzer.run(filtered_with_periods, X, y, pipeline)

        st.session_state.kfold_results = kfold_results
        progress_bar.progress(65)
        status_text.markdown(
            f"✅ K-Fold complete: MSE={kfold_results.mean_mse:.4f}, R²={kfold_results.mean_r2:.4f}"
        )
        time.sleep(0.5)

        # STEP 3: Outlier Detection
        status_text.markdown("### 🎯 Step 3/3: Detecting outlier accounts...")
        progress_bar.progress(70)

        outlier_config = OutlierConfig(**st.session_state.outlier_config)
        outlier_analyzer = OutlierAnalyzer(outlier_config)

        with st.spinner("Running outlier detection algorithms..."):
            outlier_results = outlier_analyzer.detect(
                accounts=kfold_results.get_all_accounts(),
                residuals=kfold_results.get_all_residuals(),
                weights=kfold_results.get_all_weights(),
            )

        st.session_state.outlier_results = outlier_results
        progress_bar.progress(100)
        status_text.markdown("✅ Outlier detection complete!")

        # Show summary
        time.sleep(0.5)
        st.balloons()

        st.markdown(
            '<div class="success-box"><h3>🎉 Analysis Complete!</h3></div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Mean R²", f"{kfold_results.mean_r2:.4f}")
        with col2:
            st.metric("📉 Mean MSE", f"{kfold_results.mean_mse:.4f}")
        with col3:
            st.metric("🎯 Outliers", outlier_results.n_outliers)
        with col4:
            st.metric("📈 Outlier Rate", f"{outlier_results.outlier_rate*100:.1f}%")

        st.info("👉 Go to **Step 4** to view detailed results and visualizations")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during analysis: {e}")
        st.exception(e)


def page_view_results():
    """Page 4: View and export results."""
    st.header("📊 Step 4: View Results")

    if st.session_state.outlier_results is None:
        st.warning("⚠️ Run the analysis first (Step 3)")
        return

    results = st.session_state.outlier_results
    kfold_results = st.session_state.kfold_results

    # Get period info if available
    period_weeks = st.session_state.kfold_config.get("period_weeks", 13)
    n_splits = st.session_state.kfold_config.get("n_splits", 5)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Overview", "🎯 Outliers", "📊 3D Visualization", "💾 Export"]
    )

    with tab1:
        st.subheader("Analysis Overview")

        # Show configuration used
        with st.expander("⚙️ Analysis Configuration"):
            st.markdown(
                f"""
            **Time Segmentation:**
            - Period Length: {period_weeks} weeks
            - Number of Periods: {st.session_state.filtered_data_with_periods['_Auto_Time_Period'].nunique() if 'filtered_data_with_periods' in st.session_state else 'N/A'}
            - K-Folds: {n_splits}

            **Model:**
            - Dependent Variable: {st.session_state.reg_config['dependent_var']}
            - Continuous Variables: {len(st.session_state.reg_config['continuous_vars'])}
            - Categorical Variables: {len(st.session_state.reg_config['categorical_vars'])}

            **Outlier Detection:**
            - Contamination: {st.session_state.outlier_config['contamination']:.0%}
            - Method Used: {results.density_method}
            """
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Accounts", results.n_accounts)
        with col2:
            st.metric("Outliers Found", results.n_outliers)
        with col3:
            st.metric("Outlier Rate", f"{results.outlier_rate*100:.1f}%")
        with col4:
            st.metric("Method Used", results.density_method)

        st.subheader("K-Fold Results")
        fold_summary = kfold_results.get_fold_summary()
        st.dataframe(fold_summary, use_container_width=True)

        fig_mse = create_fold_mse_comparison(kfold_results)
        st.plotly_chart(fig_mse, use_container_width=True)

        if st.session_state.get("optimal_k_results") is not None:
            st.subheader("Optimal K Analysis")
            fig_k = create_optimal_k_plot(st.session_state.optimal_k_results)
            st.plotly_chart(fig_k, use_container_width=True)

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

        # Type breakdown
        st.subheader("Outlier Type Breakdown")
        type_counts = results.agg_data[results.agg_data["outlier_flag"] == 1][
            "outlier_type"
        ].value_counts()
        st.bar_chart(type_counts)

    with tab3:
        st.subheader("3D Interactive Visualization")

        st.markdown(
            """
        **How to use:**
        - 🖱️ Click and drag to rotate
        - 🔍 Scroll to zoom
        - 📊 Use dropdowns to change axes
        - 🎨 Colors indicate outlier type
        """
        )

        fig_3d = create_interactive_3d_plot(results.agg_data)
        st.plotly_chart(
            fig_3d, use_container_width=True, config={"displayModeBar": True}
        )

    with tab4:
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📥 Data Exports")

            # Outlier summary
            csv1 = results.get_outlier_summary().to_csv(index=False)
            st.download_button(
                "📄 Outlier Summary (CSV)",
                data=csv1,
                file_name="outlier_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Full results
            csv2 = results.agg_data.to_csv(index=False)
            st.download_button(
                "📄 Full Results (CSV)",
                data=csv2,
                file_name="full_analysis_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # K-Fold results
            csv3 = kfold_results.to_dataframe().to_csv(index=False)
            st.download_button(
                "📄 K-Fold Details (CSV)",
                data=csv3,
                file_name="kfold_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.markdown("#### 📊 Visualization Exports")

            st.info("💡 Right-click on any chart and select 'Save as PNG' to export")

            # Export 3D plot as HTML
            fig_3d = create_interactive_3d_plot(results.agg_data)
            html_str = fig_3d.to_html()
            st.download_button(
                "🌐 3D Plot (Interactive HTML)",
                data=html_str,
                file_name="3d_outlier_plot.html",
                mime="text/html",
                use_container_width=True,
            )


def page_account_charts():
    """Page 5: View account time series charts."""
    st.header("📈 Step 5: Account Time Series Charts")

    # USE ORIGINAL DATA, NOT FILTERED DATA
    if st.session_state.data is None:
        st.warning("⚠️ Upload data first")
        return

    # Use unfiltered data for charts to show full picture
    data = st.session_state.data.copy()

    # Apply ONLY essential filters (keep outlier-relevant accounts)
    if st.session_state.filtered_data is not None:
        # Get the accounts that passed filtering
        filtered_accounts = st.session_state.filtered_data["Account"].unique()
        # Show only those accounts, but with ALL their data (including promos)
        data = data[data["Account"].isin(filtered_accounts)]
        st.info(
            f"📊 Showing full data (including promos) for {len(filtered_accounts)} filtered accounts"
        )

    # Get list of accounts
    accounts = sorted(data["Account"].unique())

    st.markdown(f"**Total Accounts:** {len(accounts)}")

    # Debug panel
    with st.expander("🔍 Debug: Show Column Names"):
        diagnose_data_columns(data)

    # Configuration in sidebar or top section
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        chart_mode = st.selectbox(
            "View mode", ["Single Account", "All Outliers", "All Accounts"]
        )

    with col2:
        if chart_mode == "Single Account":
            selected_account = st.selectbox("Select account", accounts)
            st.session_state.selected_account = selected_account
        elif chart_mode == "All Outliers":
            if st.session_state.outlier_results:
                outlier_accounts = (
                    st.session_state.outlier_results.get_outlier_accounts()
                )
                st.info(f"{len(outlier_accounts)} outlier accounts")
            else:
                st.warning("Run analysis first")
                return
        else:  # All Accounts
            max_charts = st.number_input(
                "Max charts", min_value=1, max_value=100, value=20
            )
            st.session_state.max_charts = max_charts

    with col3:
        if st.button("📊 Generate Charts", type="primary", use_container_width=True):
            st.session_state.chart_mode = chart_mode

    # Display charts at FULL WIDTH (not in columns)
    if st.session_state.get("chart_mode"):
        st.markdown("---")
        display_charts(data, st.session_state.chart_mode, accounts)


def display_charts(data, chart_mode, accounts):
    """Display account charts at full width."""

    # Get date range for consistency
    data["Date"] = pd.to_datetime(data["Date"])
    date_range = (data["Date"].min(), data["Date"].max())

    if chart_mode == "Single Account":
        selected_account = st.session_state.get("selected_account")

        if selected_account:
            st.subheader(f"📊 {selected_account}")

            # Create chart at FULL WIDTH
            fig = create_account_time_series(data, selected_account, date_range)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No account selected")

    elif chart_mode == "All Outliers":
        outlier_accounts = st.session_state.outlier_results.get_outlier_accounts()

        st.markdown(f"### Showing {len(outlier_accounts)} Outlier Accounts")
        st.markdown("---")

        for i, account in enumerate(outlier_accounts):
            # Show header
            st.markdown(f"#### 📊 {account}")

            # Chart at full width
            fig = create_account_time_series(data, account, date_range)
            st.plotly_chart(fig, use_container_width=True)

            # Separator
            if i < len(outlier_accounts) - 1:
                st.markdown("---")

    else:  # All Accounts
        max_charts = st.session_state.get("max_charts", 20)

        st.markdown(
            f"### Showing {min(max_charts, len(accounts))} of {len(accounts)} Accounts"
        )
        st.markdown("---")

        progress = st.progress(0)
        chart_container = st.container()

        with chart_container:
            for i, account in enumerate(accounts[:max_charts]):
                # Show header
                st.markdown(f"#### 📊 {account}")

                # Chart at full width
                fig = create_account_time_series(data, account, date_range)
                st.plotly_chart(fig, use_container_width=True)

                # Update progress
                progress.progress((i + 1) / min(max_charts, len(accounts)))

                # Separator
                if i < min(max_charts, len(accounts)) - 1:
                    st.markdown("---")

        progress.empty()

    st.success(f"✅ Charts displayed successfully")


def generate_charts(data, chart_mode, accounts):
    """Generate account charts based on selected mode."""

    # Get date range for consistency
    data["Date"] = pd.to_datetime(data["Date"])
    date_range = (data["Date"].min(), data["Date"].max())

    with st.spinner("Generating charts..."):

        if chart_mode == "Single Account":
            # GET the stored selection
            selected_account = st.session_state.get("selected_account")

            if selected_account:
                st.subheader(f"📊 {selected_account}")
                fig = create_account_time_series(data, selected_account, date_range)
                st.plotly_chart(
                    fig, use_container_width=True, key=f"chart_{selected_account}"
                )
            else:
                st.error("No account selected")

        elif chart_mode == "All Outliers":
            outlier_accounts = st.session_state.outlier_results.get_outlier_accounts()

            st.markdown(f"### Showing {len(outlier_accounts)} Outlier Accounts")

            for i, account in enumerate(outlier_accounts):
                with st.expander(f"📊 {account}", expanded=(i < 3)):
                    fig = create_account_time_series(data, account, date_range)
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"chart_{account}_{i}"
                    )

        else:  # All Accounts
            max_charts = st.session_state.get("max_charts", 20)

            st.markdown(
                f"### Showing {min(max_charts, len(accounts))} of {len(accounts)} Accounts"
            )

            progress = st.progress(0)

            for i, account in enumerate(accounts[:max_charts]):
                with st.expander(f"📊 {account}", expanded=False):
                    fig = create_account_time_series(data, account, date_range)
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"chart_{account}_{i}"
                    )

                progress.progress((i + 1) / min(max_charts, len(accounts)))

            progress.empty()

    st.success(f"✅ Charts generated successfully")


def diagnose_data_columns(data):
    """Helper function to diagnose available columns."""
    st.markdown("### 🔍 Data Column Diagnosis")

    st.markdown("#### All Available Columns:")

    # Show columns in categories
    unit_cols = [col for col in data.columns if "unit" in col.lower()]
    dollar_cols = [col for col in data.columns if "dollar" in col.lower()]
    base_cols = [col for col in data.columns if "base" in col.lower()]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Unit Columns:**")
        for col in unit_cols:
            st.write(f"- `{col}`")

    with col2:
        st.markdown("**Dollar Columns:**")
        for col in dollar_cols:
            st.write(f"- `{col}`")

    with col3:
        st.markdown("**Base Columns:**")
        for col in base_cols:
            st.write(f"- `{col}`")

    st.markdown("#### Sample Data for First Account:")

    if "Account" in data.columns:
        sample_account = data["Account"].iloc[0]
        sample_data = data[data["Account"] == sample_account].head(5).copy()

        # Show key columns
        key_cols = ["Date", "Account"]

        # Add unit/sales columns
        for col in data.columns:
            if any(
                keyword in col.lower()
                for keyword in [
                    "unit_sales",
                    "auto_base_units",
                    "dollar_sales",
                    "auto_base_dollars",
                ]
            ):
                if col not in key_cols:
                    key_cols.append(col)

        available_cols = [col for col in key_cols if col in sample_data.columns]

        if available_cols:
            st.dataframe(sample_data[available_cols], use_container_width=True)

            # VALUE COMPARISON
            st.markdown("#### 🔍 Value Comparison:")

            # Find the EXACT columns that will be used in the chart
            unit_sales_col = None
            for col in ["Unit_Sales", "UnitSales", "Unit Sales"]:
                if col in sample_data.columns:
                    unit_sales_col = col
                    break

            if not unit_sales_col:
                for col in sample_data.columns:
                    col_lower = col.lower()
                    if (
                        "unit" in col_lower
                        and "sale" in col_lower
                        and "base" not in col_lower
                        and "any_merch" not in col_lower
                        and "year_ago" not in col_lower
                    ):
                        unit_sales_col = col
                        break

            base_units_col = None
            for col in ["Auto_Base_Units", "Base_Units", "BaseUnits", "Base Units"]:
                if col in sample_data.columns:
                    base_units_col = col
                    break

            st.markdown(f"**Chart will use:**")
            st.markdown(f"- Unit Sales: `{unit_sales_col}`")
            st.markdown(f"- Base Units: `{base_units_col}`")

            if unit_sales_col and base_units_col:
                comparison = pd.DataFrame(
                    {
                        "Metric": ["Min", "Max", "Mean", "Sum"],
                        unit_sales_col: [
                            sample_data[unit_sales_col].min(),
                            sample_data[unit_sales_col].max(),
                            sample_data[unit_sales_col].mean(),
                            sample_data[unit_sales_col].sum(),
                        ],
                        base_units_col: [
                            sample_data[base_units_col].min(),
                            sample_data[base_units_col].max(),
                            sample_data[base_units_col].mean(),
                            sample_data[base_units_col].sum(),
                        ],
                    }
                )
                st.dataframe(comparison, use_container_width=True)

                # Check if they're identical
                if sample_data[unit_sales_col].equals(sample_data[base_units_col]):
                    st.error("⚠️ WARNING: These columns have IDENTICAL values!")
                else:
                    st.success("✅ Unit_Sales and Base_Units have different values")

                    # Show difference percentage
                    diff_pct = (
                        abs(
                            sample_data[unit_sales_col].sum()
                            - sample_data[base_units_col].sum()
                        )
                        / sample_data[unit_sales_col].sum()
                        * 100
                    )
                    st.info(f"📊 Difference: {diff_pct:.1f}%")


if __name__ == "__main__":
    main()
