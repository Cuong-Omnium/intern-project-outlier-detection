"""
Visualization utilities for outlier detection results.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def create_3d_residual_plot(
    agg_data: pd.DataFrame,
    x_col: str = "w_mean_residual",
    y_col: str = "w_std_residual",
    z_col: str = "w_skew_residual",
    color_col: str = "outlier_type",
    hover_data: Optional[list] = None,
    title: str = "Residual Feature Space - Outlier Detection",
) -> go.Figure:
    """
    Create 3D scatter plot of residual features with outlier highlighting.

    Args:
        agg_data: Aggregated data with residual statistics and outlier flags
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        z_col: Column name for z-axis
        color_col: Column name for color coding
        hover_data: Additional columns to show on hover
        title: Plot title

    Returns:
        Plotly Figure object

    Example:
        >>> fig = create_3d_residual_plot(result.agg_data)
        >>> fig.show()
    """
    logger.info("Creating 3D residual plot...")

    # Define color mapping
    color_map = {
        "both": "#d62728",  # Red - high confidence outlier
        "dbscan_only": "#9467bd",  # Purple - density outlier
        "hdbscan_only": "#9467bd",  # Purple - density outlier
        "iforest_only": "#1f77b4",  # Blue - isolation outlier
        "normal": "#7f7f7f",  # Gray - normal
    }

    # Map colors
    colors = agg_data[color_col].map(color_map).fillna("#7f7f7f")

    # Create hover text
    if hover_data is None:
        hover_data = ["w_mean_residual", "w_std_residual", "w_skew_residual", "count"]

    hover_text = []
    for _, row in agg_data.iterrows():
        text = f"<b>{row['Account']}</b><br>"
        text += f"Type: {row[color_col]}<br>"
        for col in hover_data:
            if col in row:
                text += f"{col}: {row[col]:.4f}<br>"
        hover_text.append(text)

    # Create scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=agg_data[x_col],
                y=agg_data[y_col],
                z=agg_data[z_col],
                mode="markers",
                marker=dict(
                    size=6,
                    color=colors,
                    opacity=0.9,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Accounts",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(
                title=x_col,
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
            yaxis=dict(
                title=y_col,
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
            zaxis=dict(
                title=z_col,
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Add legend manually
    legend_items = [
        ("Both Methods", "#d62728"),
        ("Density Only", "#9467bd"),
        ("Isolation Forest Only", "#1f77b4"),
        ("Normal", "#7f7f7f"),
    ]

    for name, color in legend_items:
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=8, color=color),
                name=name,
                showlegend=True,
            )
        )

    logger.info("3D plot created successfully")
    return fig


def create_interactive_3d_plot(
    agg_data: pd.DataFrame, available_metrics: Optional[list] = None
) -> go.Figure:
    """
    Create 3D plot with dropdown menus to change axes.

    This recreates your original interactive plot functionality where users
    can select which metrics to display on each axis.

    Args:
        agg_data: Aggregated data with residual statistics
        available_metrics: List of metric columns to allow in dropdowns

    Returns:
        Plotly Figure with interactive dropdowns

    Example:
        >>> fig = create_interactive_3d_plot(result.agg_data)
        >>> fig.show()
    """
    if available_metrics is None:
        available_metrics = [
            "w_mean_residual",
            "z_mean_residual",
            "w_std_residual",
            "w_skew_residual",
            "mad_residual",
        ]

    # Filter to only available columns
    available_metrics = [m for m in available_metrics if m in agg_data.columns]

    # Default axes
    default_x, default_y, default_z = (
        available_metrics[0],
        available_metrics[2],
        available_metrics[3],
    )

    # Create base figure
    fig = create_3d_residual_plot(
        agg_data,
        x_col=default_x,
        y_col=default_y,
        z_col=default_z,
        title="Residual Feature Space (Choose metrics for each axis)",
    )

    # Create dropdown buttons for each axis
    def create_axis_buttons(axis_name, current_metric):
        buttons = []
        for metric in available_metrics:
            update_dict = {}
            relayout_dict = {}

            if axis_name == "x":
                update_dict["x"] = [agg_data[metric]]
                relayout_dict["scene.xaxis.title.text"] = metric
            elif axis_name == "y":
                update_dict["y"] = [agg_data[metric]]
                relayout_dict["scene.yaxis.title.text"] = metric
            else:  # z
                update_dict["z"] = [agg_data[metric]]
                relayout_dict["scene.zaxis.title.text"] = metric

            buttons.append(
                dict(label=metric, method="update", args=[update_dict, relayout_dict])
            )

        return buttons

    # Add dropdown menus
    fig.update_layout(
        updatemenus=[
            # X-axis dropdown
            dict(
                buttons=create_axis_buttons("x", default_x),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.95,
                yanchor="top",
                bgcolor="rgba(245,245,245,0.9)",
                bordercolor="rgba(180,180,180,0.6)",
            ),
            # Y-axis dropdown
            dict(
                buttons=create_axis_buttons("y", default_y),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.75,
                yanchor="top",
                bgcolor="rgba(245,245,245,0.9)",
                bordercolor="rgba(180,180,180,0.6)",
            ),
            # Z-axis dropdown
            dict(
                buttons=create_axis_buttons("z", default_z),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.55,
                yanchor="top",
                bgcolor="rgba(245,245,245,0.9)",
                bordercolor="rgba(180,180,180,0.6)",
            ),
        ],
        annotations=[
            dict(
                text="<b>X-axis</b>",
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
            dict(
                text="<b>Y-axis</b>",
                x=0.02,
                y=0.78,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
            dict(
                text="<b>Z-axis</b>",
                x=0.02,
                y=0.58,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
        ],
    )

    return fig


def create_fold_mse_comparison(kfold_results) -> go.Figure:
    """
    Create bar chart comparing MSE across folds.

    Args:
        kfold_results: KFoldResult object from analysis

    Returns:
        Plotly Figure
    """
    fold_summary = kfold_results.get_fold_summary()

    fig = go.Figure(
        data=[
            go.Bar(
                x=fold_summary["fold"],
                y=fold_summary["mse"],
                marker_color="steelblue",
                text=fold_summary["mse"].round(4),
                textposition="outside",
            )
        ]
    )

    # Add mean line
    mean_mse = fold_summary["mse"].mean()
    fig.add_hline(
        y=mean_mse,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean MSE: {mean_mse:.4f}",
    )

    fig.update_layout(
        title="MSE Comparison Across Folds",
        xaxis_title="Fold",
        yaxis_title="Mean Squared Error",
        height=400,
        showlegend=False,
    )

    return fig


def create_optimal_k_plot(k_results_df: pd.DataFrame) -> go.Figure:
    """
    Create plot showing MSE vs k for optimal k selection.

    Args:
        k_results_df: DataFrame with columns ['k', 'mean_mse', 'std_mse']

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Add MSE line with error bars
    fig.add_trace(
        go.Scatter(
            x=k_results_df["k"],
            y=k_results_df["mean_mse"],
            mode="lines+markers",
            name="Mean MSE",
            line=dict(color="steelblue", width=2),
            marker=dict(size=8),
            error_y=dict(type="data", array=k_results_df["std_mse"], visible=True),
        )
    )

    # Highlight optimal k
    optimal_idx = k_results_df["mean_mse"].idxmin()
    optimal_k = k_results_df.loc[optimal_idx, "k"]
    optimal_mse = k_results_df.loc[optimal_idx, "mean_mse"]

    fig.add_trace(
        go.Scatter(
            x=[optimal_k],
            y=[optimal_mse],
            mode="markers",
            name=f"Optimal k={optimal_k}",
            marker=dict(size=15, color="red", symbol="star"),
        )
    )

    fig.update_layout(
        title="Optimal K Selection (Lowest MSE)",
        xaxis_title="Number of Folds (k)",
        yaxis_title="Mean Squared Error",
        height=400,
        hovermode="x unified",
    )

    return fig
