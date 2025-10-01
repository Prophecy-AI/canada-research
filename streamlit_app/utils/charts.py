"""
Chart creation utilities using Plotly
"""
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


def create_revenue_trend_chart(
    df: pl.DataFrame,
    benchmark: Optional[pl.DataFrame] = None
) -> go.Figure:
    """
    Create revenue trend line chart with optional benchmark overlay.

    Args:
        df: Prescriber time-series data
        benchmark: Optional benchmark data for comparison

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Main revenue line
    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=df['total_revenue'].to_list(),
        name='Monthly Revenue',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers'
    ))

    # 3-month moving average
    if 'total_revenue_ma3' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['month'].to_list(),
            y=df['total_revenue_ma3'].to_list(),
            name='3-Month MA',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            mode='lines'
        ))

    # Benchmark if provided
    if benchmark is not None and len(benchmark) > 0:
        fig.add_trace(go.Scatter(
            x=benchmark['month'].to_list(),
            y=benchmark['avg_revenue'].to_list(),
            name='Specialty Average',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            mode='lines'
        ))

    fig.update_layout(
        title='Revenue Trend Over Time',
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_volume_trend_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create dual-axis chart for prescription count and revenue.

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure with dual y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Prescription count (left axis)
    fig.add_trace(
        go.Bar(
            x=df['month'].to_list(),
            y=df['prescription_count'].to_list(),
            name='Prescription Count',
            marker_color='#636EFA',
            opacity=0.6
        ),
        secondary_y=False
    )

    # Revenue (right axis)
    fig.add_trace(
        go.Scatter(
            x=df['month'].to_list(),
            y=df['total_revenue'].to_list(),
            name='Revenue',
            line=dict(color='#EF553B', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Prescription Count', secondary_y=False)
    fig.update_yaxes(title_text='Revenue ($)', secondary_y=True)

    fig.update_layout(
        title='Volume & Revenue Trends',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_growth_rate_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create growth rate visualization (MoM % change).

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure
    """
    # Filter out nulls and infinities
    df_clean = df.filter(
        pl.col('revenue_growth_mom').is_not_null() &
        pl.col('revenue_growth_mom').is_finite()
    )

    if len(df_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for growth rate",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

    # Convert to percentage
    growth_pct = (df_clean['revenue_growth_mom'] * 100).to_list()
    colors = ['red' if x < 0 else 'green' for x in growth_pct]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_clean['month'].to_list(),
        y=growth_pct,
        name='MoM Growth',
        marker_color=colors,
        text=[f"{x:.1f}%" for x in growth_pct],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Month-over-Month Growth Rate',
        xaxis_title='Month',
        yaxis_title='Growth (%)',
        template='plotly_white',
        height=350
    )

    return fig


def create_portfolio_diversity_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create line chart showing unique drugs prescribed over time.

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=df['unique_drugs'].to_list(),
        name='Unique Drugs',
        line=dict(color='#9467bd', width=3),
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(148, 103, 189, 0.2)'
    ))

    fig.update_layout(
        title='Portfolio Diversity Over Time',
        xaxis_title='Month',
        yaxis_title='Number of Unique Drugs',
        hovermode='x unified',
        template='plotly_white',
        height=350
    )

    return fig


def create_payer_mix_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create stacked area chart showing payer channel mix evolution.

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Check if payer columns exist
    payer_cols = ['commercial_rate', 'medicare_rate', 'medicaid_rate']
    missing_cols = [col for col in payer_cols if col not in df.columns]

    if missing_cols:
        fig.add_annotation(
            text=f"Missing payer data: {', '.join(missing_cols)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=(df['commercial_rate'] * 100).to_list(),
        name='Commercial',
        mode='lines',
        stackgroup='one',
        fillcolor='#1f77b4',
        line=dict(width=0.5, color='#1f77b4')
    ))

    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=(df['medicare_rate'] * 100).to_list(),
        name='Medicare',
        mode='lines',
        stackgroup='one',
        fillcolor='#ff7f0e',
        line=dict(width=0.5, color='#ff7f0e')
    ))

    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=(df['medicaid_rate'] * 100).to_list(),
        name='Medicaid',
        mode='lines',
        stackgroup='one',
        fillcolor='#2ca02c',
        line=dict(width=0.5, color='#2ca02c')
    ))

    fig.update_layout(
        title='Payer Mix Evolution',
        xaxis_title='Month',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        template='plotly_white',
        height=350,
        yaxis=dict(range=[0, 100])
    )

    return fig


def create_refill_pattern_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create line chart for 90-day supply rate (patient retention proxy).

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure
    """
    if 'supply_90day_rate' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="90-day supply data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=(df['supply_90day_rate'] * 100).to_list(),
        name='90-Day Supply Rate',
        line=dict(color='#e377c2', width=3),
        mode='lines+markers'
    ))

    fig.update_layout(
        title='Patient Retention Proxy (90-Day Supply Rate)',
        xaxis_title='Month',
        yaxis_title='90-Day Supply Rate (%)',
        hovermode='x unified',
        template='plotly_white',
        height=350,
        yaxis=dict(range=[0, 100])
    )

    return fig


def create_clinical_metrics_chart(df: pl.DataFrame) -> go.Figure:
    """
    Create chart showing clinical activity (conditions, procedures).

    Args:
        df: Prescriber time-series data

    Returns:
        Plotly Figure with dual metrics
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Unique conditions
    if 'unique_conditions' in df.columns:
        df_conditions = df.filter(pl.col('unique_conditions').is_not_null())
        fig.add_trace(
            go.Scatter(
                x=df_conditions['month'].to_list(),
                y=df_conditions['unique_conditions'].to_list(),
                name='Unique Conditions',
                line=dict(color='#8c564b', width=2),
                mode='lines+markers'
            ),
            secondary_y=False
        )

    # Procedure rate
    if 'procedure_rate' in df.columns:
        df_procedures = df.filter(pl.col('procedure_rate').is_not_null())
        fig.add_trace(
            go.Scatter(
                x=df_procedures['month'].to_list(),
                y=(df_procedures['procedure_rate'] * 100).to_list(),
                name='Procedure Rate',
                line=dict(color='#bcbd22', width=2, dash='dash'),
                mode='lines+markers'
            ),
            secondary_y=True
        )

    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Unique Conditions', secondary_y=False)
    fig.update_yaxes(title_text='Procedure Rate (%)', secondary_y=True)

    fig.update_layout(
        title='Clinical Activity Metrics',
        hovermode='x unified',
        template='plotly_white',
        height=350
    )

    return fig


def create_forecast_chart(df: pl.DataFrame, forecast_months: int = 3) -> go.Figure:
    """
    Create simple forecast chart using linear trend.

    Args:
        df: Prescriber time-series data (sorted by month)
        forecast_months: Number of months to forecast

    Returns:
        Plotly Figure with forecast
    """
    import numpy as np

    # Get last N months for trend calculation
    recent_data = df.tail(12)

    if len(recent_data) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for forecasting",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Simple linear regression
    x = np.arange(len(recent_data))
    y = recent_data['total_revenue'].to_numpy()

    # Remove nulls
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for forecasting",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    x_clean = x[mask]
    y_clean = y[mask]

    # Fit linear model
    coeffs = np.polyfit(x_clean, y_clean, 1)
    poly = np.poly1d(coeffs)

    # Generate forecast
    forecast_x = np.arange(len(recent_data), len(recent_data) + forecast_months)
    forecast_y = poly(forecast_x)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=df['month'].to_list(),
        y=df['total_revenue'].to_list(),
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        mode='lines+markers'
    ))

    # Forecast
    last_month = df['month'][-1]
    forecast_months_list = []
    for i in range(1, forecast_months + 1):
        year, month = map(int, last_month.split('-'))
        month += i
        if month > 12:
            year += month // 12
            month = month % 12
            if month == 0:
                month = 12
                year -= 1
        forecast_months_list.append(f"{year:04d}-{month:02d}")

    fig.add_trace(go.Scatter(
        x=forecast_months_list,
        y=forecast_y.tolist(),
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines+markers'
    ))

    fig.update_layout(
        title=f'Revenue Forecast (Next {forecast_months} Months)',
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig
