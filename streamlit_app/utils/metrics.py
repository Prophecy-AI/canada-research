"""
Metric calculation utilities
"""
import polars as pl
from typing import Dict, Any


def calculate_summary_stats(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for a prescriber.

    Args:
        df: Filtered prescriber time-series data

    Returns:
        Dictionary of summary metrics
    """
    if len(df) == 0:
        return {
            'lifetime_revenue': 0,
            'current_month_revenue': 0,
            'avg_monthly_revenue': 0,
            'total_prescriptions': 0,
            'active_months': 0,
            'avg_unique_drugs': 0,
            'specialty': 'Unknown',
            'state': 'Unknown',
            'receives_payments': False,
            'total_payments': 0,
            'yoy_growth': None
        }

    # Basic stats
    stats = {
        'lifetime_revenue': float(df['total_revenue'].sum()),
        'current_month_revenue': float(df['total_revenue'][-1]) if len(df) > 0 else 0,
        'avg_monthly_revenue': float(df['total_revenue'].mean()),
        'total_prescriptions': int(df['prescription_count'].sum()),
        'active_months': int(len(df)),
        'avg_unique_drugs': float(df['unique_drugs'].mean()),
        'specialty': df['specialty'][0] if 'specialty' in df.columns else 'Unknown',
        'state': df['state'][0] if 'state' in df.columns else 'Unknown',
    }

    # Payment info (static across months)
    if 'receives_payments' in df.columns:
        stats['receives_payments'] = bool(df['receives_payments'][0])
    if 'total_payments' in df.columns:
        stats['total_payments'] = float(df['total_payments'][0])

    # Calculate YoY growth if we have enough data
    stats['yoy_growth'] = calculate_yoy_growth(df)

    return stats


def calculate_yoy_growth(df: pl.DataFrame) -> float:
    """
    Calculate year-over-year revenue growth.

    Args:
        df: Prescriber time-series data (sorted by month)

    Returns:
        YoY growth rate as decimal (e.g., 0.15 for 15% growth)
    """
    if len(df) < 13:  # Need at least 13 months for YoY
        return None

    # Get revenue from 12 months ago
    current_revenue = df['total_revenue'][-1]
    year_ago_revenue = df['total_revenue'][-13]

    if year_ago_revenue == 0 or year_ago_revenue is None:
        return None

    yoy_growth = (current_revenue - year_ago_revenue) / year_ago_revenue
    return float(yoy_growth)


def calculate_percentile_rank(
    df: pl.DataFrame,
    npi: int,
    metric: str = 'total_revenue'
) -> float:
    """
    Calculate percentile rank for a prescriber on a given metric.

    Args:
        df: Full time-series DataFrame
        npi: Prescriber NPI
        metric: Column name to rank on

    Returns:
        Percentile (0-100)
    """
    # Aggregate by prescriber
    prescriber_totals = df.group_by('PRESCRIBER_NPI_NBR').agg(
        pl.col(metric).sum().alias('total')
    )

    # Get prescriber's value
    prescriber_value = prescriber_totals.filter(
        pl.col('PRESCRIBER_NPI_NBR') == npi
    )['total'][0]

    # Calculate percentile
    rank = (prescriber_totals.filter(
        pl.col('total') < prescriber_value
    ).height / prescriber_totals.height) * 100

    return round(rank, 1)


def detect_anomalies(df: pl.DataFrame, metric: str = 'total_revenue', std_threshold: float = 2.5) -> pl.DataFrame:
    """
    Detect anomalous months using standard deviation method.

    Args:
        df: Prescriber time-series data
        metric: Column to check for anomalies
        std_threshold: Number of standard deviations for anomaly threshold

    Returns:
        DataFrame with additional 'is_anomaly' column
    """
    if len(df) < 3:
        return df.with_columns(pl.lit(False).alias('is_anomaly'))

    mean = df[metric].mean()
    std = df[metric].std()

    if std == 0 or std is None:
        return df.with_columns(pl.lit(False).alias('is_anomaly'))

    df_with_anomaly = df.with_columns([
        (
            (pl.col(metric) - mean).abs() > (std_threshold * std)
        ).alias('is_anomaly')
    ])

    return df_with_anomaly


def calculate_trend_direction(df: pl.DataFrame, metric: str = 'total_revenue', window: int = 3) -> str:
    """
    Determine overall trend direction (increasing, decreasing, stable).

    Args:
        df: Prescriber time-series data (sorted by month)
        metric: Column to analyze
        window: Number of recent months to consider

    Returns:
        Trend direction: 'increasing', 'decreasing', or 'stable'
    """
    if len(df) < window:
        return 'insufficient_data'

    recent_data = df.tail(window)
    values = recent_data[metric].to_list()

    # Calculate linear regression slope
    import numpy as np
    x = np.arange(len(values))
    y = np.array(values)

    # Remove nulls
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 'insufficient_data'

    x_clean = x[mask]
    y_clean = y[mask]

    coeffs = np.polyfit(x_clean, y_clean, 1)
    slope = coeffs[0]

    # Determine trend based on slope
    mean_value = np.mean(y_clean)
    slope_pct = (slope / mean_value) * 100 if mean_value != 0 else 0

    if slope_pct > 5:
        return 'increasing'
    elif slope_pct < -5:
        return 'decreasing'
    else:
        return 'stable'


def format_currency(value: float) -> str:
    """Format value as currency string."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    if value is None:
        return "N/A"
    return f"{value*100:+.1f}%"
