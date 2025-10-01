"""
Data loading utilities with Polars
"""
import polars as pl
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple


@st.cache_data(ttl=3600)
def load_timeseries_data(file_path: str) -> pl.DataFrame:
    """
    Load time-series prescriber data with caching.

    Args:
        file_path: Path to parquet or CSV file

    Returns:
        Polars DataFrame with time-series data
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if path.suffix == '.parquet':
        df = pl.read_parquet(file_path)
    elif path.suffix == '.csv':
        df = pl.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Ensure month is string for consistency
    if 'month' in df.columns:
        df = df.with_columns(pl.col('month').cast(pl.Utf8))

    return df


def get_prescriber_list(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get unique prescribers with summary stats for selection dropdown.

    Args:
        df: Time-series DataFrame

    Returns:
        DataFrame with prescriber summaries
    """
    prescriber_summary = df.group_by(['PRESCRIBER_NPI_NBR', 'prescriber_name']).agg([
        pl.col('total_revenue').sum().alias('lifetime_revenue'),
        pl.col('prescription_count').sum().alias('lifetime_prescriptions'),
        pl.col('month').n_unique().alias('active_months'),
        pl.col('specialty').first().alias('specialty'),
        pl.col('state').first().alias('state')
    ]).sort('lifetime_revenue', descending=True)

    return prescriber_summary


def filter_prescriber_data(
    df: pl.DataFrame,
    npi: int,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None
) -> pl.DataFrame:
    """
    Filter data for a specific prescriber and optional date range.

    Args:
        df: Full time-series DataFrame
        npi: Prescriber NPI number
        start_month: Start month (YYYY-MM format)
        end_month: End month (YYYY-MM format)

    Returns:
        Filtered DataFrame for the prescriber
    """
    filtered = df.filter(pl.col('PRESCRIBER_NPI_NBR') == npi)

    if start_month:
        filtered = filtered.filter(pl.col('month') >= start_month)
    if end_month:
        filtered = filtered.filter(pl.col('month') <= end_month)

    return filtered.sort('month')


def get_top_prescribers(df: pl.DataFrame, n: int = 10, metric: str = 'revenue') -> list[int]:
    """
    Get top N prescribers by specified metric.

    Args:
        df: Time-series DataFrame
        n: Number of top prescribers to return
        metric: Metric to sort by ('revenue', 'volume', 'growth')

    Returns:
        List of prescriber NPIs
    """
    if metric == 'revenue':
        agg_col = pl.col('total_revenue').sum().alias('total')
    elif metric == 'volume':
        agg_col = pl.col('prescription_count').sum().alias('total')
    elif metric == 'growth':
        # Calculate growth rate
        return df.filter(
            pl.col('prescription_growth_mom').is_not_null()
        ).group_by('PRESCRIBER_NPI_NBR').agg([
            pl.col('prescription_growth_mom').mean().alias('avg_growth')
        ]).sort('avg_growth', descending=True).head(n)['PRESCRIBER_NPI_NBR'].to_list()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_npis = df.group_by('PRESCRIBER_NPI_NBR').agg(agg_col).sort('total', descending=True).head(n)
    return top_npis['PRESCRIBER_NPI_NBR'].to_list()


def calculate_specialty_benchmark(df: pl.DataFrame, specialty: str) -> pl.DataFrame:
    """
    Calculate benchmark metrics for a specialty.

    Args:
        df: Time-series DataFrame
        specialty: Specialty to benchmark

    Returns:
        DataFrame with monthly specialty averages
    """
    if specialty is None:
        return pl.DataFrame()

    benchmark = df.filter(
        pl.col('specialty') == specialty
    ).group_by('month').agg([
        pl.col('total_revenue').mean().alias('avg_revenue'),
        pl.col('prescription_count').mean().alias('avg_prescription_count'),
        pl.col('unique_drugs').mean().alias('avg_unique_drugs'),
        pl.col('supply_90day_rate').mean().alias('avg_90day_rate')
    ]).sort('month')

    return benchmark


def get_date_range(df: pl.DataFrame) -> Tuple[str, str]:
    """
    Get min and max dates from the dataset.

    Args:
        df: Time-series DataFrame

    Returns:
        Tuple of (min_date, max_date) as strings
    """
    min_date = df['month'].min()
    max_date = df['month'].max()
    return (min_date, max_date)
