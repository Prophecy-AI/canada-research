"""
Reusable UI components
"""
import streamlit as st
import polars as pl
from typing import Dict, Any
from utils.metrics import format_currency, format_percentage


def render_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """
    Render a styled metric card.

    Args:
        label: Metric label
        value: Main value to display
        delta: Optional delta/change value
        help_text: Optional help text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )


def render_summary_cards(stats: Dict[str, Any]):
    """
    Render summary statistics cards in a grid.

    Args:
        stats: Dictionary of summary statistics
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Lifetime Revenue",
            value=format_currency(stats['lifetime_revenue']),
            help="Total revenue across all time"
        )

    with col2:
        current_rev = format_currency(stats['current_month_revenue'])
        yoy_growth = stats.get('yoy_growth')

        # Only show delta if we have valid YoY data
        if yoy_growth is not None and not (yoy_growth == 0):
            delta_value = format_percentage(yoy_growth)
            st.metric(
                label="📅 Current Month",
                value=current_rev,
                delta=delta_value,
                delta_color="normal",  # Green for positive, red for negative
                help="Most recent month revenue with YoY growth"
            )
        else:
            st.metric(
                label="📅 Current Month",
                value=current_rev,
                help="Most recent month revenue"
            )

    with col3:
        st.metric(
            label="📊 Total Prescriptions",
            value=f"{stats['total_prescriptions']:,}",
            help="Total prescriptions written"
        )

    with col4:
        st.metric(
            label="📈 Active Months",
            value=str(stats['active_months']),
            help="Number of months with activity"
        )

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="💊 Avg Unique Drugs",
            value=f"{stats['avg_unique_drugs']:.1f}",
            help="Average unique drugs per month"
        )

    with col6:
        st.metric(
            label="🏥 Specialty",
            value=stats['specialty'] or 'Unknown',
            help="Clinical specialty"
        )

    with col7:
        st.metric(
            label="📍 State",
            value=stats['state'] or 'Unknown',
            help="Primary practice state"
        )

    with col8:
        payment_status = "✅ Yes" if stats.get('receives_payments') else "❌ No"
        payment_amt = stats.get('total_payments', 0)

        if stats.get('receives_payments') and payment_amt > 0:
            st.metric(
                label="💵 Pharma Payments",
                value=payment_status,
                delta=format_currency(payment_amt),
                delta_color="off",  # No color, just info
                help="Receives pharma payments"
            )
        else:
            st.metric(
                label="💵 Pharma Payments",
                value=payment_status,
                help="Receives pharma payments"
            )


def render_prescriber_selector(
    prescriber_list: pl.DataFrame,
    key: str = "prescriber_select"
) -> int:
    """
    Render prescriber selection dropdown with search.

    Args:
        prescriber_list: DataFrame with prescriber summaries
        key: Unique key for the selectbox

    Returns:
        Selected prescriber NPI
    """
    # Create display strings
    prescriber_options = [
        f"{row['prescriber_name']} (NPI: {row['PRESCRIBER_NPI_NBR']}) - "
        f"{format_currency(row['lifetime_revenue'])} - {row['specialty']}"
        for row in prescriber_list.iter_rows(named=True)
    ]

    # Add NPI to mapping
    npi_mapping = {
        option: row['PRESCRIBER_NPI_NBR']
        for option, row in zip(prescriber_options, prescriber_list.iter_rows(named=True))
    }

    selected = st.selectbox(
        "Select Prescriber",
        options=prescriber_options,
        key=key,
        help="Search by name, NPI, or specialty"
    )

    return npi_mapping[selected]


def render_quick_filters(df: pl.DataFrame, prescriber_list: pl.DataFrame):
    """
    Render quick filter buttons for common selections.

    Args:
        df: Full time-series DataFrame
        prescriber_list: DataFrame with prescriber summaries

    Returns:
        Selected NPI if any button clicked, else None
    """
    st.markdown("**Quick Filters:**")
    col1, col2, col3, col4 = st.columns(4)

    selected_npi = None

    with col1:
        if st.button("🥇 Top by Revenue", width="stretch"):
            selected_npi = prescriber_list[0, 'PRESCRIBER_NPI_NBR']

    with col2:
        if st.button("📈 Fastest Growing", width="stretch"):
            from utils.data_loader import get_top_prescribers
            top_growth = get_top_prescribers(df, n=1, metric='growth')
            if len(top_growth) > 0:
                selected_npi = top_growth[0]

    with col3:
        if st.button("💊 Most Active", width="stretch"):
            from utils.data_loader import get_top_prescribers
            top_volume = get_top_prescribers(df, n=1, metric='volume')
            if len(top_volume) > 0:
                selected_npi = top_volume[0]

    with col4:
        if st.button("🎲 Random", width="stretch"):
            import random
            selected_npi = random.choice(prescriber_list['PRESCRIBER_NPI_NBR'].to_list())

    return selected_npi


def render_date_range_selector(min_date: str, max_date: str, key_prefix: str = ""):
    """
    Render date range selector.

    Args:
        min_date: Minimum available date (YYYY-MM)
        max_date: Maximum available date (YYYY-MM)
        key_prefix: Prefix for widget keys

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    st.markdown("**Time Range:**")

    col1, col2 = st.columns(2)

    with col1:
        use_date_filter = st.checkbox("Filter by date range", value=False, key=f"{key_prefix}_use_filter")

    if use_date_filter:
        with col2:
            quick_range = st.selectbox(
                "Quick Select",
                ["Custom", "Last 3 Months", "Last 6 Months", "Last 12 Months", "All Time"],
                key=f"{key_prefix}_quick_range"
            )

        if quick_range == "Custom":
            col3, col4 = st.columns(2)
            with col3:
                start = st.text_input("Start Month (YYYY-MM)", value=min_date, key=f"{key_prefix}_start")
            with col4:
                end = st.text_input("End Month (YYYY-MM)", value=max_date, key=f"{key_prefix}_end")
            return start, end

        elif quick_range == "Last 3 Months":
            # Calculate 3 months back
            year, month = map(int, max_date.split('-'))
            month -= 3
            if month <= 0:
                month += 12
                year -= 1
            return f"{year:04d}-{month:02d}", max_date

        elif quick_range == "Last 6 Months":
            year, month = map(int, max_date.split('-'))
            month -= 6
            if month <= 0:
                month += 12
                year -= 1
            return f"{year:04d}-{month:02d}", max_date

        elif quick_range == "Last 12 Months":
            year, month = map(int, max_date.split('-'))
            year -= 1
            return f"{year:04d}-{month:02d}", max_date

        else:  # All Time
            return None, None

    return None, None


def render_export_options(df: pl.DataFrame, prescriber_name: str):
    """
    Render data export options.

    Args:
        df: DataFrame to export
        prescriber_name: Name of prescriber for filename
    """
    st.markdown("**Export Options:**")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.write_csv()
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{prescriber_name.replace(' ', '_')}_data.csv",
            mime="text/csv",
            width="stretch"
        )

    with col2:
        # Note: Plotly charts have built-in download as PNG feature
        st.info("💡 Charts can be downloaded using the 📷 button in the top-right of each chart", icon="ℹ️")
