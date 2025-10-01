"""
High-Value Prescriber Time-Series Analytics Dashboard

Interactive Streamlit app for exploring prescriber trends and patterns.
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Prescriber Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load time-series prescriber data from parquet"""
    data_path = Path("outputs/timeseries_prescriber_monthly.parquet")
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    df = pl.read_parquet(data_path)
    return df

@st.cache_data
def get_prescriber_list(df):
    """Get unique prescribers with name and NPI"""
    prescribers = (
        df
        .group_by(['PRESCRIBER_NPI_NBR', 'prescriber_name'])
        .agg([
            pl.col('total_revenue').sum().alias('lifetime_revenue'),
            pl.col('month').count().alias('active_months')
        ])
        .sort('lifetime_revenue', descending=True)
    )
    return prescribers

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(value):
    """Format value as currency"""
    if value is None or np.isnan(value):
        return "N/A"
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"

def format_number(value):
    """Format number with commas"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{int(value):,}"

def format_percent(value):
    """Format as percentage"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value*100:.1f}%"

def calculate_yoy_growth(df_prescriber):
    """Calculate year-over-year growth"""
    if len(df_prescriber) < 12:
        return None

    df_sorted = df_prescriber.sort('month')
    current_year = df_sorted.filter(pl.col('year') == df_sorted['year'].max())
    prev_year = df_sorted.filter(pl.col('year') == df_sorted['year'].max() - 1)

    if len(current_year) == 0 or len(prev_year) == 0:
        return None

    current_rev = current_year['total_revenue'].sum()
    prev_rev = prev_year['total_revenue'].sum()

    if prev_rev == 0:
        return None

    return (current_rev - prev_rev) / prev_rev

def simple_forecast(df_prescriber, periods=3):
    """Simple linear forecast for next N periods"""
    if len(df_prescriber) < 3:
        return None

    df_sorted = df_prescriber.sort('month').tail(12)  # Use last 12 months

    x = np.arange(len(df_sorted))
    y = df_sorted['total_revenue'].to_numpy()

    # Simple linear regression
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    # Forecast
    future_x = np.arange(len(df_sorted), len(df_sorted) + periods)
    forecast = slope * future_x + intercept

    return forecast

def get_peer_benchmark(df, npi, specialty):
    """Get specialty average for benchmarking"""
    peer_data = (
        df
        .filter(pl.col('specialty') == specialty)
        .filter(pl.col('PRESCRIBER_NPI_NBR') != npi)
        .group_by('month')
        .agg([
            pl.col('total_revenue').mean().alias('avg_revenue'),
            pl.col('prescription_count').mean().alias('avg_prescriptions'),
            pl.col('unique_drugs').mean().alias('avg_unique_drugs')
        ])
        .sort('month')
    )
    return peer_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_revenue_trend(df_prescriber, show_ma=True):
    """Plot revenue trend with optional moving average"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    fig = go.Figure()

    # Actual revenue
    fig.add_trace(go.Scatter(
        x=df_sorted['month'],
        y=df_sorted['total_revenue'],
        mode='lines+markers',
        name='Monthly Revenue',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6)
    ))

    # Moving average
    if show_ma and 'total_revenue_ma3' in df_sorted.columns:
        fig.add_trace(go.Scatter(
            x=df_sorted['month'],
            y=df_sorted['total_revenue_ma3'],
            mode='lines',
            name='3-Month MA',
            line=dict(color='#A23B72', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_volume_trend(df_prescriber):
    """Plot prescription volume trend"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Prescription count
    fig.add_trace(
        go.Bar(
            x=df_sorted['month'],
            y=df_sorted['prescription_count'],
            name='Prescription Count',
            marker_color='#06A77D'
        ),
        secondary_y=False
    )

    # Unique drugs
    fig.add_trace(
        go.Scatter(
            x=df_sorted['month'],
            y=df_sorted['unique_drugs'],
            mode='lines+markers',
            name='Unique Drugs',
            line=dict(color='#F18F01', width=2),
            marker=dict(size=6)
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Prescription Count", secondary_y=False)
    fig.update_yaxes(title_text="Unique Drugs", secondary_y=True)

    fig.update_layout(
        title="Volume & Portfolio Diversity",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_payer_mix(df_prescriber):
    """Plot payer mix evolution"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sorted['month'],
        y=df_sorted['commercial_rate'],
        mode='lines',
        name='Commercial',
        stackgroup='one',
        fillcolor='#2E86AB'
    ))

    fig.add_trace(go.Scatter(
        x=df_sorted['month'],
        y=df_sorted['medicare_rate'],
        mode='lines',
        name='Medicare',
        stackgroup='one',
        fillcolor='#A23B72'
    ))

    fig.add_trace(go.Scatter(
        x=df_sorted['month'],
        y=df_sorted['medicaid_rate'],
        mode='lines',
        name='Medicaid',
        stackgroup='one',
        fillcolor='#F18F01'
    ))

    fig.update_layout(
        title="Payer Mix Evolution",
        xaxis_title="Month",
        yaxis_title="Proportion",
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified',
        height=400
    )

    return fig

def plot_retention_metrics(df_prescriber):
    """Plot patient retention proxies"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 90-day supply rate
    fig.add_trace(
        go.Scatter(
            x=df_sorted['month'],
            y=df_sorted['supply_90day_rate'],
            mode='lines+markers',
            name='90-Day Supply Rate',
            line=dict(color='#06A77D', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )

    # Average days supply
    fig.add_trace(
        go.Scatter(
            x=df_sorted['month'],
            y=df_sorted['avg_days_supply'],
            mode='lines+markers',
            name='Avg Days Supply',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="90-Day Supply Rate", secondary_y=False, tickformat='.0%')
    fig.update_yaxes(title_text="Avg Days Supply", secondary_y=True)

    fig.update_layout(
        title="Patient Retention Indicators",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_clinical_activity(df_prescriber):
    """Plot clinical complexity metrics"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Unique conditions
    if 'unique_conditions' in df_sorted.columns:
        fig.add_trace(
            go.Scatter(
                x=df_sorted['month'],
                y=df_sorted['unique_conditions'],
                mode='lines+markers',
                name='Unique Conditions',
                line=dict(color='#A23B72', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

    # Procedure rate
    if 'procedure_rate' in df_sorted.columns:
        fig.add_trace(
            go.Scatter(
                x=df_sorted['month'],
                y=df_sorted['procedure_rate'],
                mode='lines+markers',
                name='Procedure Rate',
                line=dict(color='#F18F01', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True
        )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Unique Conditions", secondary_y=False)
    fig.update_yaxes(title_text="Procedure Rate", secondary_y=True, tickformat='.0%')

    fig.update_layout(
        title="Clinical Activity Metrics",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_growth_rate(df_prescriber):
    """Plot month-over-month growth rate"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    if 'revenue_growth_mom' not in df_sorted.columns:
        return None

    # Remove infinite values
    df_sorted = df_sorted[np.isfinite(df_sorted['revenue_growth_mom'])]

    colors = ['#06A77D' if x >= 0 else '#E63946' for x in df_sorted['revenue_growth_mom']]

    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted['month'],
            y=df_sorted['revenue_growth_mom'],
            marker_color=colors,
            name='MoM Growth'
        )
    ])

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Month-over-Month Revenue Growth",
        xaxis_title="Month",
        yaxis_title="Growth Rate",
        yaxis=dict(tickformat='.0%'),
        height=300
    )

    return fig

def plot_peer_comparison(df_prescriber, peer_data):
    """Compare prescriber to peer average"""
    df_prescriber_pd = df_prescriber.sort('month').to_pandas()
    peer_pd = peer_data.to_pandas()

    fig = go.Figure()

    # Prescriber revenue
    fig.add_trace(go.Scatter(
        x=df_prescriber_pd['month'],
        y=df_prescriber_pd['total_revenue'],
        mode='lines+markers',
        name='This Prescriber',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))

    # Peer average
    fig.add_trace(go.Scatter(
        x=peer_pd['month'],
        y=peer_pd['avg_revenue'],
        mode='lines',
        name='Specialty Average',
        line=dict(color='#A9A9A9', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Peer Benchmarking: Revenue vs Specialty Average",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_forecast(df_prescriber, forecast_values):
    """Plot historical data with forecast"""
    df_sorted = df_prescriber.sort('month').to_pandas()

    # Get last N months as historical
    historical_months = df_sorted['month'].tolist()
    historical_revenue = df_sorted['total_revenue'].tolist()

    # Generate future months (simple increment)
    last_month = historical_months[-1]
    future_months = []
    for i in range(1, len(forecast_values) + 1):
        # Simple month increment (not perfect but works for demo)
        year, month = last_month.split('-')
        year, month = int(year), int(month)
        month += i
        if month > 12:
            year += 1
            month = month % 12
        future_months.append(f"{year}-{month:02d}")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_months,
        y=historical_revenue,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_months,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#F18F01', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Revenue Forecast (3-Month Projection)",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_seasonality_heatmap(df_prescriber):
    """Create seasonality heatmap by month and year"""
    df_sorted = df_prescriber.sort('month')

    # Pivot data
    pivot_data = (
        df_sorted
        .select(['year', 'month_num', 'total_revenue'])
        .to_pandas()
        .pivot(index='year', columns='month_num', values='total_revenue')
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot_data.index,
        colorscale='Blues',
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Revenue: $%{z:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Revenue Seasonality Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        height=300
    )

    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.title("📊 Prescriber Time-Series Analytics")
    st.markdown("Interactive dashboard for exploring prescriber trends and patterns")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        prescribers = get_prescriber_list(df)

    # =============================================================================
    # SIDEBAR
    # =============================================================================

    st.sidebar.header("🔍 Prescriber Selection")

    # Quick filters
    filter_option = st.sidebar.radio(
        "Quick Filters:",
        ["Custom Search", "Top 10 by Revenue", "Top 10 by Volume", "Random Prescriber"]
    )

    selected_npi = None

    if filter_option == "Top 10 by Revenue":
        top_10 = prescribers.head(10)
        options = [
            f"{row['prescriber_name']} (NPI: {row['PRESCRIBER_NPI_NBR']}) - {format_currency(row['lifetime_revenue'])}"
            for row in top_10.iter_rows(named=True)
        ]
        selected = st.sidebar.selectbox("Select Prescriber:", options)
        selected_npi = int(selected.split("NPI: ")[1].split(")")[0])

    elif filter_option == "Top 10 by Volume":
        top_10_volume = (
            df.group_by(['PRESCRIBER_NPI_NBR', 'prescriber_name'])
            .agg([pl.col('prescription_count').sum().alias('total_prescriptions')])
            .sort('total_prescriptions', descending=True)
            .head(10)
        )
        options = [
            f"{row['prescriber_name']} (NPI: {row['PRESCRIBER_NPI_NBR']}) - {format_number(row['total_prescriptions'])} Rx"
            for row in top_10_volume.iter_rows(named=True)
        ]
        selected = st.sidebar.selectbox("Select Prescriber:", options)
        selected_npi = int(selected.split("NPI: ")[1].split(")")[0])

    elif filter_option == "Random Prescriber":
        if st.sidebar.button("🎲 Pick Random Prescriber"):
            random_prescriber = prescribers.sample(n=1)
            selected_npi = random_prescriber['PRESCRIBER_NPI_NBR'][0]
        else:
            random_prescriber = prescribers.sample(n=1)
            selected_npi = random_prescriber['PRESCRIBER_NPI_NBR'][0]

        prescriber_info = prescribers.filter(pl.col('PRESCRIBER_NPI_NBR') == selected_npi)
        st.sidebar.info(
            f"**{prescriber_info['prescriber_name'][0]}**\n\n"
            f"NPI: {selected_npi}\n\n"
            f"Revenue: {format_currency(prescriber_info['lifetime_revenue'][0])}"
        )

    else:  # Custom Search
        # Searchable dropdown
        search_term = st.sidebar.text_input("Search by name or NPI:")

        if search_term:
            filtered = prescribers.filter(
                (pl.col('prescriber_name').str.to_lowercase().str.contains(search_term.lower())) |
                (pl.col('PRESCRIBER_NPI_NBR').cast(pl.Utf8).str.contains(search_term))
            ).head(50)
        else:
            filtered = prescribers.head(50)

        if len(filtered) > 0:
            options = [
                f"{row['prescriber_name']} (NPI: {row['PRESCRIBER_NPI_NBR']}) - {format_currency(row['lifetime_revenue'])}"
                for row in filtered.iter_rows(named=True)
            ]
            selected = st.sidebar.selectbox("Select Prescriber:", options)
            selected_npi = int(selected.split("NPI: ")[1].split(")")[0])
        else:
            st.sidebar.warning("No prescribers found")
            return

    # Time range selector
    st.sidebar.header("📅 Time Range")

    df_prescriber = df.filter(pl.col('PRESCRIBER_NPI_NBR') == selected_npi).sort('month')

    if len(df_prescriber) == 0:
        st.error("No data found for selected prescriber")
        return

    all_months = df_prescriber['month'].unique().sort().to_list()

    time_range_option = st.sidebar.radio(
        "Select Range:",
        ["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Custom"]
    )

    if time_range_option == "Last 12 Months":
        df_prescriber = df_prescriber.tail(12)
    elif time_range_option == "Last 6 Months":
        df_prescriber = df_prescriber.tail(6)
    elif time_range_option == "Last 3 Months":
        df_prescriber = df_prescriber.tail(3)
    elif time_range_option == "Custom":
        if len(all_months) >= 2:
            start_month = st.sidebar.selectbox("Start Month:", all_months, index=0)
            end_month = st.sidebar.selectbox("End Month:", all_months, index=len(all_months)-1)
            df_prescriber = df_prescriber.filter(
                (pl.col('month') >= start_month) & (pl.col('month') <= end_month)
            )

    # =============================================================================
    # MAIN DASHBOARD
    # =============================================================================

    if len(df_prescriber) == 0:
        st.warning("No data in selected time range")
        return

    # Get prescriber info
    prescriber_info = prescribers.filter(pl.col('PRESCRIBER_NPI_NBR') == selected_npi)

    # Summary stats
    st.header(f"👤 {prescriber_info['prescriber_name'][0]}")
    st.caption(f"NPI: {selected_npi}")

    # Get latest month data
    latest_month_data = df_prescriber.tail(1)

    # Calculate metrics
    lifetime_revenue = df_prescriber['total_revenue'].sum()
    current_month_revenue = latest_month_data['total_revenue'][0] if len(latest_month_data) > 0 else 0
    yoy_growth = calculate_yoy_growth(df_prescriber)
    active_months = len(df_prescriber)
    specialty = latest_month_data['specialty'][0] if len(latest_month_data) > 0 and latest_month_data['specialty'][0] else "Unknown"
    state = latest_month_data['state'][0] if len(latest_month_data) > 0 and latest_month_data['state'][0] else "Unknown"
    receives_payments = latest_month_data['receives_payments'][0] if len(latest_month_data) > 0 else 0

    # Display metrics in cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Lifetime Revenue", format_currency(lifetime_revenue))

    with col2:
        st.metric("Current Month", format_currency(current_month_revenue))

    with col3:
        if yoy_growth is not None:
            st.metric("YoY Growth", format_percent(yoy_growth),
                     delta=f"{yoy_growth*100:.1f}%" if yoy_growth != 0 else None)
        else:
            st.metric("YoY Growth", "N/A")

    with col4:
        st.metric("Active Months", active_months)

    with col5:
        payment_status = "✅ Receives" if receives_payments == 1 else "❌ None"
        st.metric("Pharma Payments", payment_status)

    # Additional info
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Specialty:** {specialty}")
    with col2:
        st.caption(f"**State:** {state}")

    st.divider()

    # =============================================================================
    # CHARTS
    # =============================================================================

    # Row 1: Revenue and Volume Trends
    col1, col2 = st.columns(2)

    with col1:
        fig_revenue = plot_revenue_trend(df_prescriber, show_ma=True)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        fig_volume = plot_volume_trend(df_prescriber)
        st.plotly_chart(fig_volume, use_container_width=True)

    # Growth rate
    fig_growth = plot_growth_rate(df_prescriber)
    if fig_growth:
        st.plotly_chart(fig_growth, use_container_width=True)

    # Row 2: Payer Mix and Retention
    col1, col2 = st.columns(2)

    with col1:
        fig_payer = plot_payer_mix(df_prescriber)
        st.plotly_chart(fig_payer, use_container_width=True)

    with col2:
        fig_retention = plot_retention_metrics(df_prescriber)
        st.plotly_chart(fig_retention, use_container_width=True)

    # Row 3: Clinical Activity and Seasonality
    col1, col2 = st.columns(2)

    with col1:
        fig_clinical = plot_clinical_activity(df_prescriber)
        st.plotly_chart(fig_clinical, use_container_width=True)

    with col2:
        if len(df_prescriber) >= 12:
            fig_seasonality = plot_seasonality_heatmap(df_prescriber)
            st.plotly_chart(fig_seasonality, use_container_width=True)
        else:
            st.info("Seasonality heatmap requires at least 12 months of data")

    st.divider()

    # =============================================================================
    # COMPARATIVE ANALYTICS
    # =============================================================================

    st.header("📊 Comparative Analytics")

    # Peer benchmarking
    if specialty and specialty != "Unknown":
        st.subheader("Peer Benchmarking")

        with st.spinner("Loading peer data..."):
            peer_data = get_peer_benchmark(df, selected_npi, specialty)

        if len(peer_data) > 0:
            fig_peer = plot_peer_comparison(df_prescriber, peer_data)
            st.plotly_chart(fig_peer, use_container_width=True)

            # Calculate percentile rank
            specialty_prescribers = df.filter(pl.col('specialty') == specialty)
            total_revenues = (
                specialty_prescribers
                .group_by('PRESCRIBER_NPI_NBR')
                .agg([pl.col('total_revenue').sum().alias('total_rev')])
                .sort('total_rev')
            )

            rank = (total_revenues.filter(pl.col('PRESCRIBER_NPI_NBR') == selected_npi).select(
                pl.col('total_rev').rank('ordinal').alias('rank')
            ))

            if len(rank) > 0:
                percentile = (rank['rank'][0] / len(total_revenues)) * 100
                st.info(f"📈 This prescriber ranks in the **{percentile:.1f}th percentile** among {specialty} specialists by revenue")
        else:
            st.warning("Not enough peer data available for comparison")

    # Forecasting
    st.subheader("🔮 Revenue Forecast")

    forecast_values = simple_forecast(df_prescriber, periods=3)

    if forecast_values is not None:
        fig_forecast = plot_forecast(df_prescriber, forecast_values)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption("Note: Forecast based on simple linear trend from last 12 months. For demonstration purposes only.")
    else:
        st.info("Insufficient data for forecasting (need at least 3 months)")

    st.divider()

    # =============================================================================
    # EXPORT
    # =============================================================================

    st.header("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export prescriber data as CSV
        csv_buffer = io.StringIO()
        df_prescriber.write_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Prescriber Data (CSV)",
            data=csv_data,
            file_name=f"prescriber_{selected_npi}_data.csv",
            mime="text/csv"
        )

    with col2:
        # Export summary stats
        summary_stats = {
            'NPI': [selected_npi],
            'Name': [prescriber_info['prescriber_name'][0]],
            'Specialty': [specialty],
            'State': [state],
            'Lifetime Revenue': [lifetime_revenue],
            'Active Months': [active_months],
            'YoY Growth': [yoy_growth if yoy_growth else None],
            'Receives Payments': [receives_payments]
        }

        summary_df = pl.DataFrame(summary_stats)
        summary_csv = io.StringIO()
        summary_df.write_csv(summary_csv)

        st.download_button(
            label="📥 Download Summary Stats (CSV)",
            data=summary_csv.getvalue(),
            file_name=f"prescriber_{selected_npi}_summary.csv",
            mime="text/csv"
        )

    # Footer
    st.divider()
    st.caption("High-Value Prescriber Analytics Dashboard | Built with Streamlit + Polars + Plotly")

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
