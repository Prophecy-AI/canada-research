"""
High-Value Prescriber Time-Series Analysis Dashboard

A comprehensive Streamlit application for visualizing and analyzing
prescriber behavioral patterns over time.
"""
import streamlit as st
import polars as pl
from pathlib import Path

# Import utilities
from utils.data_loader import (
    load_timeseries_data,
    get_prescriber_list,
    filter_prescriber_data,
    calculate_specialty_benchmark,
    get_date_range
)
from utils.charts import (
    create_revenue_trend_chart,
    create_volume_trend_chart,
    create_growth_rate_chart,
    create_portfolio_diversity_chart,
    create_payer_mix_chart,
    create_refill_pattern_chart,
    create_clinical_metrics_chart,
    create_forecast_chart
)
from utils.metrics import (
    calculate_summary_stats,
    calculate_percentile_rank,
    detect_anomalies,
    calculate_trend_direction
)
from utils.components import (
    render_summary_cards,
    render_prescriber_selector,
    render_quick_filters,
    render_date_range_selector,
    render_export_options
)

# Page configuration
st.set_page_config(
    page_title="Prescriber Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    /* Fix metric card visibility */
    [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""

    # Title
    st.title("💊 High-Value Prescriber Analytics Dashboard")
    st.markdown("---")

    # Load data
    try:
        with st.spinner("Loading time-series data..."):
            # Try to load from outputs directory
            data_path = Path(__file__).parent.parent / "outputs" / "timeseries_prescriber_monthly.parquet"
            if not data_path.exists():
                data_path = Path(__file__).parent.parent / "outputs" / "timeseries_prescriber_monthly.csv"

            if not data_path.exists():
                st.error(
                    "⚠️ Data file not found. Please run the time-series dataset preparation notebook first.\n\n"
                    f"Expected location: {data_path}"
                )
                st.stop()

            df = load_timeseries_data(str(data_path))
            st.success(f"✅ Loaded {len(df):,} observations for {df['PRESCRIBER_NPI_NBR'].n_unique():,} prescribers")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Get prescriber list
    prescriber_list = get_prescriber_list(df)
    min_date, max_date = get_date_range(df)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["👨‍⚕️ Individual Dashboard", "📊 Comparative Analytics", "🔍 Data Explorer"])

    # ============================================================================
    # TAB 1: INDIVIDUAL PRESCRIBER DASHBOARD
    # ============================================================================
    with tab1:
        # Sidebar for prescriber selection
        with st.sidebar:
            st.header("Prescriber Selection")

            # Quick filters
            quick_npi = render_quick_filters(df, prescriber_list)

            st.markdown("---")

            # Main selector
            if 'selected_npi' not in st.session_state:
                st.session_state.selected_npi = prescriber_list[0, 'PRESCRIBER_NPI_NBR']

            if quick_npi is not None:
                st.session_state.selected_npi = quick_npi

            selected_npi = render_prescriber_selector(prescriber_list, key="main_selector")
            st.session_state.selected_npi = selected_npi

            st.markdown("---")

            # Date range selector
            start_date, end_date = render_date_range_selector(min_date, max_date, key_prefix="main")

            st.markdown("---")

            # Export options
            prescriber_data = filter_prescriber_data(df, selected_npi, start_date, end_date)
            prescriber_name = prescriber_data['prescriber_name'][0] if len(prescriber_data) > 0 else "Unknown"
            render_export_options(prescriber_data, prescriber_name)

        # Main content area
        if len(prescriber_data) == 0:
            st.warning("No data available for the selected filters.")
            st.stop()

        # Calculate summary stats
        stats = calculate_summary_stats(prescriber_data)

        # Display prescriber info
        st.header(f"📋 {prescriber_name}")
        st.markdown(f"**NPI:** {selected_npi} | **Specialty:** {stats['specialty']} | **State:** {stats['state']}")

        # Summary cards
        render_summary_cards(stats)

        st.markdown("---")

        # Calculate benchmark data
        benchmark_data = calculate_specialty_benchmark(df, stats['specialty'])

        # Chart layout
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_revenue_trend_chart(prescriber_data, benchmark_data),
                width="stretch"
            )

        with col2:
            st.plotly_chart(
                create_volume_trend_chart(prescriber_data),
                width="stretch"
            )

        # Growth rate
        st.plotly_chart(
            create_growth_rate_chart(prescriber_data),
            width="stretch"
        )

        # Portfolio and payer mix
        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                create_portfolio_diversity_chart(prescriber_data),
                width="stretch"
            )

        with col4:
            st.plotly_chart(
                create_payer_mix_chart(prescriber_data),
                width="stretch"
            )

        # Clinical metrics and refill patterns
        col5, col6 = st.columns(2)

        with col5:
            st.plotly_chart(
                create_refill_pattern_chart(prescriber_data),
                width="stretch"
            )

        with col6:
            st.plotly_chart(
                create_clinical_metrics_chart(prescriber_data),
                width="stretch"
            )

        # Forecasting
        st.markdown("---")
        st.subheader("🔮 Revenue Forecasting")

        forecast_months = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)

        st.plotly_chart(
            create_forecast_chart(prescriber_data, forecast_months),
            width="stretch"
        )

        # Insights section
        st.markdown("---")
        st.subheader("💡 Key Insights")

        col7, col8, col9 = st.columns(3)

        with col7:
            percentile = calculate_percentile_rank(df, selected_npi, 'total_revenue')
            st.metric("Revenue Percentile", f"{percentile}%", help="Where this prescriber ranks among all prescribers")

        with col8:
            trend = calculate_trend_direction(prescriber_data, 'total_revenue', window=3)
            trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}.get(trend, '❓')
            st.metric("Recent Trend", f"{trend_emoji} {trend.title()}", help="Based on last 3 months")

        with col9:
            anomaly_data = detect_anomalies(prescriber_data, 'total_revenue')
            anomaly_count = anomaly_data['is_anomaly'].sum()
            st.metric("Anomalous Months", str(anomaly_count), help="Months with unusual revenue patterns")

    # ============================================================================
    # TAB 2: COMPARATIVE ANALYTICS
    # ============================================================================
    with tab2:
        st.header("📊 Comparative Analytics")

        compare_type = st.radio(
            "Comparison Type",
            ["Specialty Benchmark", "State Comparison", "Revenue Tier Comparison"],
            horizontal=True
        )

        if compare_type == "Specialty Benchmark":
            st.subheader("Specialty Benchmark Analysis")

            # Select specialty
            specialties = df['specialty'].unique().sort().to_list()
            selected_specialty = st.selectbox("Select Specialty", specialties)

            # Get specialty data
            specialty_df = df.filter(pl.col('specialty') == selected_specialty)

            # Calculate benchmarks
            monthly_benchmark = specialty_df.group_by('month').agg([
                pl.col('total_revenue').mean().alias('avg_revenue'),
                pl.col('total_revenue').median().alias('median_revenue'),
                pl.col('prescription_count').mean().alias('avg_prescriptions'),
                pl.col('unique_drugs').mean().alias('avg_unique_drugs'),
                pl.len().alias('n_prescribers')
            ]).sort('month')

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_revenue_trend_chart(monthly_benchmark.rename({'avg_revenue': 'total_revenue'})),
                    width="stretch"
                )

            with col2:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_benchmark['month'].to_list(),
                    y=monthly_benchmark['n_prescribers'].to_list(),
                    mode='lines+markers',
                    name='Active Prescribers',
                    line=dict(color='#2ca02c', width=3)
                ))
                fig.update_layout(
                    title=f'Active Prescribers in {selected_specialty}',
                    xaxis_title='Month',
                    yaxis_title='Number of Prescribers',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

            # Summary stats
            st.markdown("### Specialty Summary Statistics")
            col3, col4, col5, col6 = st.columns(4)

            total_prescribers = specialty_df['PRESCRIBER_NPI_NBR'].n_unique()
            total_revenue = specialty_df['total_revenue'].sum()
            avg_monthly_rev = specialty_df.group_by('PRESCRIBER_NPI_NBR').agg(
                pl.col('total_revenue').mean()
            )['total_revenue'].mean()

            with col3:
                st.metric("Total Prescribers", f"{total_prescribers:,}")
            with col4:
                from utils.metrics import format_currency
                st.metric("Total Revenue", format_currency(total_revenue))
            with col5:
                st.metric("Avg Monthly Revenue", format_currency(avg_monthly_rev))
            with col6:
                avg_drugs = specialty_df['unique_drugs'].mean()
                st.metric("Avg Unique Drugs", f"{avg_drugs:.1f}")

        elif compare_type == "State Comparison":
            st.subheader("State-Level Comparison")

            # Get top states by revenue
            top_states = df.group_by('state').agg([
                pl.col('total_revenue').sum().alias('total_revenue'),
                pl.len().alias('observations')
            ]).sort('total_revenue', descending=True).head(10)

            import plotly.express as px
            fig = px.bar(
                top_states.to_pandas(),
                x='state',
                y='total_revenue',
                title='Top 10 States by Total Revenue',
                labels={'total_revenue': 'Total Revenue ($)', 'state': 'State'},
                color='total_revenue',
                color_continuous_scale='Blues'
            )
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, width="stretch")

            # State details table
            st.markdown("### State Details")
            st.dataframe(
                top_states.to_pandas(),
                width="stretch",
                hide_index=True
            )

        else:  # Revenue Tier Comparison
            st.subheader("Revenue Tier Analysis")

            # Calculate revenue tiers
            tier_df = df.group_by('PRESCRIBER_NPI_NBR').agg([
                pl.col('total_revenue').sum().alias('total_revenue')
            ])

            # Define percentiles (calculate individually - Polars doesn't support list)
            p10 = tier_df['total_revenue'].quantile(0.10)
            p20 = tier_df['total_revenue'].quantile(0.20)
            p40 = tier_df['total_revenue'].quantile(0.40)
            p60 = tier_df['total_revenue'].quantile(0.60)
            p80 = tier_df['total_revenue'].quantile(0.80)
            p90 = tier_df['total_revenue'].quantile(0.90)

            tier_df = tier_df.with_columns([
                pl.when(pl.col('total_revenue') <= p10)
                .then(pl.lit('Tier 6 (Bottom 10%)'))
                .when(pl.col('total_revenue') <= p20)
                .then(pl.lit('Tier 5 (10-20%)'))
                .when(pl.col('total_revenue') <= p40)
                .then(pl.lit('Tier 4 (20-40%)'))
                .when(pl.col('total_revenue') <= p60)
                .then(pl.lit('Tier 3 (Middle 40-60%)'))
                .when(pl.col('total_revenue') <= p80)
                .then(pl.lit('Tier 2 (60-80%)'))
                .when(pl.col('total_revenue') <= p90)
                .then(pl.lit('Tier 1 (Top 20%)'))
                .otherwise(pl.lit('Tier 0 (Top 10%)'))
                .alias('tier')
            ])

            # Join back to main df
            df_with_tiers = df.join(
                tier_df.select(['PRESCRIBER_NPI_NBR', 'tier']),
                on='PRESCRIBER_NPI_NBR',
                how='left'
            )

            # Calculate tier statistics
            tier_stats = df_with_tiers.group_by('tier').agg([
                pl.col('PRESCRIBER_NPI_NBR').n_unique().alias('n_prescribers'),
                pl.col('total_revenue').mean().alias('avg_monthly_revenue'),
                pl.col('total_revenue').sum().alias('total_tier_revenue'),
                pl.col('prescription_count').mean().alias('avg_prescriptions'),
                pl.col('unique_drugs').mean().alias('avg_unique_drugs')
            ]).sort('tier')

            # Add percentage of total revenue
            total_revenue_all = df_with_tiers['total_revenue'].sum()
            tier_stats = tier_stats.with_columns([
                ((pl.col('total_tier_revenue') / total_revenue_all) * 100).alias('pct_of_total_revenue')
            ])

            # Format for display
            tier_stats_display = tier_stats.to_pandas()
            from utils.metrics import format_currency
            tier_stats_display['avg_monthly_revenue'] = tier_stats_display['avg_monthly_revenue'].apply(format_currency)
            tier_stats_display['total_tier_revenue'] = tier_stats_display['total_tier_revenue'].apply(format_currency)
            tier_stats_display['pct_of_total_revenue'] = tier_stats_display['pct_of_total_revenue'].apply(lambda x: f"{x:.1f}%")
            tier_stats_display['avg_prescriptions'] = tier_stats_display['avg_prescriptions'].apply(lambda x: f"{x:.1f}")
            tier_stats_display['avg_unique_drugs'] = tier_stats_display['avg_unique_drugs'].apply(lambda x: f"{x:.1f}")

            st.dataframe(
                tier_stats_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "tier": "Revenue Tier",
                    "n_prescribers": "# Prescribers",
                    "avg_monthly_revenue": "Avg Monthly Revenue",
                    "total_tier_revenue": "Total Tier Revenue",
                    "pct_of_total_revenue": "% of Total Revenue",
                    "avg_prescriptions": "Avg Prescriptions",
                    "avg_unique_drugs": "Avg Unique Drugs"
                }
            )

            # Key insights
            st.markdown("### Key Insights")
            col1, col2, col3 = st.columns(3)

            top10_row = tier_stats.filter(pl.col('tier') == 'Tier 0 (Top 10%)')
            if len(top10_row) > 0:
                top10_pct = (top10_row['total_tier_revenue'][0] / total_revenue_all) * 100
                top10_prescribers = top10_row['n_prescribers'][0]

                with col1:
                    st.metric(
                        "Top 10% Revenue Share",
                        f"{top10_pct:.1f}%",
                        help="Percentage of total revenue from top 10% prescribers"
                    )

                with col2:
                    st.metric(
                        "Top 10% Count",
                        f"{top10_prescribers:,}",
                        help="Number of prescribers in top 10%"
                    )

                with col3:
                    avg_rev_top10 = top10_row['avg_monthly_revenue'][0]
                    st.metric(
                        "Avg Revenue (Top 10%)",
                        format_currency(avg_rev_top10),
                        help="Average monthly revenue for top 10%"
                    )

    # ============================================================================
    # TAB 3: DATA EXPLORER
    # ============================================================================
    with tab3:
        st.header("🔍 Data Explorer")

        st.markdown("""
        Explore the raw time-series data. Click on a prescriber name to view their individual dashboard.
        """)

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            specialty_filter = st.multiselect(
                "Filter by Specialty",
                options=df['specialty'].unique().sort().to_list(),
                default=None
            )

        with col2:
            state_filter = st.multiselect(
                "Filter by State",
                options=df['state'].unique().sort().to_list(),
                default=None
            )

        with col3:
            month_filter = st.multiselect(
                "Filter by Month",
                options=df['month'].unique().sort().to_list(),
                default=None
            )

        # Apply filters
        filtered_df = df

        if specialty_filter:
            filtered_df = filtered_df.filter(pl.col('specialty').is_in(specialty_filter))
        if state_filter:
            filtered_df = filtered_df.filter(pl.col('state').is_in(state_filter))
        if month_filter:
            filtered_df = filtered_df.filter(pl.col('month').is_in(month_filter))

        st.markdown(f"**Showing {len(filtered_df):,} observations**")

        # Display options
        display_cols = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns,
            default=[
                'PRESCRIBER_NPI_NBR', 'prescriber_name', 'month', 'prescription_count',
                'total_revenue', 'unique_drugs', 'specialty', 'state'
            ]
        )

        # Show data with clickable names
        if display_cols:
            display_df = filtered_df.select(display_cols).head(1000)  # Limit to first 1000 rows for performance

            st.dataframe(
                display_df.to_pandas(),
                width="stretch",
                hide_index=True,
                height=600
            )

            # Add note about clicking
            st.info("💡 To view a specific prescriber's dashboard, use the prescriber selector in the sidebar of the Individual Dashboard tab.")

            # Download full filtered data
            st.markdown("### Download Filtered Data")
            csv = filtered_df.write_csv()
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_prescriber_data.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
