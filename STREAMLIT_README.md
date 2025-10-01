# Prescriber Time-Series Analytics Dashboard

Interactive Streamlit dashboard for exploring prescriber trends and patterns.

## Features

### ✅ Implemented Features

1. **Prescriber Selection**
   - Searchable dropdown with name + NPI
   - Quick filters: Top 10 by revenue, Top 10 by volume, Random prescriber
   - Custom search functionality

2. **Summary Stats Cards**
   - Lifetime revenue
   - Current month revenue
   - Year-over-year growth
   - Active months
   - Pharma payment status
   - Specialty and state info

3. **Interactive Visualizations**
   - **Revenue Trend**: Monthly revenue with 3-month moving average
   - **Volume Trend**: Prescription count + unique drugs (dual-axis)
   - **Growth Rate**: Month-over-month growth rate bar chart
   - **Payer Mix Evolution**: Stacked area chart (Commercial/Medicare/Medicaid)
   - **Retention Metrics**: 90-day supply rate + avg days supply
   - **Clinical Activity**: Unique conditions + procedure rate
   - **Seasonality Heatmap**: Revenue by month across years

4. **Comparative Analytics**
   - Peer benchmarking: Compare to specialty average
   - Percentile ranking within specialty
   - Revenue forecast (3-month projection)

5. **Time Range Selector**
   - All time
   - Last 12/6/3 months
   - Custom date range

6. **Export Features**
   - Download prescriber data as CSV
   - Download summary stats as CSV

## Installation

1. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Running the App

1. **First, generate the time-series data** by running the notebook:
   ```python
   # Run: research/timeseries_prescriber_dataset.ipynb
   # This creates: outputs/timeseries_prescriber_monthly.parquet
   ```

2. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in terminal

## Usage

1. **Select a Prescriber:**
   - Use quick filters (Top 10, Random) or search by name/NPI
   - Sidebar shows prescriber selection options

2. **Adjust Time Range:**
   - Choose preset ranges or custom date range
   - All charts update automatically

3. **Explore Charts:**
   - Hover over charts for detailed values
   - All charts are interactive (zoom, pan, select)
   - Toggle traces by clicking legend items

4. **Compare to Peers:**
   - Scroll to "Comparative Analytics" section
   - View specialty averages and percentile ranking
   - See revenue forecast

5. **Export Data:**
   - Download full prescriber time-series data
   - Download summary statistics
   - CSV format for easy analysis in Excel/Python

## Chart Descriptions

- **Revenue Trend**: Shows monthly revenue with 3-month moving average overlay
- **Volume & Portfolio Diversity**: Bar chart of prescription count with line overlay for unique drugs
- **Month-over-Month Revenue Growth**: Bar chart showing growth rate (green = positive, red = negative)
- **Payer Mix Evolution**: Stacked area showing proportion of Commercial/Medicare/Medicaid over time
- **Patient Retention Indicators**: 90-day supply rate (proxy for chronic therapy) and average days supply
- **Clinical Activity Metrics**: Unique conditions treated and procedure rate from medical claims
- **Revenue Seasonality Heatmap**: Heat map showing revenue patterns by month and year
- **Peer Benchmarking**: Your prescriber vs specialty average revenue
- **Revenue Forecast**: Simple linear projection for next 3 months

## Data Requirements

The app expects a parquet file at:
```
outputs/timeseries_prescriber_monthly.parquet
```

Required columns:
- `PRESCRIBER_NPI_NBR`: Prescriber NPI (integer)
- `prescriber_name`: Prescriber name (string)
- `month`: Month in YYYY-MM format (string)
- `total_revenue`: Monthly revenue (float)
- `prescription_count`: Monthly prescription count (integer)
- `unique_drugs`: Unique drugs prescribed (integer)
- `brand_rate`: Brand prescription rate (float)
- `commercial_rate`, `medicare_rate`, `medicaid_rate`: Payer mix (float)
- `supply_90day_rate`: 90-day supply rate (float)
- `avg_days_supply`: Average days supply (float)
- `unique_conditions`: Unique conditions from medical claims (integer)
- `procedure_rate`: Procedure rate from medical claims (float)
- `specialty`: Clinical specialty (string)
- `state`: State code (string)
- `receives_payments`: Binary indicator (0/1)
- `year`, `month_num`: Time features (integer)
- `total_revenue_ma3`: 3-month moving average (float)
- `revenue_growth_mom`: Month-over-month growth rate (float)

## Troubleshooting

**Error: "Data file not found"**
- Make sure you've run the `timeseries_prescriber_dataset.ipynb` notebook first
- Check that `outputs/timeseries_prescriber_monthly.parquet` exists

**Error: Column not found**
- Ensure all required columns are present in the dataset
- Re-run the time-series dataset notebook

**Slow performance**
- The app uses Polars for fast data processing
- Large datasets (>1M rows) may take a few seconds to load
- Data is cached after first load

**Charts not displaying**
- Check that Plotly is installed: `pip install plotly`
- Clear browser cache and reload

## Technology Stack

- **Streamlit**: Web app framework
- **Polars**: Fast dataframe operations
- **Plotly**: Interactive charts
- **NumPy**: Numerical computations

## Performance Notes

- Data is cached using `@st.cache_data` for fast reloads
- Polars operations are optimized for performance
- Charts render client-side for smooth interactions
- Typical load time: 2-5 seconds for 500K+ rows

## Future Enhancements

Potential additions:
- Multi-prescriber comparison view
- Anomaly detection highlights
- Advanced forecasting models (ARIMA, Prophet)
- Geographic map visualization
- Drug-level drill-down
- Export charts as PNG
- Custom metric builder
