# High-Value Prescriber Analytics Dashboard

A comprehensive Streamlit application for visualizing and analyzing prescriber behavioral patterns over time.

## Features

### 1. Individual Prescriber Dashboard
- **Summary Statistics Cards**: Lifetime revenue, current month performance, YoY growth, active months
- **Revenue & Volume Trends**: Interactive line charts with moving averages and specialty benchmarks
- **Growth Rate Analysis**: Month-over-month growth visualization
- **Portfolio Diversity**: Track unique drugs prescribed over time
- **Payer Mix Evolution**: Stacked area chart showing Commercial/Medicare/Medicaid distribution
- **Patient Retention Proxy**: 90-day supply rate trends
- **Clinical Metrics**: Conditions treated and procedure rates
- **Forecasting**: Simple linear forecast for next 3-6 months
- **Key Insights**: Percentile ranking, trend direction, anomaly detection

### 2. Comparative Analytics
- **Specialty Benchmark**: Compare against specialty-wide averages
- **State Comparison**: Top states by revenue
- **Revenue Tier Analysis**: Quintile-based segmentation

### 3. Data Explorer
- **Interactive Table**: Browse raw time-series data with filters
- **Multi-column Filtering**: Filter by specialty, state, month
- **Column Selection**: Choose which columns to display
- **CSV Export**: Download filtered data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your time-series data file exists:
   - Expected location: `../outputs/timeseries_prescriber_monthly.parquet` (or `.csv`)
   - Run the `timeseries_prescriber_dataset.ipynb` notebook first if data doesn't exist

## Usage

Launch the dashboard:
```bash
streamlit run app.py
```

Or use the launch script:
```bash
./run.sh
```

The app will open in your default browser at http://localhost:8501

## Project Structure

```
streamlit_app/
├── app.py                      # Main Streamlit application
├── utils/
│   ├── data_loader.py         # Data loading and filtering utilities
│   ├── charts.py              # Plotly chart creation functions
│   ├── metrics.py             # Statistical calculations
│   └── components.py          # Reusable UI components
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Data Requirements

The app expects a Polars-compatible time-series dataset with the following columns:

**Required:**
- `PRESCRIBER_NPI_NBR`: Prescriber NPI (integer)
- `prescriber_name`: Prescriber name (string)
- `month`: Month in YYYY-MM format (string)
- `prescription_count`: Number of prescriptions (integer)
- `total_revenue`: Total revenue (float)
- `unique_drugs`: Number of unique drugs prescribed (integer)

**Optional (for full functionality):**
- `specialty`: Clinical specialty (string)
- `state`: Practice state (string)
- `total_revenue_ma3`: 3-month moving average (float)
- `prescription_growth_mom`: Month-over-month growth rate (float)
- `revenue_growth_mom`: Revenue MoM growth rate (float)
- `commercial_rate`, `medicare_rate`, `medicaid_rate`: Payer mix (floats 0-1)
- `supply_90day_rate`: 90-day supply rate (float 0-1)
- `unique_conditions`: Number of conditions treated (integer)
- `procedure_rate`: Procedure involvement rate (float 0-1)
- `receives_payments`: Binary payment indicator (integer 0/1)
- `total_payments`: Total pharma payments (float)

## Features by Tab

### Tab 1: Individual Dashboard
- Sidebar prescriber selector with search
- Quick filters (Top by Revenue, Fastest Growing, Most Active, Random)
- Date range filtering
- Export prescriber data to CSV
- 8 interactive charts
- 8 summary metric cards
- 3 key insight metrics

### Tab 2: Comparative Analytics
- Specialty benchmark analysis with time-series charts
- State-level revenue comparison
- Revenue tier segmentation (quintiles)
- Aggregate statistics tables

### Tab 3: Data Explorer
- Full data table view (up to 1000 rows)
- Multi-select filters for specialty, state, month
- Customizable column display
- CSV export of filtered data

## Performance

- **Data Loading**: Cached with `@st.cache_data` for 1-hour TTL
- **Chart Rendering**: Plotly for interactive, performant visualizations
- **Data Processing**: Polars for fast dataframe operations
- **Display Limit**: Data Explorer shows first 1000 rows (full export available)

## Troubleshooting

**Data file not found:**
- Ensure you've run `timeseries_prescriber_dataset.ipynb` notebook
- Check that output file exists in `../outputs/` directory
- Verify file format (parquet or csv)

**Memory issues:**
- Use parquet format instead of CSV for better performance
- Consider filtering data to specific time ranges
- Close other applications if running locally

**Charts not displaying:**
- Check that required columns exist in your dataset
- Verify data types match expected formats
- Look for null values in key columns

## Technologies

- **Streamlit**: Web application framework
- **Polars**: High-performance dataframe library
- **Plotly**: Interactive visualization library
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (minimal usage)

## License

Internal use only.
