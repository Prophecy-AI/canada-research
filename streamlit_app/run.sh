#!/bin/bash

# Launch script for High-Value Prescriber Analytics Dashboard

echo "🚀 Launching Prescriber Analytics Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if data file exists
DATA_FILE="../outputs/timeseries_prescriber_monthly.parquet"
DATA_FILE_CSV="../outputs/timeseries_prescriber_monthly.csv"

if [ ! -f "$DATA_FILE" ] && [ ! -f "$DATA_FILE_CSV" ]; then
    echo "⚠️  Warning: Time-series data file not found."
    echo "📊 Please run the timeseries_prescriber_dataset.ipynb notebook first."
    echo ""
    echo "Expected location:"
    echo "  - $DATA_FILE"
    echo "  - or $DATA_FILE_CSV"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

# Launch Streamlit
echo "✅ Starting dashboard..."
echo "🌐 Opening browser at http://localhost:8501"
echo ""
streamlit run app.py
