"""Time Series Forecasting Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for time series forecasting competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for time series forecasting approach"""
        
        return """# Time Series Forecasting Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Temporal patterns: trend, seasonality, cycles
- Stationarity testing and transformations
- Lag features and rolling statistics
- Handle missing values in time series
- Forecast horizon and validation strategy

SUGGESTED APPROACHES:
- Statistical: ARIMA, SARIMA, Prophet
- ML: Lag features with XGBoost/LightGBM
- Deep Learning: LSTM, GRU, Transformer
- Ensemble multiple approaches

COMMON PITFALLS TO AVOID:
- Data leakage through improper splitting
- Not checking for stationarity
- Ignoring seasonal patterns
- Wrong validation (use time-based split)

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
