import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """Time series analysis for market data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get("time_series_window", 30)
        self.forecast_horizon = config.get("forecast_horizon", 7)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze time series data"""
        try:
            # Get historical data
            historical_data = self._get_historical_data()
            
            # Perform analysis
            analysis = {
                "trend": self._analyze_trend(historical_data),
                "seasonality": self._analyze_seasonality(historical_data),
                "volatility": self._analyze_volatility(historical_data),
                "patterns": self._identify_patterns(historical_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            return {}
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical market data"""
        try:
            # In a real implementation, this would fetch data from a database or API
            # For now, we'll create sample data
            dates = pd.date_range(
                end=datetime.now(),
                periods=self.window_size,
                freq='D'
            )
            
            data = pd.DataFrame({
                'date': dates,
                'price': np.random.normal(100, 10, self.window_size),
                'volume': np.random.normal(1000, 100, self.window_size)
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend"""
        try:
            if data.empty:
                return {}
            
            # Calculate moving averages
            ma7 = data['price'].rolling(window=7).mean()
            ma30 = data['price'].rolling(window=30).mean()
            
            # Calculate trend direction
            current_price = data['price'].iloc[-1]
            ma7_current = ma7.iloc[-1]
            ma30_current = ma30.iloc[-1]
            
            trend = {
                "direction": "up" if current_price > ma7_current > ma30_current else "down",
                "strength": abs(current_price - ma30_current) / ma30_current,
                "ma7": ma7_current,
                "ma30": ma30_current
            }
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {}
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        try:
            if data.empty:
                return {}
            
            # Calculate daily returns
            data['returns'] = data['price'].pct_change()
            
            # Group by day of week
            daily_patterns = data.groupby(data['date'].dt.dayofweek)['returns'].mean()
            
            seasonality = {
                "patterns": daily_patterns.to_dict(),
                "strongest_day": daily_patterns.idxmax(),
                "weakest_day": daily_patterns.idxmin()
            }
            
            return seasonality
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return {}
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price volatility"""
        try:
            if data.empty:
                return {}
            
            # Calculate daily returns
            data['returns'] = data['price'].pct_change()
            
            # Calculate volatility metrics
            volatility = {
                "daily_std": data['returns'].std(),
                "weekly_std": data['returns'].rolling(window=7).std().iloc[-1],
                "monthly_std": data['returns'].rolling(window=30).std().iloc[-1],
                "current_level": "high" if data['returns'].std() > 0.02 else "low"
            }
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            return {}
    
    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify price patterns"""
        try:
            if data.empty:
                return []
            
            patterns = []
            
            # Look for double tops/bottoms
            if self._is_double_top(data['price']):
                patterns.append({
                    "type": "double_top",
                    "confidence": 0.8,
                    "price_level": data['price'].max()
                })
            
            if self._is_double_bottom(data['price']):
                patterns.append({
                    "type": "double_bottom",
                    "confidence": 0.8,
                    "price_level": data['price'].min()
                })
            
            # Look for trend reversals
            if self._is_trend_reversal(data['price']):
                patterns.append({
                    "type": "trend_reversal",
                    "confidence": 0.7,
                    "direction": "up" if data['price'].iloc[-1] > data['price'].iloc[-2] else "down"
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return []
    
    def _is_double_top(self, prices: pd.Series) -> bool:
        """Check for double top pattern"""
        try:
            # Find local maxima
            peaks = prices[(
                (prices > prices.shift(1)) &
                (prices > prices.shift(-1))
            )]
            
            if len(peaks) < 2:
                return False
            
            # Check if last two peaks are similar
            last_two_peaks = peaks.tail(2)
            return abs(last_two_peaks.iloc[0] - last_two_peaks.iloc[1]) / last_two_peaks.iloc[0] < 0.02
            
        except Exception as e:
            logger.error(f"Error checking double top: {str(e)}")
            return False
    
    def _is_double_bottom(self, prices: pd.Series) -> bool:
        """Check for double bottom pattern"""
        try:
            # Find local minima
            troughs = prices[(
                (prices < prices.shift(1)) &
                (prices < prices.shift(-1))
            )]
            
            if len(troughs) < 2:
                return False
            
            # Check if last two troughs are similar
            last_two_troughs = troughs.tail(2)
            return abs(last_two_troughs.iloc[0] - last_two_troughs.iloc[1]) / last_two_troughs.iloc[0] < 0.02
            
        except Exception as e:
            logger.error(f"Error checking double bottom: {str(e)}")
            return False
    
    def _is_trend_reversal(self, prices: pd.Series) -> bool:
        """Check for trend reversal"""
        try:
            if len(prices) < 3:
                return False
            
            # Calculate short-term trend
            short_ma = prices.rolling(window=3).mean()
            long_ma = prices.rolling(window=7).mean()
            
            # Check for crossover
            current = short_ma.iloc[-1] > long_ma.iloc[-1]
            previous = short_ma.iloc[-2] <= long_ma.iloc[-2]
            
            return current and previous
            
        except Exception as e:
            logger.error(f"Error checking trend reversal: {str(e)}")
            return False 