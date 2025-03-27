import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re

# Import specialized forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

logger = logging.getLogger(__name__)

class Forecaster:
    """
    Performs time series forecasting with various models
    """
    
    def __init__(self, config: Dict[str, Any], distributed_compute=None):
        """
        Initialize the forecaster
        
        Args:
            config: Configuration dictionary
            distributed_compute: Distributed computation engine
        """
        self.config = config
        self.prediction_horizon = config.get("prediction_horizon", 7)
        self.confidence_interval = config.get("confidence_interval", 0.95)
        self.methods = config.get("methods", ["arima", "prophet", "lstm"])
        
        # Distributed compute for parallel model training
        self.distributed_compute = distributed_compute
        
        # Cache for recent forecasts
        self.forecast_cache = {}
        self.max_cache_size = 10
        
        logger.info(f"Forecaster initialized with horizon {self.prediction_horizon}")
    
    def forecast(self, query: str) -> Dict[str, Any]:
        """
        Generate forecasts based on a natural language query
        
        Args:
            query: Natural language query describing the forecast needed
            
        Returns:
            Dictionary with forecast results
        """
        logger.info(f"Generating forecast with query: {query}")
        
        try:
            # Check if this query is in the cache
            cache_key = self._generate_cache_key(query)
            if cache_key in self.forecast_cache:
                logger.info(f"Using cached forecast for query: {query}")
                return self.forecast_cache[cache_key]
            
            # Parse the query to determine forecast parameters
            horizon, data_source, parameters = self._parse_query(query)
            
            # Get data from the appropriate source
            data = self._get_data(data_source, parameters)
            
            if data.empty:
                return {"status": "error", "message": "No data available for forecasting"}
            
            # Generate the forecast
            results = self._generate_forecast(data, horizon, parameters)
            
            # Cache the results
            self._cache_forecast(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key from a query"""
        # Simple hash function for the query
        return f"fc_{hash(query) % 1000000}"
    
    def _cache_forecast(self, key: str, results: Dict[str, Any]) -> None:
        """Add forecast results to the cache"""
        # Add to cache
        self.forecast_cache[key] = results
        
        # Remove oldest entries if cache is too large
        if len(self.forecast_cache) > self.max_cache_size:
            oldest_key = next(iter(self.forecast_cache))
            del self.forecast_cache[oldest_key]
    
    def _parse_query(self, query: str) -> Tuple[int, str, Dict[str, Any]]:
        """
        Parse a natural language query to determine forecast parameters
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (horizon, data_source, parameters)
        """
        query = query.lower()
        
        # Default values
        horizon = self.prediction_horizon
        data_source = "sample"
        parameters = {
            "methods": self.methods.copy()
        }
        
        # Extract prediction horizon
        horizon_match = re.search(r'(?:next|for|predict)\s+(\d+)\s+(day|week|month|year)s?', query)
        if horizon_match:
            amount = int(horizon_match.group(1))
            unit = horizon_match.group(2)
            
            if unit == "day":
                horizon = amount
            elif unit == "week":
                horizon = amount * 7
            elif unit == "month":
                horizon = amount * 30
            elif unit == "year":
                horizon = amount * 365
        
        # Determine data source
        if "database" in query or "db" in query:
            data_source = "database"
        elif "csv" in query or "file" in query:
            data_source = "file"
        elif "api" in query or "service" in query:
            data_source = "api"
        elif "sensor" in query or "iot" in query:
            data_source = "sensor"
        
        # Determine forecast methods
        if "arima" in query:
            parameters["methods"] = ["arima"]
        elif "prophet" in query:
            parameters["methods"] = ["prophet"]
        elif "lstm" in query:
            parameters["methods"] = ["lstm"]
        elif "ensemble" in query:
            parameters["methods"] = ["arima", "prophet"]
        
        # Specific field or metric
        field_match = re.search(r'for\s+([a-zA-Z0-9_]+)', query)
        if field_match:
            parameters["field"] = field_match.group(1)
        
        return horizon, data_source, parameters
    
    def _get_data(self, data_source: str, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Get data from the specified source
        
        Args:
            data_source: Data source type
            parameters: Parameters for data retrieval
            
        Returns:
            Pandas DataFrame with time series data
        """
        # For now, we'll just generate sample data
        # In a real implementation, this would connect to databases, files, APIs, etc.
        
        # Generate date range
        window_days = parameters.get("window_days", 90)  # Use more historical data for forecasting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        # Generate sample data
        if data_source == "sensor" or data_source == "iot":
            # More variable data for sensors
            base = 100
            trend = np.linspace(0, 20, len(date_range))
            seasonality = 15 * np.sin(np.linspace(0, 4 * np.pi, len(date_range)))
            noise = np.random.normal(0, 5, len(date_range))
            values = base + trend + seasonality + noise
            
        else:
            # More stable data for general time series
            base = 500
            trend = np.linspace(0, 50, len(date_range))
            seasonality = 30 * np.sin(np.linspace(0, 2 * np.pi, len(date_range)))
            noise = np.random.normal(0, 10, len(date_range))
            values = base + trend + seasonality + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'value': values
        })
        
        return df
    
    def _generate_forecast(
        self, 
        data: pd.DataFrame, 
        horizon: int, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate forecast for the given data
        
        Args:
            data: Time series data
            horizon: Forecast horizon in days
            parameters: Forecast parameters
            
        Returns:
            Dictionary with forecast results
        """
        # Set timestamp as index if it's not already
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        
        # Select the field to forecast
        field = parameters.get("field", "value")
        if field not in data.columns and data.index.name != field:
            field = data.columns[0]
        
        series = data[field] if field in data.columns else data.iloc[:, 0]
        
        # Generate forecast dates
        last_date = series.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='1D'
        )
        
        # Determine which methods to use
        methods = parameters.get("methods", self.methods)
        
        # Generate forecasts using different methods
        forecasts = {}
        futures = []
        
        # Use distributed compute for parallel model training if available
        if self.distributed_compute:
            # Submit forecast jobs
            for method in methods:
                if method == "arima" and STATSMODELS_AVAILABLE:
                    futures.append((
                        method,
                        self.distributed_compute.submit(
                            self._forecast_arima,
                            series,
                            horizon,
                            self.confidence_interval
                        )
                    ))
                
                elif method == "prophet" and PROPHET_AVAILABLE:
                    futures.append((
                        method,
                        self.distributed_compute.submit(
                            self._forecast_prophet,
                            series,
                            horizon,
                            self.confidence_interval
                        )
                    ))
                
                elif method == "lstm":
                    futures.append((
                        method,
                        self.distributed_compute.submit(
                            self._forecast_lstm,
                            series,
                            horizon,
                            self.confidence_interval
                        )
                    ))
            
            # Collect results
            for method, future in futures:
                try:
                    forecasts[method] = future.result()
                except Exception as e:
                    logger.error(f"Error with {method} forecast: {str(e)}")
                    forecasts[method] = None
        
        else:
            # Sequential execution
            for method in methods:
                try:
                    if method == "arima" and STATSMODELS_AVAILABLE:
                        forecasts[method] = self._forecast_arima(series, horizon, self.confidence_interval)
                    
                    elif method == "prophet" and PROPHET_AVAILABLE:
                        forecasts[method] = self._forecast_prophet(series, horizon, self.confidence_interval)
                    
                    elif method == "lstm":
                        forecasts[method] = self._forecast_lstm(series, horizon, self.confidence_interval)
                
                except Exception as e:
                    logger.error(f"Error with {method} forecast: {str(e)}")
                    forecasts[method] = None
        
        # Create ensemble forecast if multiple methods were successful
        successful_forecasts = [f for f in forecasts.values() if f is not None]
        
        if len(successful_forecasts) > 1:
            ensemble_forecast = self._create_ensemble_forecast(successful_forecasts, forecast_dates)
            forecasts["ensemble"] = ensemble_forecast
        
        # Prepare forecast data for visualization
        forecast_data = {}
        for method, forecast in forecasts.items():
            if forecast is not None:
                forecast_data[method] = []
                
                for i, date in enumerate(forecast_dates):
                    forecast_data[method].append({
                        "timestamp": date.isoformat(),
                        "value": float(forecast["mean"][i]),
                        "lower_bound": float(forecast["lower"][i]),
                        "upper_bound": float(forecast["upper"][i])
                    })
        
        # Prepare historical data
        historical_data = []
        for timestamp, value in zip(series.index, series.values):
            historical_data.append({
                "timestamp": timestamp.isoformat(),
                "value": float(value)
            })
        
        return {
            "status": "success",
            "field": field,
            "horizon": horizon,
            "confidence_interval": self.confidence_interval,
            "methods": list(forecast_data.keys()),
            "historical_data": historical_data,
            "forecast_data": forecast_data
        }
    
    def _forecast_arima(
        self, 
        series: pd.Series, 
        horizon: int, 
        confidence_interval: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast using ARIMA model
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            confidence_interval: Confidence interval for prediction bounds
            
        Returns:
            Dictionary with forecast results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")
        
        # Simple automatic order selection (p,d,q)
        # In a real application, this would be more sophisticated
        try:
            # Try to find the best order using different combinations
            orders = [(1,1,1), (2,1,2), (1,1,0), (0,1,1)]
            best_model = None
            best_aic = float('inf')
            
            for order in orders:
                try:
                    model = ARIMA(series, order=order)
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_model = model_fit
                except:
                    continue
            
            if best_model is None:
                # Fallback to default order
                model = ARIMA(series, order=(1,1,1))
                best_model = model.fit()
            
            # Generate forecast
            forecast = best_model.forecast(steps=horizon)
            
            # Get prediction intervals
            alpha = 1 - confidence_interval
            pred_int = best_model.get_forecast(steps=horizon).conf_int(alpha=alpha)
            lower = pred_int.iloc[:, 0].values
            upper = pred_int.iloc[:, 1].values
            
            return {
                "mean": forecast.values,
                "lower": lower,
                "upper": upper
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast: {str(e)}")
            
            # Fallback to naive forecast
            return self._naive_forecast(series, horizon, confidence_interval)
    
    def _forecast_prophet(
        self, 
        series: pd.Series, 
        horizon: int, 
        confidence_interval: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast using Facebook Prophet
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            confidence_interval: Confidence interval for prediction bounds
            
        Returns:
            Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet forecasting")
        
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(interval_width=confidence_interval)
            model.fit(df)
            
            # Create future dataframe for prediction
            future = model.make_future_dataframe(periods=horizon)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast values
            mean = forecast.tail(horizon)['yhat'].values
            lower = forecast.tail(horizon)['yhat_lower'].values
            upper = forecast.tail(horizon)['yhat_upper'].values
            
            return {
                "mean": mean,
                "lower": lower,
                "upper": upper
            }
            
        except Exception as e:
            logger.error(f"Error in Prophet forecast: {str(e)}")
            
            # Fallback to naive forecast
            return self._naive_forecast(series, horizon, confidence_interval)
    
    def _forecast_lstm(
        self, 
        series: pd.Series, 
        horizon: int, 
        confidence_interval: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast using LSTM (simplified)
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            confidence_interval: Confidence interval for prediction bounds
            
        Returns:
            Dictionary with forecast results
        """
        # This is a simplified version that doesn't actually use LSTM
        # In a real application, this would use PyTorch or TensorFlow
        
        try:
            # Use simple exponential smoothing as a substitute
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Fit model
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Assume weekly seasonality
            )
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(horizon)
            
            # Simple confidence intervals based on historical variance
            std = series.std()
            z_value = 1.96  # Approximate 95% confidence
            if confidence_interval > 0.95:
                z_value = 2.576  # Approximate 99% confidence
            elif confidence_interval < 0.95:
                z_value = 1.645  # Approximate 90% confidence
            
            margin = z_value * std
            
            return {
                "mean": forecast.values,
                "lower": forecast.values - margin,
                "upper": forecast.values + margin
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM forecast: {str(e)}")
            
            # Fallback to naive forecast
            return self._naive_forecast(series, horizon, confidence_interval)
    
    def _naive_forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        confidence_interval: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate naive forecast (simple extrapolation)
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            confidence_interval: Confidence interval for prediction bounds
            
        Returns:
            Dictionary with forecast results
        """
        # Calculate trend from last 30 days (or less if not available)
        n = min(30, len(series))
        recent_values = series[-n:]
        
        if n >= 2:
            # Linear trend
            x = np.arange(n)
            y = recent_values.values
            
            # Simple linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Project forward
            forecast = np.array([intercept + slope * (n + i) for i in range(horizon)])
            
            # Confidence intervals based on recent variance
            std = recent_values.std()
            z_value = 1.96  # Approximate 95% confidence
            if confidence_interval > 0.95:
                z_value = 2.576  # Approximate 99% confidence
            elif confidence_interval < 0.95:
                z_value = 1.645  # Approximate 90% confidence
            
            margin = z_value * std
            
            return {
                "mean": forecast,
                "lower": forecast - margin,
                "upper": forecast + margin
            }
            
        else:
            # Not enough data, use last value
            last_value = series.iloc[-1]
            forecast = np.array([last_value] * horizon)
            
            # Arbitrary confidence interval
            margin = last_value * 0.1
            
            return {
                "mean": forecast,
                "lower": forecast - margin,
                "upper": forecast + margin
            }
    
    def _create_ensemble_forecast(
        self, 
        forecasts: List[Dict[str, np.ndarray]],
        dates: pd.DatetimeIndex
    ) -> Dict[str, np.ndarray]:
        """
        Create an ensemble forecast by averaging multiple forecasts
        
        Args:
            forecasts: List of individual forecasts
            dates: Forecast dates
            
        Returns:
            Dictionary with ensemble forecast
        """
        # Average the forecasts
        mean_values = np.mean([f["mean"] for f in forecasts], axis=0)
        lower_values = np.mean([f["lower"] for f in forecasts], axis=0)
        upper_values = np.mean([f["upper"] for f in forecasts], axis=0)
        
        return {
            "mean": mean_values,
            "lower": lower_values,
            "upper": upper_values
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the forecaster
        
        Returns:
            Dictionary with status information
        """
        return {
            "prediction_horizon": self.prediction_horizon,
            "confidence_interval": self.confidence_interval,
            "methods": self.methods,
            "cache_size": len(self.forecast_cache),
            "max_cache_size": self.max_cache_size,
            "statsmodels_available": STATSMODELS_AVAILABLE,
            "prophet_available": PROPHET_AVAILABLE
        }
