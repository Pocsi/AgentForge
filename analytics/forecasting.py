import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
        self.methods = config.get("methods", ["arima", "naive"])
        
        # Distributed compute for parallel processing
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
        logger.info(f"Forecasting with query: {query}")
        
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
            
            if data is None or len(data) == 0:
                return {"status": "error", "message": "No data available for forecasting"}
            
            # Generate forecast
            results = self._generate_forecast(data, horizon, parameters)
            
            # Cache the results
            self._cache_forecast(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error forecasting: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key from a query"""
        # Simple hash function for the query
        return f"forecast_{hash(query) % 1000000}"
    
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
        parameters = {}
        
        # Extract forecast horizon
        horizon_match = re.search(r'(next|following)\s+(\d+)\s+(day|week|month|year)s?', query)
        if horizon_match:
            amount = int(horizon_match.group(2))
            unit = horizon_match.group(3)
            
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
        elif "file" in query or "csv" in query:
            data_source = "file"
        
        # Extract history length for analysis
        history_match = re.search(r'(using|with|based on|from)\s+(last|past)\s+(\d+)\s+(day|week|month|year)s?', query)
        if history_match:
            amount = int(history_match.group(3))
            unit = history_match.group(4)
            
            if unit == "day":
                parameters["history_window"] = amount
            elif unit == "week":
                parameters["history_window"] = amount * 7
            elif unit == "month":
                parameters["history_window"] = amount * 30
            elif unit == "year":
                parameters["history_window"] = amount * 365
        
        # Extract model preferences
        if "arima" in query:
            parameters["preferred_model"] = "arima"
        elif "prophet" in query:
            parameters["preferred_model"] = "prophet"
        elif "lstm" in query or "neural" in query:
            parameters["preferred_model"] = "lstm"
        elif "ensemble" in query:
            parameters["preferred_model"] = "ensemble"
        
        # Extract confidence interval
        ci_match = re.search(r'(\d+)%\s+confidence', query)
        if ci_match:
            parameters["confidence_interval"] = float(ci_match.group(1)) / 100.0
        
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
        if data_source == "database" or data_source == "file":
            # In a real implementation, this would connect to databases or files
            # For now, just generate sample data
            logger.info(f"Using sample data for {data_source} source")
            return self._generate_sample_data(parameters)
        else:
            # Generate sample data
            return self._generate_sample_data(parameters)
    
    def _generate_sample_data(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate sample time series data for forecasting
        
        Args:
            parameters: Data generation parameters
            
        Returns:
            Pandas DataFrame with time series data
        """
        # Data parameters
        days = parameters.get("history_window", 90)  # Default to 90 days of history
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Generate trend component (linear with slight exponential growth)
        trend = np.linspace(100, 200, num=len(dates))
        trend = trend * (1 + 0.001 * np.arange(len(dates)))
        
        # Generate seasonal component (weekly pattern)
        t = np.arange(len(dates))
        weekly_pattern = 10 * np.sin(2 * np.pi * t / 7)
        
        # Add a long-term seasonal component (yearly pattern)
        yearly_pattern = 30 * np.sin(2 * np.pi * t / 365)
        
        # Generate random component
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 5, size=len(dates))
        
        # Combine components
        values = trend + weekly_pattern + yearly_pattern + noise
        
        # Create DataFrame
        df = pd.DataFrame({"value": values}, index=dates)
        
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
        # Extract the time series
        if isinstance(data, pd.DataFrame):
            series = data.iloc[:, 0]
        else:
            series = data
        
        # Forecast dates (continuing from data's last date)
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")
        
        # Choose forecasting method
        method = parameters.get("preferred_model", "ensemble")
        confidence_interval = parameters.get("confidence_interval", self.confidence_interval)
        
        forecasts = []
        
        # Generate forecasts with different methods
        if method == "ensemble" or method == "arima" or method in self.methods:
            arima_forecast = self._forecast_arima(series, horizon, confidence_interval)
            forecasts.append(arima_forecast)
        
        if (method == "ensemble" or method == "prophet" or method in self.methods) and "prophet" in self.methods:
            try:
                prophet_forecast = self._forecast_prophet(series, horizon, confidence_interval)
                forecasts.append(prophet_forecast)
            except ImportError:
                logger.warning("Prophet not available, skipping prophet forecast")
        
        if (method == "ensemble" or method == "lstm" or method in self.methods) and "lstm" in self.methods:
            try:
                lstm_forecast = self._forecast_lstm(series, horizon, confidence_interval)
                forecasts.append(lstm_forecast)
            except ImportError:
                logger.warning("Deep learning libraries not available, skipping LSTM forecast")
        
        # Always include naive forecast as a baseline
        naive_forecast = self._naive_forecast(series, horizon, confidence_interval)
        if method != "ensemble":
            forecasts = [naive_forecast]  # Only use naive as fallback
        else:
            forecasts.append(naive_forecast)
        
        # Create ensemble forecast if multiple methods were used
        if len(forecasts) > 1:
            final_forecast = self._create_ensemble_forecast(forecasts, forecast_dates)
            model_name = "Ensemble"
        else:
            final_forecast = forecasts[0]
            model_name = "ARIMA" if method == "arima" else "Prophet" if method == "prophet" else "LSTM" if method == "lstm" else "Naive"
        
        # Prepare historical data for visualization
        historical_data = []
        for idx, value in enumerate(series):
            historical_data.append({
                "time": series.index[idx].isoformat(),
                "value": float(value)
            })
        
        # Prepare forecast data
        forecast_data = []
        lower_bound_data = []
        upper_bound_data = []
        
        for i in range(len(forecast_dates)):
            # Main forecast
            forecast_data.append({
                "time": forecast_dates[i].isoformat(),
                "value": float(final_forecast["forecast"][i])
            })
            
            # Bounds
            if "lower_bound" in final_forecast and "upper_bound" in final_forecast:
                lower_bound_data.append({
                    "time": forecast_dates[i].isoformat(),
                    "value": float(final_forecast["lower_bound"][i])
                })
                
                upper_bound_data.append({
                    "time": forecast_dates[i].isoformat(),
                    "value": float(final_forecast["upper_bound"][i])
                })
        
        return {
            "status": "success",
            "model": model_name,
            "horizon": horizon,
            "confidence_interval": confidence_interval,
            "historical_data": historical_data,
            "forecast_data": forecast_data,
            "lower_bound": lower_bound_data,
            "upper_bound": upper_bound_data
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
            logger.warning("Statsmodels not available, using naive forecast")
            return self._naive_forecast(series, horizon, confidence_interval)
        
        try:
            # Try to fit ARIMA model
            # Use auto_arima to determine optimal order if available, otherwise use default (1,1,1)
            try:
                from pmdarima import auto_arima
                model = auto_arima(
                    series,
                    seasonal=True,
                    m=7,  # Weekly seasonality
                    max_p=5,
                    max_d=2,
                    max_q=5,
                    suppress_warnings=True,
                    error_action="ignore"
                )
                order = model.order
                seasonal_order = model.seasonal_order
                
                # Re-fit with statsmodels ARIMA for forecasting
                model = sm.tsa.statespace.SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order
                )
            except ImportError:
                # Default ARIMA(1,1,1) model
                model = ARIMA(series, order=(1, 1, 1))
            
            # Fit the model
            fit_model = model.fit()
            
            # Make forecast
            forecast = fit_model.forecast(steps=horizon)
            forecast_values = forecast.values if hasattr(forecast, 'values') else np.array(forecast)
            
            # Get prediction intervals
            alpha = 1 - confidence_interval
            try:
                pred_interval = fit_model.get_forecast(steps=horizon).conf_int(alpha=alpha)
                lower_bound = pred_interval.iloc[:, 0].values
                upper_bound = pred_interval.iloc[:, 1].values
            except:
                # Fallback method for confidence intervals
                std_errors = np.sqrt(fit_model.cov_params().diagonal())
                std_forecast = np.std(forecast_values) if len(forecast_values) > 1 else std_errors.mean()
                
                z_value = 1.96  # Approximation for 95% confidence
                if confidence_interval > 0.95:
                    z_value = 2.58  # Approximation for 99% confidence
                elif confidence_interval < 0.9:
                    z_value = 1.65  # Approximation for 90% confidence
                
                margin = z_value * std_forecast
                lower_bound = forecast_values - margin
                upper_bound = forecast_values + margin
            
            return {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast: {str(e)}")
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
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet (requires specific format)
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(interval_width=confidence_interval)
            model.fit(df)
            
            # Make future dataframe for prediction
            future = model.make_future_dataframe(periods=horizon)
            
            # Forecast
            forecast = model.predict(future)
            
            # Extract results (only for the forecast horizon)
            forecast_values = forecast['yhat'].values[-horizon:]
            lower_bound = forecast['yhat_lower'].values[-horizon:]
            upper_bound = forecast['yhat_upper'].values[-horizon:]
            
            return {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
        except ImportError:
            logger.warning("Prophet not available, using naive forecast")
            return self._naive_forecast(series, horizon, confidence_interval)
        except Exception as e:
            logger.error(f"Error in Prophet forecast: {str(e)}")
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
        try:
            # In a real implementation, this would use TensorFlow or PyTorch for LSTM
            # For simplicity, we'll use a basic statistical method as a proxy
            
            # Use simple exponential smoothing as a proxy for LSTM
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Weekly seasonality
            )
            fit_model = model.fit()
            
            # Forecast
            forecast = fit_model.forecast(horizon)
            forecast_values = forecast.values if hasattr(forecast, 'values') else np.array(forecast)
            
            # Simple confidence intervals (based on in-sample error)
            residuals = fit_model.resid
            resid_std = np.std(residuals)
            
            z_value = 1.96  # Approximation for 95% confidence
            if confidence_interval > 0.95:
                z_value = 2.58  # Approximation for 99% confidence
            elif confidence_interval < 0.9:
                z_value = 1.65  # Approximation for 90% confidence
            
            margin = z_value * resid_std
            lower_bound = forecast_values - margin
            upper_bound = forecast_values + margin
            
            return {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM forecast: {str(e)}")
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
        # Use the mean of the last 7 days for the forecast
        window = min(7, len(series))
        last_values = series[-window:].values
        forecast_value = np.mean(last_values)
        
        # Create constant forecast
        forecast_values = np.full(horizon, forecast_value)
        
        # Calculate standard deviation for confidence intervals
        std_dev = np.std(last_values)
        
        z_value = 1.96  # Approximation for 95% confidence
        if confidence_interval > 0.95:
            z_value = 2.58  # Approximation for 99% confidence
        elif confidence_interval < 0.9:
            z_value = 1.65  # Approximation for 90% confidence
        
        margin = z_value * std_dev
        lower_bound = forecast_values - margin
        upper_bound = forecast_values + margin
        
        return {
            "forecast": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
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
        forecast_arrays = [f["forecast"] for f in forecasts]
        ensemble_forecast = np.mean(forecast_arrays, axis=0)
        
        # Average the lower bounds
        lower_arrays = [f["lower_bound"] for f in forecasts if "lower_bound" in f]
        ensemble_lower = np.mean(lower_arrays, axis=0) if lower_arrays else None
        
        # Average the upper bounds
        upper_arrays = [f["upper_bound"] for f in forecasts if "upper_bound" in f]
        ensemble_upper = np.mean(upper_arrays, axis=0) if upper_arrays else None
        
        result = {
            "forecast": ensemble_forecast
        }
        
        if ensemble_lower is not None and ensemble_upper is not None:
            result["lower_bound"] = ensemble_lower
            result["upper_bound"] = ensemble_upper
        
        return result
    
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
            "statsmodels_available": STATSMODELS_AVAILABLE,
            "cached_forecasts": len(self.forecast_cache)
        }