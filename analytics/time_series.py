import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Analyzes time series data for trends, patterns, and anomalies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window_size = config.get("window_size", 30)
        self.resample_frequency = config.get("resample_frequency", "1D")
        self.decomposition_method = config.get("decomposition_method", "additive")
        
        # Cache for recent analyses
        self.analysis_cache = {}
        self.max_cache_size = 10
        
        logger.info(f"Time series analyzer initialized with window size {self.window_size}")
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze time series data based on a natural language query
        
        Args:
            query: Natural language query describing the analysis needed
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing time series with query: {query}")
        
        try:
            # Check if this query is in the cache
            cache_key = self._generate_cache_key(query)
            if cache_key in self.analysis_cache:
                logger.info(f"Using cached analysis for query: {query}")
                return self.analysis_cache[cache_key]
            
            # Parse the query to determine what kind of analysis is needed
            analysis_type, data_source, parameters = self._parse_query(query)
            
            # Get data from the appropriate source
            data = self._get_data(data_source, parameters)
            
            if data is None or len(data) == 0:
                return {"status": "error", "message": "No data available for analysis"}
            
            # Apply the requested analysis
            results = self._apply_analysis(data, analysis_type, parameters)
            
            # Cache the results
            self._cache_analysis(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing time series: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key from a query"""
        # Simple hash function for the query
        return f"ts_{hash(query) % 1000000}"
    
    def _cache_analysis(self, key: str, results: Dict[str, Any]) -> None:
        """Add analysis results to the cache"""
        # Add to cache
        self.analysis_cache[key] = results
        
        # Remove oldest entries if cache is too large
        if len(self.analysis_cache) > self.max_cache_size:
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
    
    def _parse_query(self, query: str) -> tuple:
        """
        Parse a natural language query to determine analysis parameters
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (analysis_type, data_source, parameters)
        """
        query = query.lower()
        
        # Default values
        analysis_type = "trend"
        data_source = "sample"
        parameters = {}
        
        # Determine analysis type
        if "trend" in query:
            analysis_type = "trend"
        elif "seasonal" in query or "decompos" in query:
            analysis_type = "seasonal_decomposition"
        elif "anomaly" in query or "outlier" in query:
            analysis_type = "anomaly_detection"
        elif "feature" in query or "extract" in query:
            analysis_type = "feature_extraction"
        
        # Determine data source
        if "database" in query or "db" in query:
            data_source = "database"
        elif "file" in query or "csv" in query:
            data_source = "file"
        
        # Extract time window
        time_window_match = re.search(r'(last|past)\s+(\d+)\s+(day|week|month|year)s?', query)
        if time_window_match:
            amount = int(time_window_match.group(2))
            unit = time_window_match.group(3)
            
            if unit == "day":
                parameters["time_window"] = amount
            elif unit == "week":
                parameters["time_window"] = amount * 7
            elif unit == "month":
                parameters["time_window"] = amount * 30
            elif unit == "year":
                parameters["time_window"] = amount * 365
        
        # Extract specific parameters
        if "hourly" in query:
            parameters["frequency"] = "H"
        elif "daily" in query:
            parameters["frequency"] = "D"
        elif "weekly" in query:
            parameters["frequency"] = "W"
        elif "monthly" in query:
            parameters["frequency"] = "M"
        
        return analysis_type, data_source, parameters
    
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
        Generate sample time series data for testing
        
        Args:
            parameters: Data generation parameters
            
        Returns:
            Pandas DataFrame with time series data
        """
        # Data parameters
        days = parameters.get("time_window", self.window_size)
        frequency = parameters.get("frequency", "D")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if frequency == "H":
            periods = days * 24
            freq = "H"
        elif frequency == "D":
            periods = days
            freq = "D"
        elif frequency == "W":
            periods = days // 7 + 1
            freq = "W"
        elif frequency == "M":
            periods = days // 30 + 1
            freq = "M"
        else:
            periods = days
            freq = "D"
        
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Generate trend component
        trend = np.linspace(10, 15, num=len(dates))
        
        # Generate seasonal component (period depends on frequency)
        if frequency == "H":
            # Daily seasonality for hourly data
            seasonal_period = 24
        elif frequency == "D":
            # Weekly seasonality for daily data
            seasonal_period = 7
        elif frequency == "W":
            # Quarterly seasonality for weekly data
            seasonal_period = 13
        elif frequency == "M":
            # Yearly seasonality for monthly data
            seasonal_period = 12
        else:
            seasonal_period = 7
        
        t = np.arange(len(dates))
        seasonal = 2 * np.sin(2 * np.pi * t / seasonal_period)
        
        # Generate random component
        np.random.seed(42)  # For reproducibility
        random = np.random.normal(0, 0.5, size=len(dates))
        
        # Combine components
        values = trend + seasonal + random
        
        # Create DataFrame
        df = pd.DataFrame({"value": values}, index=dates)
        
        # Add some anomalies
        if len(df) > 5:
            # Add a few spike anomalies
            spike_indices = np.random.choice(len(df), size=2, replace=False)
            for idx in spike_indices:
                df.iloc[idx, 0] += 5 * np.random.choice([-1, 1])
            
            # Add a level shift anomaly
            if len(df) > 10:
                shift_start = len(df) // 2
                shift_end = shift_start + 3
                df.iloc[shift_start:shift_end, 0] += 3
        
        return df
    
    def _apply_analysis(
        self, 
        data: pd.DataFrame, 
        analysis_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply the requested analysis to the data
        
        Args:
            data: Time series data
            analysis_type: Type of analysis to apply
            parameters: Analysis parameters
            
        Returns:
            Dictionary with analysis results
        """
        if analysis_type == "trend":
            return self._analyze_trend(data, parameters)
        elif analysis_type == "seasonal_decomposition":
            return self._analyze_seasonal_decomposition(data, parameters)
        elif analysis_type == "anomaly_detection":
            return self._analyze_anomaly_detection(data, parameters)
        elif analysis_type == "feature_extraction":
            return self._analyze_feature_extraction(data, parameters)
        else:
            return {
                "status": "error",
                "message": f"Unknown analysis type: {analysis_type}"
            }
    
    def _analyze_trend(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trend in time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Extract the time series
            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
            else:
                series = data
            
            # Calculate basic trend metrics
            start_value = float(series.iloc[0])
            end_value = float(series.iloc[-1])
            
            percent_change = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0
            
            # Determine trend direction
            if percent_change > 3:
                direction = "increasing"
            elif percent_change < -3:
                direction = "decreasing"
            else:
                direction = "stable"
            
            # Prepare data for response (convert to list of time, value points)
            time_series_data = []
            for idx, value in enumerate(series):
                time_series_data.append({
                    "time": series.index[idx].isoformat(),
                    "value": float(value)
                })
            
            # Prepare trend data
            trend_data = {
                "direction": direction,
                "start_value": start_value,
                "end_value": end_value,
                "percent_change": percent_change
            }
            
            # Calculate statistics
            stats = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median())
            }
            
            return {
                "status": "success",
                "analysis_type": "trend",
                "time_series": time_series_data,
                "trend": trend_data,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing trend: {str(e)}"
            }
    
    def _analyze_seasonal_decomposition(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with decomposition results
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Extract the time series
            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
            else:
                series = data
            
            # Determine the period based on the data frequency
            if len(series) >= 24 and series.index.freq == 'H':
                period = 24  # Daily cycle for hourly data
            elif len(series) >= 7 and series.index.freq == 'D':
                period = 7   # Weekly cycle for daily data
            elif len(series) >= 12 and series.index.freq == 'M':
                period = 12  # Yearly cycle for monthly data
            else:
                # Default to 7 if we can't determine or if it's too short
                period = min(7, len(series) // 2)
            
            # Ensure period is at least 2 and not larger than the series length
            period = max(2, min(period, len(series) // 2))
            
            # Apply decomposition
            model = parameters.get("decomposition_method", self.decomposition_method)
            result = seasonal_decompose(series, model=model, period=period)
            
            # Prepare data for response
            components = {}
            
            # Original series
            components["original"] = []
            for idx, value in enumerate(series):
                components["original"].append({
                    "time": series.index[idx].isoformat(),
                    "value": float(value)
                })
            
            # Trend component
            components["trend"] = []
            for idx, value in enumerate(result.trend.dropna()):
                components["trend"].append({
                    "time": result.trend.dropna().index[idx].isoformat(),
                    "value": float(value)
                })
            
            # Seasonal component
            components["seasonal"] = []
            for idx, value in enumerate(result.seasonal.dropna()):
                components["seasonal"].append({
                    "time": result.seasonal.dropna().index[idx].isoformat(),
                    "value": float(value)
                })
            
            # Residual component
            components["residual"] = []
            for idx, value in enumerate(result.resid.dropna()):
                components["residual"].append({
                    "time": result.resid.dropna().index[idx].isoformat(),
                    "value": float(value)
                })
            
            return {
                "status": "success",
                "analysis_type": "seasonal_decomposition",
                "components": components,
                "period": period,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {str(e)}")
            return {
                "status": "error",
                "message": f"Error in seasonal decomposition: {str(e)}"
            }
    
    def _analyze_anomaly_detection(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Extract the time series
            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
            else:
                series = data
            
            # Simple anomaly detection using mean and standard deviation
            threshold = parameters.get("threshold", 2.0)  # Default: 2 standard deviations
            
            mean = series.mean()
            std = series.std()
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Find anomalies
            anomalies = series[(series < lower_bound) | (series > upper_bound)]
            
            # Prepare data for response
            time_series_data = []
            for idx, value in enumerate(series):
                time_series_data.append({
                    "time": series.index[idx].isoformat(),
                    "value": float(value)
                })
            
            anomaly_data = []
            for idx, value in enumerate(anomalies):
                anomaly_data.append({
                    "time": anomalies.index[idx].isoformat(),
                    "value": float(value),
                    "deviation": float((value - mean) / std)
                })
            
            return {
                "status": "success",
                "analysis_type": "anomaly_detection",
                "time_series": time_series_data,
                "anomalies": anomaly_data,
                "threshold": threshold,
                "mean": float(mean),
                "std": float(std),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                "status": "error",
                "message": f"Error detecting anomalies: {str(e)}"
            }
    
    def _analyze_feature_extraction(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with feature extraction results
        """
        try:
            # Extract the time series
            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
            else:
                series = data
            
            # Extract features
            features = {}
            
            # Statistical features
            features["mean"] = float(series.mean())
            features["std"] = float(series.std())
            features["min"] = float(series.min())
            features["max"] = float(series.max())
            features["median"] = float(series.median())
            features["range"] = float(series.max() - series.min())
            
            # Trend features
            features["start_value"] = float(series.iloc[0])
            features["end_value"] = float(series.iloc[-1])
            features["percent_change"] = float(((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100) if series.iloc[0] != 0 else 0
            
            # Volatility features
            features["variance"] = float(series.var())
            features["skewness"] = float(series.skew())
            features["kurtosis"] = float(series.kurtosis())
            
            # Autocorrelation (lag 1)
            try:
                import scipy.stats as stats
                acf_lag1 = series.autocorr(lag=1)
                features["autocorrelation_lag1"] = float(acf_lag1) if not np.isnan(acf_lag1) else 0
            except:
                features["autocorrelation_lag1"] = 0
            
            # Prepare data for response
            time_series_data = []
            for idx, value in enumerate(series):
                time_series_data.append({
                    "time": series.index[idx].isoformat(),
                    "value": float(value)
                })
            
            return {
                "status": "success",
                "analysis_type": "feature_extraction",
                "time_series": time_series_data,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {
                "status": "error",
                "message": f"Error extracting features: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the time series analyzer
        
        Returns:
            Dictionary with status information
        """
        return {
            "window_size": self.window_size,
            "resample_frequency": self.resample_frequency,
            "decomposition_method": self.decomposition_method,
            "cached_analyses": len(self.analysis_cache)
        }