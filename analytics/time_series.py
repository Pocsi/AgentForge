import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re

# Import specialized time series libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from tsfresh import extract_features
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Performs advanced time series analysis on data
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
            
            if data.empty:
                return {"status": "error", "message": "No data available for analysis"}
            
            # Perform the requested analysis
            results = self._perform_analysis(data, analysis_type, parameters)
            
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
    
    def _parse_query(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse a natural language query to determine the analysis needed
        
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
        if "trend" in query or "pattern" in query:
            analysis_type = "trend"
        elif "decompose" in query or "breakdown" in query:
            analysis_type = "decomposition"
        elif "anomaly" in query or "outlier" in query:
            analysis_type = "anomaly"
        elif "correlation" in query or "relationship" in query:
            analysis_type = "correlation"
        elif "feature" in query or "extract" in query:
            analysis_type = "feature_extraction"
        
        # Determine data source
        if "database" in query or "db" in query:
            data_source = "database"
        elif "csv" in query or "file" in query:
            data_source = "file"
        elif "api" in query or "service" in query:
            data_source = "api"
        elif "sensor" in query or "iot" in query:
            data_source = "sensor"
        
        # Extract parameters
        # Time window
        window_match = re.search(r'last\s+(\d+)\s+(day|week|month|year)s?', query)
        if window_match:
            amount = int(window_match.group(1))
            unit = window_match.group(2)
            
            if unit == "day":
                parameters["window_days"] = amount
            elif unit == "week":
                parameters["window_days"] = amount * 7
            elif unit == "month":
                parameters["window_days"] = amount * 30
            elif unit == "year":
                parameters["window_days"] = amount * 365
        
        # Specific field or metric
        field_match = re.search(r'for\s+([a-zA-Z0-9_]+)', query)
        if field_match:
            parameters["field"] = field_match.group(1)
        
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
        # For now, we'll just generate sample data
        # In a real implementation, this would connect to databases, files, APIs, etc.
        
        # Generate date range
        window_days = parameters.get("window_days", self.window_size)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.resample_frequency)
        
        # Generate sample data
        if data_source == "sensor" or data_source == "iot":
            # More variable data for sensors
            base = 100
            trend = np.linspace(0, 20, len(date_range))
            seasonality = 15 * np.sin(np.linspace(0, 4 * np.pi, len(date_range)))
            noise = np.random.normal(0, 5, len(date_range))
            values = base + trend + seasonality + noise
            
            # Add some anomalies
            anomaly_indices = np.random.choice(range(len(date_range)), size=3, replace=False)
            for idx in anomaly_indices:
                values[idx] += np.random.choice([-50, 50])
            
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
    
    def _perform_analysis(
        self, 
        data: pd.DataFrame, 
        analysis_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform the requested analysis on the data
        
        Args:
            data: Time series data
            analysis_type: Type of analysis to perform
            parameters: Analysis parameters
            
        Returns:
            Dictionary with analysis results
        """
        # Set timestamp as index if it's not already
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        
        if analysis_type == "trend":
            return self._analyze_trend(data, parameters)
        elif analysis_type == "decomposition":
            return self._decompose_time_series(data, parameters)
        elif analysis_type == "anomaly":
            return self._detect_anomalies(data, parameters)
        elif analysis_type == "correlation":
            return self._analyze_correlation(data, parameters)
        elif analysis_type == "feature_extraction":
            return self._extract_features(data, parameters)
        else:
            return {"status": "error", "message": f"Unknown analysis type: {analysis_type}"}
    
    def _analyze_trend(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the trend in time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with trend analysis results
        """
        # Select the field to analyze
        field = parameters.get("field", "value")
        if field not in data.columns and data.index.name != field:
            field = data.columns[0]
        
        series = data[field] if field in data.columns else data.iloc[:, 0]
        
        # Calculate basic statistics
        stats = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "median": series.median()
        }
        
        # Calculate moving averages
        window_short = max(3, len(series) // 10)
        window_long = max(7, len(series) // 5)
        
        ma_short = series.rolling(window=window_short).mean()
        ma_long = series.rolling(window=window_long).mean()
        
        # Determine overall trend
        overall_trend = "up" if series.iloc[-1] > series.iloc[0] else "down"
        
        # Calculate percentage change
        pct_change = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
        
        # Prepare time series data for visualization
        # Convert the time series to a format suitable for JSON serialization
        time_data = []
        for timestamp, value in zip(series.index, series.values):
            time_data.append({
                "timestamp": timestamp.isoformat(),
                "value": float(value),
                "ma_short": float(ma_short.get(timestamp, 0)) if not pd.isna(ma_short.get(timestamp, 0)) else None,
                "ma_long": float(ma_long.get(timestamp, 0)) if not pd.isna(ma_long.get(timestamp, 0)) else None
            })
        
        return {
            "status": "success",
            "analysis_type": "trend",
            "field": field,
            "stats": stats,
            "trend": {
                "direction": overall_trend,
                "percent_change": pct_change,
                "start_value": float(series.iloc[0]),
                "end_value": float(series.iloc[-1])
            },
            "moving_averages": {
                "short_window": window_short,
                "long_window": window_long
            },
            "time_series": time_data
        }
    
    def _decompose_time_series(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with decomposition results
        """
        if not STATSMODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Seasonal decomposition requires statsmodels package"
            }
        
        # Select the field to analyze
        field = parameters.get("field", "value")
        if field not in data.columns and data.index.name != field:
            field = data.columns[0]
        
        series = data[field] if field in data.columns else data.iloc[:, 0]
        
        # Ensure the series is regular
        if not isinstance(series.index, pd.DatetimeIndex):
            return {
                "status": "error",
                "message": "Decomposition requires datetime index"
            }
        
        # Decompose the time series
        try:
            # Determine period automatically if not specified
            period = parameters.get("period")
            if not period:
                # Try to infer from frequency
                if series.index.freq:
                    if series.index.freq.name == 'D':
                        period = 7  # Weekly seasonality
                    elif series.index.freq.name == 'M':
                        period = 12  # Yearly seasonality
                    elif series.index.freq.name == 'H':
                        period = 24  # Daily seasonality
                    else:
                        period = 12  # Default
                else:
                    # Default to 12 if can't determine
                    period = 12
            
            decomposition = seasonal_decompose(
                series, 
                model=self.decomposition_method,
                period=period
            )
            
            # Prepare components for visualization
            components = {}
            for component_name in ["trend", "seasonal", "resid"]:
                component_series = getattr(decomposition, component_name)
                component_data = []
                
                for timestamp, value in zip(component_series.index, component_series.values):
                    if not pd.isna(value):
                        component_data.append({
                            "timestamp": timestamp.isoformat(),
                            "value": float(value)
                        })
                
                components[component_name] = component_data
            
            return {
                "status": "success",
                "analysis_type": "decomposition",
                "field": field,
                "method": self.decomposition_method,
                "period": period,
                "components": components
            }
            
        except Exception as e:
            logger.error(f"Error decomposing time series: {str(e)}")
            return {
                "status": "error",
                "message": f"Error decomposing time series: {str(e)}"
            }
    
    def _detect_anomalies(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Select the field to analyze
        field = parameters.get("field", "value")
        if field not in data.columns and data.index.name != field:
            field = data.columns[0]
        
        series = data[field] if field in data.columns else data.iloc[:, 0]
        
        # Simple anomaly detection using mean and standard deviation
        # More sophisticated methods would be used in a real system
        mean = series.mean()
        std = series.std()
        threshold = parameters.get("threshold", 3.0)  # Default to 3 standard deviations
        
        upper_bound = mean + (threshold * std)
        lower_bound = mean - (threshold * std)
        
        # Find anomalies
        anomalies = []
        for timestamp, value in zip(series.index, series.values):
            if value > upper_bound or value < lower_bound:
                anomalies.append({
                    "timestamp": timestamp.isoformat(),
                    "value": float(value),
                    "deviation": (value - mean) / std
                })
        
        # Prepare time series data for visualization
        time_data = []
        for timestamp, value in zip(series.index, series.values):
            time_data.append({
                "timestamp": timestamp.isoformat(),
                "value": float(value),
                "upper_bound": float(upper_bound),
                "lower_bound": float(lower_bound),
                "is_anomaly": value > upper_bound or value < lower_bound
            })
        
        return {
            "status": "success",
            "analysis_type": "anomaly_detection",
            "field": field,
            "threshold": threshold,
            "anomalies": {
                "count": len(anomalies),
                "items": anomalies
            },
            "stats": {
                "mean": float(mean),
                "std": float(std),
                "upper_bound": float(upper_bound),
                "lower_bound": float(lower_bound)
            },
            "time_series": time_data
        }
    
    def _analyze_correlation(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlation between fields in the data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Need at least 2 columns for correlation
        if len(data.columns) < 2:
            return {
                "status": "error",
                "message": "Correlation analysis requires at least 2 data columns"
            }
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Convert to list format for easier JSON serialization
        correlations = []
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    correlations.append({
                        "field1": col1,
                        "field2": col2,
                        "correlation": float(corr_matrix.loc[col1, col2])
                    })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "status": "success",
            "analysis_type": "correlation",
            "correlations": correlations,
            "fields": list(data.columns)
        }
    
    def _extract_features(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from time series data
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with feature extraction results
        """
        if not TSFRESH_AVAILABLE:
            # Fallback to basic feature extraction
            return self._basic_feature_extraction(data, parameters)
        
        try:
            # Select the field to analyze
            field = parameters.get("field", "value")
            if field not in data.columns and data.index.name != field:
                field = data.columns[0]
            
            series = data[field] if field in data.columns else data.iloc[:, 0]
            
            # Prepare data for tsfresh
            df_for_extraction = pd.DataFrame({
                'id': 0,
                'time': range(len(series)),
                'value': series.values
            })
            
            # Extract features
            features = extract_features(df_for_extraction, column_id='id', column_sort='time')
            
            # Convert to a more readable format
            feature_list = []
            for feature_name, value in features.iloc[0].items():
                if not pd.isna(value):
                    feature_list.append({
                        "name": feature_name,
                        "value": float(value)
                    })
            
            # Sort by feature name
            feature_list.sort(key=lambda x: x["name"])
            
            return {
                "status": "success",
                "analysis_type": "feature_extraction",
                "field": field,
                "features": feature_list,
                "count": len(feature_list)
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return self._basic_feature_extraction(data, parameters)
    
    def _basic_feature_extraction(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic feature extraction when tsfresh is not available
        
        Args:
            data: Time series data
            parameters: Analysis parameters
            
        Returns:
            Dictionary with basic feature extraction results
        """
        # Select the field to analyze
        field = parameters.get("field", "value")
        if field not in data.columns and data.index.name != field:
            field = data.columns[0]
        
        series = data[field] if field in data.columns else data.iloc[:, 0]
        
        # Calculate basic statistical features
        features = [
            {"name": "mean", "value": float(series.mean())},
            {"name": "std", "value": float(series.std())},
            {"name": "min", "value": float(series.min())},
            {"name": "max", "value": float(series.max())},
            {"name": "median", "value": float(series.median())},
            {"name": "range", "value": float(series.max() - series.min())},
            {"name": "iqr", "value": float(series.quantile(0.75) - series.quantile(0.25))},
            {"name": "skewness", "value": float(series.skew())},
            {"name": "kurtosis", "value": float(series.kurtosis())},
            {"name": "count", "value": float(len(series))},
            {"name": "sum", "value": float(series.sum())}
        ]
        
        # Add autocorrelation if available
        if STATSMODELS_AVAILABLE:
            try:
                acf = sm.tsa.acf(series, nlags=10)
                for i, val in enumerate(acf):
                    features.append({
                        "name": f"autocorrelation_lag_{i}",
                        "value": float(val)
                    })
            except:
                pass
        
        return {
            "status": "success",
            "analysis_type": "feature_extraction",
            "field": field,
            "features": features,
            "count": len(features),
            "note": "Basic feature extraction (tsfresh not available)"
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
            "cache_size": len(self.analysis_cache),
            "max_cache_size": self.max_cache_size,
            "statsmodels_available": STATSMODELS_AVAILABLE,
            "tsfresh_available": TSFRESH_AVAILABLE
        }
