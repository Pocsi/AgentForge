import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

logger = logging.getLogger(__name__)

class Forecaster:
    """Market forecasting using LSTM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("model_type", "lstm")
        self.sequence_length = config.get("sequence_length", 24)
        self.prediction_window = config.get("prediction_window", 7)
        self.update_frequency = config.get("update_frequency", 3600)
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        
        # Initialize model
        self.model = None
    
    def forecast(self, query: str) -> Dict[str, Any]:
        """Generate market forecasts"""
        try:
            # Get historical data
            historical_data = self._get_historical_data()
            
            if historical_data.empty:
                return {}
            
            # Prepare data
            X, y = self._prepare_data(historical_data)
            
            # Train model if needed
            if self.model is None:
                self.model = self._build_model()
                self._train_model(X, y)
            
            # Generate forecast
            forecast = self._generate_forecast(historical_data)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(forecast)
            
            return {
                "forecast": forecast,
                "confidence_intervals": confidence_intervals,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            return {}
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical market data"""
        try:
            # In a real implementation, this would fetch data from a database or API
            # For now, we'll create sample data
            dates = pd.date_range(
                end=datetime.now(),
                periods=self.sequence_length + self.prediction_window,
                freq='H'
            )
            
            data = pd.DataFrame({
                'date': dates,
                'price': np.random.normal(100, 10, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates))
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for model training"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[['price']])
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return np.array([]), np.array([])
    
    def _build_model(self) -> Sequential:
        """Build LSTM model"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        try:
            if self.model is None or X.size == 0 or y.size == 0:
                return
            
            self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
    
    def _generate_forecast(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate price forecasts"""
        try:
            if self.model is None or data.empty:
                return []
            
            # Prepare last sequence
            last_sequence = self.scaler.transform(data[['price']].tail(self.sequence_length))
            last_sequence = last_sequence.reshape((1, self.sequence_length, 1))
            
            # Generate forecast
            forecast = []
            current_sequence = last_sequence.copy()
            
            for _ in range(self.prediction_window):
                # Predict next value
                next_pred = self.model.predict(current_sequence)
                forecast.append(next_pred[0][0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = next_pred[0][0]
            
            # Inverse transform
            forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            
            # Create forecast dates
            last_date = data['date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(hours=1),
                periods=self.prediction_window,
                freq='H'
            )
            
            # Format results
            results = []
            for date, price in zip(forecast_dates, forecast):
                results.append({
                    "timestamp": date.isoformat(),
                    "price": float(price[0])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return []
    
    def _calculate_confidence_intervals(
        self,
        forecast: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate confidence intervals for forecast"""
        try:
            if not forecast:
                return {}
            
            # Calculate standard deviation of historical data
            std = np.std([f["price"] for f in forecast])
            
            # Create confidence intervals
            intervals = {
                "95": [],
                "99": []
            }
            
            for pred in forecast:
                price = pred["price"]
                timestamp = pred["timestamp"]
                
                # 95% confidence interval
                intervals["95"].append({
                    "timestamp": timestamp,
                    "lower": price - 1.96 * std,
                    "upper": price + 1.96 * std
                })
                
                # 99% confidence interval
                intervals["99"].append({
                    "timestamp": timestamp,
                    "lower": price - 2.576 * std,
                    "upper": price + 2.576 * std
                })
            
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return {} 