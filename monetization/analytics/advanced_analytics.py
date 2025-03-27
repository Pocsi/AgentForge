import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from prophet import Prophet

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics with machine learning and predictive modeling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        self.predictions = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize machine learning models"""
        try:
            # Random Forest for price prediction
            self.models["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Gradient Boosting for trend prediction
            self.models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            # XGBoost for volatility prediction
            self.models["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # LSTM for sequence prediction
            self.models["lstm"] = self._build_lstm_model()
            
            # Prophet for time series forecasting
            self.models["prophet"] = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model for sequence prediction"""
        try:
            model = models.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(50, 10)),
                layers.Dropout(0.2),
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            return None
    
    async def analyze_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data using multiple models"""
        try:
            # Prepare data
            X, y = self._prepare_data(data)
            
            # Train models
            await self._train_models(X, y)
            
            # Generate predictions
            predictions = await self._generate_predictions(X)
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(X, y)
            
            # Analyze market trends
            trends = self._analyze_trends(data)
            
            return {
                "predictions": predictions,
                "feature_importance": importance,
                "trends": trends,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market data: {str(e)}")
            return {}
    
    def _prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for model training"""
        try:
            # Create features
            features = pd.DataFrame()
            
            # Price-based features
            features["returns"] = data["price"].pct_change()
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["momentum"] = data["price"].pct_change(periods=10)
            
            # Volume-based features
            features["volume_ma"] = data["volume"].rolling(window=20).mean()
            features["volume_std"] = data["volume"].rolling(window=20).std()
            features["volume_ratio"] = data["volume"] / features["volume_ma"]
            
            # Technical indicators
            features["rsi"] = self._calculate_rsi(data["price"])
            features["macd"] = self._calculate_macd(data["price"])
            features["bollinger_bands"] = self._calculate_bollinger_bands(data["price"])
            
            # Market sentiment features
            features["sentiment"] = self._calculate_sentiment(data)
            
            # Remove NaN values
            features = features.dropna()
            
            # Scale features
            X = self.scaler.fit_transform(features)
            y = data["price"].values[features.index]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return np.array([]), np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            return exp1 - exp2
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands"""
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            return (prices - ma) / (2 * std)
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series()
    
    def _calculate_sentiment(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market sentiment"""
        try:
            # This would use NLP to analyze market sentiment
            # For now, we'll use a simple heuristic
            sentiment = pd.Series(index=data.index)
            sentiment[data["price"].diff() > 0] = 1
            sentiment[data["price"].diff() < 0] = -1
            sentiment = sentiment.fillna(0)
            return sentiment
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return pd.Series()
    
    async def _train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train machine learning models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            self.models["random_forest"].fit(X_train, y_train)
            
            # Train Gradient Boosting
            self.models["gradient_boosting"].fit(X_train, y_train)
            
            # Train XGBoost
            self.models["xgboost"].fit(X_train, y_train)
            
            # Train LSTM
            X_lstm = X.reshape((X.shape[0], 50, 10))
            self.models["lstm"].fit(
                X_lstm, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Train Prophet
            prophet_data = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(y)),
                'y': y
            })
            self.models["prophet"].fit(prophet_data)
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    async def _generate_predictions(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using all models"""
        try:
            predictions = {}
            
            # Random Forest predictions
            predictions["random_forest"] = self.models["random_forest"].predict(X)
            
            # Gradient Boosting predictions
            predictions["gradient_boosting"] = self.models["gradient_boosting"].predict(X)
            
            # XGBoost predictions
            predictions["xgboost"] = self.models["xgboost"].predict(X)
            
            # LSTM predictions
            X_lstm = X.reshape((X.shape[0], 50, 10))
            predictions["lstm"] = self.models["lstm"].predict(X_lstm).flatten()
            
            # Prophet predictions
            future = self.models["prophet"].make_future_dataframe(periods=30)
            prophet_forecast = self.models["prophet"].predict(future)
            predictions["prophet"] = prophet_forecast["yhat"].values[-len(X):]
            
            # Ensemble predictions
            predictions["ensemble"] = np.mean([
                predictions["random_forest"],
                predictions["gradient_boosting"],
                predictions["xgboost"],
                predictions["lstm"],
                predictions["prophet"]
            ], axis=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {}
    
    def _calculate_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            importance = {}
            
            # Random Forest feature importance
            rf_importance = self.models["random_forest"].feature_importances_
            
            # Gradient Boosting feature importance
            gb_importance = self.models["gradient_boosting"].feature_importances_
            
            # XGBoost feature importance
            xgb_importance = self.models["xgboost"].feature_importances_
            
            # Average importance across models
            for i in range(X.shape[1]):
                importance[f"feature_{i}"] = np.mean([
                    rf_importance[i],
                    gb_importance[i],
                    xgb_importance[i]
                ])
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market trends"""
        try:
            trends = {
                "price_trend": self._analyze_price_trend(data),
                "volume_trend": self._analyze_volume_trend(data),
                "volatility_trend": self._analyze_volatility_trend(data),
                "momentum_trend": self._analyze_momentum_trend(data)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {}
    
    def _analyze_price_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend"""
        try:
            # Calculate moving averages
            ma20 = data["price"].rolling(window=20).mean()
            ma50 = data["price"].rolling(window=50).mean()
            
            # Determine trend
            current_price = data["price"].iloc[-1]
            ma20_current = ma20.iloc[-1]
            ma50_current = ma50.iloc[-1]
            
            trend = {
                "direction": "up" if current_price > ma20_current > ma50_current else "down",
                "strength": abs(current_price - ma50_current) / ma50_current,
                "ma20": ma20_current,
                "ma50": ma50_current
            }
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing price trend: {str(e)}")
            return {}
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trend"""
        try:
            # Calculate volume moving average
            volume_ma = data["volume"].rolling(window=20).mean()
            
            # Calculate volume trend
            current_volume = data["volume"].iloc[-1]
            volume_ma_current = volume_ma.iloc[-1]
            
            trend = {
                "direction": "up" if current_volume > volume_ma_current else "down",
                "strength": abs(current_volume - volume_ma_current) / volume_ma_current,
                "volume_ma": volume_ma_current
            }
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {str(e)}")
            return {}
    
    def _analyze_volatility_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility trend"""
        try:
            # Calculate returns
            returns = data["price"].pct_change()
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std()
            
            # Calculate volatility trend
            current_volatility = volatility.iloc[-1]
            volatility_ma = volatility.rolling(window=50).mean().iloc[-1]
            
            trend = {
                "direction": "up" if current_volatility > volatility_ma else "down",
                "strength": abs(current_volatility - volatility_ma) / volatility_ma,
                "volatility_ma": volatility_ma
            }
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing volatility trend: {str(e)}")
            return {}
    
    def _analyze_momentum_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum trend"""
        try:
            # Calculate momentum
            momentum = data["price"].pct_change(periods=10)
            
            # Calculate momentum trend
            current_momentum = momentum.iloc[-1]
            momentum_ma = momentum.rolling(window=20).mean().iloc[-1]
            
            trend = {
                "direction": "up" if current_momentum > momentum_ma else "down",
                "strength": abs(current_momentum - momentum_ma) / abs(momentum_ma),
                "momentum_ma": momentum_ma
            }
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing momentum trend: {str(e)}")
            return {} 