import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
from ..analytics.edge_integration import EdgeIntegrationManager

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager with edge computing capabilities."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize edge integration
        self.edge_manager = EdgeIntegrationManager(config)
        
        # Initialize risk metrics
        self.risk_metrics = {
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'volatility': 0.0,
            'beta': 0.0,
            'correlation': 0.0,
            'drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'recovery_factor': 0.0
        }
        
        # Initialize position limits
        self.position_limits = {
            'max_position_size': config.get('max_position_size', 100000),
            'max_leverage': config.get('max_leverage', 2.0),
            'max_drawdown': config.get('max_drawdown', 0.1),
            'max_correlation': config.get('max_correlation', 0.7),
            'min_liquidity': config.get('min_liquidity', 1000000)
        }
        
        # Initialize risk thresholds
        self.risk_thresholds = {
            'var_threshold': config.get('var_threshold', 0.02),
            'cvar_threshold': config.get('cvar_threshold', 0.03),
            'volatility_threshold': config.get('volatility_threshold', 0.2),
            'drawdown_threshold': config.get('drawdown_threshold', 0.1)
        }
        
        self.logger.info("Risk Manager initialized successfully")
    
    def analyze_risk(self, market_data: Dict[str, Any], positions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk using edge computing capabilities."""
        try:
            # Process market data through edge integration
            processed_data = self.edge_manager.process_market_data(market_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(processed_data, positions)
            
            # Get ML-based risk predictions
            ml_risk = self._get_ml_risk_predictions(processed_data)
            
            # Analyze market conditions
            market_risk = self._analyze_market_conditions(processed_data)
            
            # Combine risk analyses
            risk_analysis = {
                'risk_metrics': risk_metrics,
                'ml_predictions': ml_risk,
                'market_conditions': market_risk,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update risk metrics
            self.risk_metrics.update(risk_metrics)
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk: {str(e)}")
            raise
    
    def validate_trade(self, trade: Dict[str, Any], risk_analysis: Dict[str, Any]) -> bool:
        """Validate trade against risk limits."""
        try:
            # Check position limits
            if not self._check_position_limits(trade):
                return False
            
            # Check risk thresholds
            if not self._check_risk_thresholds(risk_analysis):
                return False
            
            # Check market conditions
            if not self._check_market_conditions(risk_analysis['market_conditions']):
                return False
            
            # Check ML predictions
            if not self._check_ml_predictions(risk_analysis['ml_predictions']):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            raise
    
    def optimize_position_size(self, trade: Dict[str, Any], risk_analysis: Dict[str, Any]) -> float:
        """Optimize position size based on risk analysis."""
        try:
            # Get base position size
            base_size = trade.get('position_size', 0)
            
            # Calculate risk-adjusted size
            risk_adjusted_size = self._calculate_risk_adjusted_size(
                base_size,
                risk_analysis
            )
            
            # Apply position limits
            final_size = self._apply_position_limits(risk_adjusted_size)
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error optimizing position size: {str(e)}")
            raise
    
    def _calculate_risk_metrics(self, market_data: Dict[str, Any], positions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics using edge computing."""
        try:
            # Prepare data
            returns = self._calculate_returns(market_data)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(returns, 0.95)
            var_99, cvar_99 = self._calculate_var_cvar(returns, 0.99)
            
            # Calculate volatility
            volatility = self._calculate_volatility(returns)
            
            # Calculate beta and correlation
            beta, correlation = self._calculate_beta_correlation(returns, market_data)
            
            # Calculate drawdown metrics
            drawdown, max_drawdown, recovery_factor = self._calculate_drawdown_metrics(returns)
            
            # Calculate risk-adjusted returns
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'volatility': volatility,
                'beta': beta,
                'correlation': correlation,
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'recovery_factor': recovery_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def _get_ml_risk_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML-based risk predictions."""
        try:
            # Prepare data for ML models
            input_data = self._prepare_ml_input(market_data)
            
            # Get predictions from edge ML models
            tflite_pred = self.edge_manager.tflite_model.predict(input_data)
            pytorch_pred = self.edge_manager.pytorch_mobile.predict(input_data)
            onnx_pred = self.edge_manager.onnx_runtime.predict(input_data)
            
            # Combine predictions
            ensemble_pred = self._ensemble_predictions([tflite_pred, pytorch_pred, onnx_pred])
            
            return {
                'tflite': tflite_pred,
                'pytorch': pytorch_pred,
                'onnx': onnx_pred,
                'ensemble': ensemble_pred
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ML risk predictions: {str(e)}")
            raise
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions for risk assessment."""
        try:
            # Get market metrics
            metrics = market_data.get('metrics', {})
            
            # Analyze liquidity
            liquidity = self._analyze_liquidity(metrics)
            
            # Analyze volatility
            volatility = self._analyze_volatility(metrics)
            
            # Analyze market depth
            depth = self._analyze_market_depth(metrics)
            
            # Analyze spread
            spread = self._analyze_spread(metrics)
            
            return {
                'liquidity': liquidity,
                'volatility': volatility,
                'depth': depth,
                'spread': spread
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            raise
    
    def _check_position_limits(self, trade: Dict[str, Any]) -> bool:
        """Check if trade violates position limits."""
        try:
            # Check position size
            if trade.get('position_size', 0) > self.position_limits['max_position_size']:
                return False
            
            # Check leverage
            if trade.get('leverage', 1.0) > self.position_limits['max_leverage']:
                return False
            
            # Check liquidity
            if trade.get('liquidity', 0) < self.position_limits['min_liquidity']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {str(e)}")
            raise
    
    def _check_risk_thresholds(self, risk_analysis: Dict[str, Any]) -> bool:
        """Check if risk analysis violates thresholds."""
        try:
            # Get risk metrics
            metrics = risk_analysis['risk_metrics']
            
            # Check VaR
            if metrics['var_95'] > self.risk_thresholds['var_threshold']:
                return False
            
            # Check CVaR
            if metrics['cvar_95'] > self.risk_thresholds['cvar_threshold']:
                return False
            
            # Check volatility
            if metrics['volatility'] > self.risk_thresholds['volatility_threshold']:
                return False
            
            # Check drawdown
            if metrics['max_drawdown'] > self.risk_thresholds['drawdown_threshold']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk thresholds: {str(e)}")
            raise
    
    def _check_market_conditions(self, market_conditions: Dict[str, Any]) -> bool:
        """Check if market conditions are suitable for trading."""
        try:
            # Check liquidity
            if market_conditions['liquidity'] < self.position_limits['min_liquidity']:
                return False
            
            # Check spread
            if market_conditions['spread'] > self.config.get('max_spread', 0.02):
                return False
            
            # Check depth
            if market_conditions['depth'] < self.config.get('min_depth', 100000):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            raise
    
    def _check_ml_predictions(self, ml_predictions: Dict[str, Any]) -> bool:
        """Check if ML predictions indicate high risk."""
        try:
            # Get ensemble predictions
            ensemble = ml_predictions['ensemble']
            
            # Check risk score
            if ensemble.get('risk_score', 0) > self.config.get('max_ml_risk_score', 0.8):
                return False
            
            # Check confidence
            if ensemble.get('confidence', 0) < self.config.get('min_ml_confidence', 0.7):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking ML predictions: {str(e)}")
            raise
    
    def _calculate_risk_adjusted_size(self, base_size: float, risk_analysis: Dict[str, Any]) -> float:
        """Calculate risk-adjusted position size."""
        try:
            # Get risk metrics
            metrics = risk_analysis['risk_metrics']
            
            # Calculate risk adjustment factor
            risk_factor = self._calculate_risk_factor(metrics)
            
            # Apply risk adjustment
            adjusted_size = base_size * risk_factor
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted size: {str(e)}")
            raise
    
    def _apply_position_limits(self, size: float) -> float:
        """Apply position limits to size."""
        try:
            # Apply maximum position size limit
            size = min(size, self.position_limits['max_position_size'])
            
            # Apply minimum position size limit
            size = max(size, self.config.get('min_position_size', 0))
            
            return size
            
        except Exception as e:
            self.logger.error(f"Error applying position limits: {str(e)}")
            raise
    
    def _calculate_returns(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Calculate returns from market data."""
        try:
            # Get price data
            prices = market_data.get('prices', [])
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def _calculate_var_cvar(self, returns: np.ndarray, confidence: float) -> tuple:
        """Calculate VaR and CVaR."""
        try:
            # Calculate VaR
            var = np.percentile(returns, (1 - confidence) * 100)
            
            # Calculate CVaR
            cvar = returns[returns <= var].mean()
            
            return var, cvar
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR and CVaR: {str(e)}")
            raise
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate volatility."""
        try:
            # Calculate annualized volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            raise
    
    def _calculate_beta_correlation(self, returns: np.ndarray, market_data: Dict[str, Any]) -> tuple:
        """Calculate beta and correlation."""
        try:
            # Get market returns
            market_returns = market_data.get('market_returns', returns)
            
            # Calculate beta
            beta = np.cov(returns, market_returns)[0,1] / np.var(market_returns)
            
            # Calculate correlation
            correlation = np.corrcoef(returns, market_returns)[0,1]
            
            return beta, correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating beta and correlation: {str(e)}")
            raise
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> tuple:
        """Calculate drawdown metrics."""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max
            
            # Calculate maximum drawdown
            max_drawdown = drawdown.min()
            
            # Calculate recovery factor
            recovery_factor = abs(cum_returns[-1] / max_drawdown)
            
            return drawdown[-1], max_drawdown, recovery_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {str(e)}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Calculate annualized Sharpe ratio
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            raise
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        try:
            # Calculate downside deviation
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns)
            
            # Calculate annualized Sortino ratio
            sortino = np.mean(returns) / downside_std * np.sqrt(252)
            
            return sortino
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            raise
    
    def _calculate_risk_factor(self, metrics: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor."""
        try:
            # Get risk metrics
            var = metrics['var_95']
            volatility = metrics['volatility']
            drawdown = metrics['max_drawdown']
            
            # Calculate risk scores
            var_score = 1 - (var / self.risk_thresholds['var_threshold'])
            volatility_score = 1 - (volatility / self.risk_thresholds['volatility_threshold'])
            drawdown_score = 1 - (abs(drawdown) / self.risk_thresholds['drawdown_threshold'])
            
            # Calculate weighted average
            risk_factor = (
                var_score * 0.4 +
                volatility_score * 0.3 +
                drawdown_score * 0.3
            )
            
            return max(0.1, min(1.0, risk_factor))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk factor: {str(e)}")
            raise
    
    def _prepare_ml_input(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare data for ML models."""
        try:
            # Get relevant features
            features = [
                market_data.get('price', 0),
                market_data.get('volume', 0),
                market_data.get('volatility', 0),
                market_data.get('liquidity', 0),
                market_data.get('spread', 0)
            ]
            
            # Convert to numpy array
            input_data = np.array(features)
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(input_data.reshape(1, -1))
            
            return scaled_data
            
        except Exception as e:
            self.logger.error(f"Error preparing ML input: {str(e)}")
            raise
    
    def _ensemble_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions from different models."""
        try:
            # Get weights from config
            weights = self.config['ml_model_weights']
            
            # Initialize ensemble prediction
            ensemble_pred = {}
            
            # Combine predictions
            for key in predictions[0].keys():
                ensemble_pred[key] = sum(
                    pred[key] * weights[model]
                    for pred, model in zip(predictions, ['tflite', 'pytorch', 'onnx'])
                )
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error in ensemble predictions: {str(e)}")
            raise 