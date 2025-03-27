import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from ..analytics.edge_integration import EdgeIntegrationManager
from ..analytics.risk_manager import RiskManager
from ..analytics.network_analysis import NetworkAnalyzer
from ..analytics.advanced_analytics import AdvancedAnalytics

class TradingStrategies:
    def __init__(self, config: Dict[str, Any]):
        """Initialize trading strategies with edge computing capabilities."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.edge_manager = EdgeIntegrationManager(config)
        self.risk_manager = RiskManager(config)
        self.network_analyzer = NetworkAnalyzer(config)
        self.advanced_analytics = AdvancedAnalytics(config)
        
        # Initialize strategy parameters
        self.strategy_params = {
            'statistical_arbitrage': {
                'lookback_period': config.get('stat_arb_lookback', 60),
                'entry_threshold': config.get('stat_arb_entry', 2.0),
                'exit_threshold': config.get('stat_arb_exit', 1.0),
                'max_positions': config.get('stat_arb_max_positions', 5)
            },
            'market_making': {
                'spread_multiplier': config.get('mm_spread_mult', 1.5),
                'min_spread': config.get('mm_min_spread', 0.001),
                'max_position': config.get('mm_max_position', 1000),
                'inventory_target': config.get('mm_inventory_target', 0)
            },
            'momentum': {
                'lookback_period': config.get('momentum_lookback', 20),
                'entry_threshold': config.get('momentum_entry', 0.02),
                'exit_threshold': config.get('momentum_exit', 0.01),
                'max_holdings': config.get('momentum_max_holdings', 3)
            },
            'mean_reversion': {
                'lookback_period': config.get('mean_rev_lookback', 30),
                'entry_threshold': config.get('mean_rev_entry', 2.0),
                'exit_threshold': config.get('mean_rev_exit', 1.0),
                'max_positions': config.get('mean_rev_max_positions', 4)
            }
        }
        
        self.logger.info("Trading Strategies initialized successfully")
    
    def execute_statistical_arbitrage(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical arbitrage strategy."""
        try:
            # Process market data through edge integration
            processed_data = self.edge_manager.process_market_data(market_data)
            
            # Get network analysis for optimal execution
            network_analysis = self.network_analyzer.analyze_topology(processed_data)
            
            # Find arbitrage opportunities
            opportunities = self._find_arbitrage_opportunities(processed_data)
            
            # Filter opportunities based on risk
            valid_opportunities = self._filter_opportunities(opportunities)
            
            # Execute trades
            trades = self._execute_arbitrage_trades(valid_opportunities, network_analysis)
            
            return {
                'strategy': 'statistical_arbitrage',
                'opportunities': valid_opportunities,
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing statistical arbitrage: {str(e)}")
            raise
    
    def execute_market_making(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market making strategy."""
        try:
            # Process market data
            processed_data = self.edge_manager.process_market_data(market_data)
            
            # Get network analysis
            network_analysis = self.network_analyzer.analyze_topology(processed_data)
            
            # Calculate optimal quotes
            quotes = self._calculate_optimal_quotes(processed_data)
            
            # Adjust quotes based on inventory
            adjusted_quotes = self._adjust_quotes_for_inventory(quotes, processed_data)
            
            # Execute market making trades
            trades = self._execute_market_making_trades(adjusted_quotes, network_analysis)
            
            return {
                'strategy': 'market_making',
                'quotes': adjusted_quotes,
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing market making: {str(e)}")
            raise
    
    def execute_momentum_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute momentum trading strategy."""
        try:
            # Process market data
            processed_data = self.edge_manager.process_market_data(market_data)
            
            # Get network analysis
            network_analysis = self.network_analyzer.analyze_topology(processed_data)
            
            # Identify momentum signals
            signals = self._identify_momentum_signals(processed_data)
            
            # Filter signals based on risk
            valid_signals = self._filter_signals(signals)
            
            # Execute momentum trades
            trades = self._execute_momentum_trades(valid_signals, network_analysis)
            
            return {
                'strategy': 'momentum',
                'signals': valid_signals,
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing momentum strategy: {str(e)}")
            raise
    
    def execute_mean_reversion(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mean reversion strategy."""
        try:
            # Process market data
            processed_data = self.edge_manager.process_market_data(market_data)
            
            # Get network analysis
            network_analysis = self.network_analyzer.analyze_topology(processed_data)
            
            # Identify mean reversion opportunities
            opportunities = self._identify_mean_reversion_opportunities(processed_data)
            
            # Filter opportunities based on risk
            valid_opportunities = self._filter_opportunities(opportunities)
            
            # Execute mean reversion trades
            trades = self._execute_mean_reversion_trades(valid_opportunities, network_analysis)
            
            return {
                'strategy': 'mean_reversion',
                'opportunities': valid_opportunities,
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing mean reversion: {str(e)}")
            raise
    
    def _find_arbitrage_opportunities(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find statistical arbitrage opportunities."""
        try:
            # Get price data
            prices = market_data.get('prices', {})
            
            # Calculate price differences
            price_diffs = self._calculate_price_differences(prices)
            
            # Calculate z-scores
            z_scores = self._calculate_z_scores(price_diffs)
            
            # Find opportunities
            opportunities = []
            for symbol, z_score in z_scores.items():
                if abs(z_score) > self.strategy_params['statistical_arbitrage']['entry_threshold']:
                    opportunities.append({
                        'symbol': symbol,
                        'z_score': z_score,
                        'direction': 'short' if z_score > 0 else 'long',
                        'confidence': self._calculate_confidence(z_score)
                    })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {str(e)}")
            raise
    
    def _calculate_optimal_quotes(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal market making quotes."""
        try:
            # Get order book data
            order_book = market_data.get('order_book', {})
            
            # Calculate mid price
            mid_price = self._calculate_mid_price(order_book)
            
            # Calculate spread
            spread = self._calculate_spread(order_book)
            
            # Calculate optimal quotes
            quotes = {}
            for symbol in order_book.keys():
                quotes[symbol] = {
                    'bid': mid_price[symbol] - spread[symbol] * self.strategy_params['market_making']['spread_multiplier'],
                    'ask': mid_price[symbol] + spread[symbol] * self.strategy_params['market_making']['spread_multiplier'],
                    'size': self._calculate_quote_size(order_book[symbol])
                }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal quotes: {str(e)}")
            raise
    
    def _identify_momentum_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify momentum trading signals."""
        try:
            # Get price data
            prices = market_data.get('prices', {})
            
            # Calculate returns
            returns = self._calculate_returns(prices)
            
            # Calculate momentum indicators
            momentum = self._calculate_momentum(returns)
            
            # Identify signals
            signals = []
            for symbol, mom in momentum.items():
                if abs(mom) > self.strategy_params['momentum']['entry_threshold']:
                    signals.append({
                        'symbol': symbol,
                        'momentum': mom,
                        'direction': 'long' if mom > 0 else 'short',
                        'strength': self._calculate_signal_strength(mom)
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error identifying momentum signals: {str(e)}")
            raise
    
    def _identify_mean_reversion_opportunities(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify mean reversion opportunities."""
        try:
            # Get price data
            prices = market_data.get('prices', {})
            
            # Calculate moving averages
            ma = self._calculate_moving_averages(prices)
            
            # Calculate deviations
            deviations = self._calculate_deviations(prices, ma)
            
            # Find opportunities
            opportunities = []
            for symbol, dev in deviations.items():
                if abs(dev) > self.strategy_params['mean_reversion']['entry_threshold']:
                    opportunities.append({
                        'symbol': symbol,
                        'deviation': dev,
                        'direction': 'long' if dev < 0 else 'short',
                        'confidence': self._calculate_confidence(dev)
                    })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying mean reversion opportunities: {str(e)}")
            raise
    
    def _filter_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter opportunities based on risk analysis."""
        try:
            filtered = []
            for opp in opportunities:
                # Get risk analysis
                risk_analysis = self.risk_manager.analyze_risk(
                    {'opportunity': opp},
                    {'current_positions': {}}
                )
                
                # Validate opportunity
                if self.risk_manager.validate_trade(opp, risk_analysis):
                    filtered.append(opp)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering opportunities: {str(e)}")
            raise
    
    def _filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter trading signals based on risk analysis."""
        try:
            filtered = []
            for signal in signals:
                # Get risk analysis
                risk_analysis = self.risk_manager.analyze_risk(
                    {'signal': signal},
                    {'current_positions': {}}
                )
                
                # Validate signal
                if self.risk_manager.validate_trade(signal, risk_analysis):
                    filtered.append(signal)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {str(e)}")
            raise
    
    def _execute_arbitrage_trades(self, opportunities: List[Dict[str, Any]], network_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute statistical arbitrage trades."""
        try:
            trades = []
            for opp in opportunities:
                # Optimize position size
                size = self.risk_manager.optimize_position_size(opp, {'risk_metrics': {}})
                
                # Execute trade through optimal network path
                trade = self._execute_trade(opp, size, network_analysis)
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage trades: {str(e)}")
            raise
    
    def _execute_market_making_trades(self, quotes: Dict[str, Any], network_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute market making trades."""
        try:
            trades = []
            for symbol, quote in quotes.items():
                # Create market making orders
                orders = self._create_market_making_orders(quote)
                
                # Execute orders through optimal network path
                for order in orders:
                    trade = self._execute_trade(order, quote['size'], network_analysis)
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing market making trades: {str(e)}")
            raise
    
    def _execute_momentum_trades(self, signals: List[Dict[str, Any]], network_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute momentum trades."""
        try:
            trades = []
            for signal in signals:
                # Optimize position size
                size = self.risk_manager.optimize_position_size(signal, {'risk_metrics': {}})
                
                # Execute trade through optimal network path
                trade = self._execute_trade(signal, size, network_analysis)
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing momentum trades: {str(e)}")
            raise
    
    def _execute_mean_reversion_trades(self, opportunities: List[Dict[str, Any]], network_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute mean reversion trades."""
        try:
            trades = []
            for opp in opportunities:
                # Optimize position size
                size = self.risk_manager.optimize_position_size(opp, {'risk_metrics': {}})
                
                # Execute trade through optimal network path
                trade = self._execute_trade(opp, size, network_analysis)
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing mean reversion trades: {str(e)}")
            raise
    
    def _execute_trade(self, trade: Dict[str, Any], size: float, network_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade through optimal network path."""
        try:
            # Get optimal network path
            network_path = self.network_analyzer.get_optimal_path(network_analysis)
            
            # Execute trade
            execution = self.edge_manager.execute_trade(trade, size, network_path)
            
            return {
                'trade': trade,
                'size': size,
                'execution': execution,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def _calculate_price_differences(self, prices: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate price differences between correlated assets."""
        try:
            diffs = {}
            for symbol, price_list in prices.items():
                if len(price_list) >= 2:
                    diffs[symbol] = price_list[-1] - price_list[-2]
            return diffs
            
        except Exception as e:
            self.logger.error(f"Error calculating price differences: {str(e)}")
            raise
    
    def _calculate_z_scores(self, price_diffs: Dict[str, float]) -> Dict[str, float]:
        """Calculate z-scores for price differences."""
        try:
            z_scores = {}
            for symbol, diff in price_diffs.items():
                if diff != 0:
                    z_scores[symbol] = diff / np.std([diff])
            return z_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating z-scores: {str(e)}")
            raise
    
    def _calculate_confidence(self, value: float) -> float:
        """Calculate confidence score based on deviation."""
        try:
            # Normalize value to [0, 1] range
            normalized = min(1.0, abs(value) / 3.0)
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            raise
    
    def _calculate_mid_price(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Calculate mid price from order book."""
        try:
            mid_prices = {}
            for symbol, book in order_book.items():
                if book.get('bids') and book.get('asks'):
                    mid_prices[symbol] = (book['bids'][0][0] + book['asks'][0][0]) / 2
            return mid_prices
            
        except Exception as e:
            self.logger.error(f"Error calculating mid price: {str(e)}")
            raise
    
    def _calculate_spread(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Calculate spread from order book."""
        try:
            spreads = {}
            for symbol, book in order_book.items():
                if book.get('bids') and book.get('asks'):
                    spreads[symbol] = (book['asks'][0][0] - book['bids'][0][0]) / book['bids'][0][0]
            return spreads
            
        except Exception as e:
            self.logger.error(f"Error calculating spread: {str(e)}")
            raise
    
    def _calculate_quote_size(self, order_book: Dict[str, Any]) -> float:
        """Calculate optimal quote size."""
        try:
            # Get market depth
            depth = sum(level[1] for level in order_book.get('bids', []))
            
            # Calculate size based on depth
            size = min(
                depth * 0.1,
                self.strategy_params['market_making']['max_position']
            )
            
            return size
            
        except Exception as e:
            self.logger.error(f"Error calculating quote size: {str(e)}")
            raise
    
    def _calculate_returns(self, prices: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate returns from price data."""
        try:
            returns = {}
            for symbol, price_list in prices.items():
                if len(price_list) >= 2:
                    returns[symbol] = (price_list[-1] - price_list[-2]) / price_list[-2]
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def _calculate_momentum(self, returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate momentum indicators."""
        try:
            momentum = {}
            for symbol, ret in returns.items():
                momentum[symbol] = ret * self.strategy_params['momentum']['lookback_period']
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            raise
    
    def _calculate_signal_strength(self, momentum: float) -> float:
        """Calculate signal strength based on momentum."""
        try:
            # Normalize momentum to [0, 1] range
            normalized = min(1.0, abs(momentum) / self.strategy_params['momentum']['entry_threshold'])
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {str(e)}")
            raise
    
    def _calculate_moving_averages(self, prices: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate moving averages."""
        try:
            ma = {}
            for symbol, price_list in prices.items():
                if len(price_list) >= self.strategy_params['mean_reversion']['lookback_period']:
                    ma[symbol] = np.mean(price_list[-self.strategy_params['mean_reversion']['lookback_period']:])
            return ma
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {str(e)}")
            raise
    
    def _calculate_deviations(self, prices: Dict[str, List[float]], ma: Dict[str, float]) -> Dict[str, float]:
        """Calculate deviations from moving average."""
        try:
            deviations = {}
            for symbol, price_list in prices.items():
                if symbol in ma and price_list:
                    deviations[symbol] = (price_list[-1] - ma[symbol]) / ma[symbol]
            return deviations
            
        except Exception as e:
            self.logger.error(f"Error calculating deviations: {str(e)}")
            raise
    
    def _create_market_making_orders(self, quote: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create market making orders."""
        try:
            orders = [
                {
                    'type': 'limit',
                    'side': 'bid',
                    'price': quote['bid'],
                    'size': quote['size']
                },
                {
                    'type': 'limit',
                    'side': 'ask',
                    'price': quote['ask'],
                    'size': quote['size']
                }
            ]
            return orders
            
        except Exception as e:
            self.logger.error(f"Error creating market making orders: {str(e)}")
            raise 