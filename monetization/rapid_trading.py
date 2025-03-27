import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from integrations.mcp_connector import MCPConnector
from integrations.goat_connector import GOATConnector
from analytics.time_series import TimeSeriesAnalyzer
from analytics.forecasting import Forecaster

logger = logging.getLogger(__name__)

class RapidTrader:
    """High-speed trading system for quick profits"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp = MCPConnector(config.get("mcp", {}))
        self.goat = GOATConnector(config.get("goat", {}))
        self.time_series = TimeSeriesAnalyzer(config.get("analytics", {}))
        self.forecaster = Forecaster(config.get("forecasting", {}))
        
        # Aggressive trading parameters
        self.min_profit_threshold = 0.005  # 0.5%
        self.max_position_size = 0.2  # ETH
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.05  # 5%
        self.max_trades_per_hour = 10
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {}
        self.hourly_trades = 0
        self.last_hour_reset = datetime.now()
    
    async def start_rapid_trading(self) -> None:
        """Start the rapid trading system"""
        while True:
            try:
                # Reset hourly trade counter if needed
                self._reset_hourly_counter()
                
                # Get high-frequency market data
                market_data = await self._get_high_freq_data()
                
                # Find immediate opportunities
                opportunities = await self._find_immediate_opportunities(market_data)
                
                # Execute trades quickly
                for opportunity in opportunities:
                    if self.hourly_trades < self.max_trades_per_hour:
                        await self._execute_rapid_trade(opportunity)
                        self.hourly_trades += 1
                
                # Update metrics
                await self._update_metrics()
                
                # Very short wait between iterations
                await asyncio.sleep(5)  # 5 seconds
                
            except Exception as e:
                logger.error(f"Error in rapid trading loop: {str(e)}")
                await asyncio.sleep(1)
    
    def _reset_hourly_counter(self) -> None:
        """Reset hourly trade counter if needed"""
        now = datetime.now()
        if (now - self.last_hour_reset).total_seconds() >= 3600:
            self.hourly_trades = 0
            self.last_hour_reset = now
    
    async def _get_high_freq_data(self) -> Dict[str, Any]:
        """Get high-frequency market data"""
        try:
            # Get real-time price data
            price_data = self.goat.execute_tool(
                "insights",
                {"type": "realtime_prices", "pairs": ["ETH/USDT", "ETH/USDC"]}
            )
            
            # Get order book data
            order_book = self.goat.execute_tool(
                "insights",
                {"type": "order_book", "depth": 5}
            )
            
            # Get recent trades
            recent_trades = self.goat.execute_tool(
                "insights",
                {"type": "recent_trades", "limit": 100}
            )
            
            return {
                "prices": price_data.get("result", {}),
                "order_book": order_book.get("result", {}),
                "recent_trades": recent_trades.get("result", [])
            }
        except Exception as e:
            logger.error(f"Error getting high-frequency data: {str(e)}")
            return {}
    
    async def _find_immediate_opportunities(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find immediate trading opportunities"""
        opportunities = []
        
        try:
            # Analyze price differences
            price_opportunities = self._analyze_price_differences(market_data)
            opportunities.extend(price_opportunities)
            
            # Analyze order book imbalances
            order_book_opportunities = self._analyze_order_book(market_data)
            opportunities.extend(order_book_opportunities)
            
            # Analyze recent trade patterns
            pattern_opportunities = self._analyze_trade_patterns(market_data)
            opportunities.extend(pattern_opportunities)
            
            # Filter and sort opportunities by potential profit
            opportunities.sort(key=lambda x: x.get("potential_profit", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding opportunities: {str(e)}")
        
        return opportunities
    
    def _analyze_price_differences(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze price differences for arbitrage"""
        opportunities = []
        try:
            prices = market_data.get("prices", {})
            for pair1, price1 in prices.items():
                for pair2, price2 in prices.items():
                    if pair1 != pair2:
                        diff = abs(price1 - price2) / price1
                        if diff > self.min_profit_threshold:
                            opportunities.append({
                                "type": "arbitrage",
                                "pair1": pair1,
                                "pair2": pair2,
                                "price1": price1,
                                "price2": price2,
                                "potential_profit": diff,
                                "action": "buy" if price1 < price2 else "sell"
                            })
        except Exception as e:
            logger.error(f"Error analyzing price differences: {str(e)}")
        return opportunities
    
    def _analyze_order_book(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze order book for opportunities"""
        opportunities = []
        try:
            order_book = market_data.get("order_book", {})
            for pair, book in order_book.items():
                # Calculate buy/sell pressure
                buy_pressure = sum(bid[1] for bid in book.get("bids", []))
                sell_pressure = sum(ask[1] for ask in book.get("asks", []))
                
                # Find imbalances
                if buy_pressure > sell_pressure * 1.5:
                    opportunities.append({
                        "type": "order_book",
                        "pair": pair,
                        "action": "buy",
                        "pressure_ratio": buy_pressure / sell_pressure,
                        "potential_profit": 0.01  # 1% potential
                    })
                elif sell_pressure > buy_pressure * 1.5:
                    opportunities.append({
                        "type": "order_book",
                        "pair": pair,
                        "action": "sell",
                        "pressure_ratio": sell_pressure / buy_pressure,
                        "potential_profit": 0.01  # 1% potential
                    })
        except Exception as e:
            logger.error(f"Error analyzing order book: {str(e)}")
        return opportunities
    
    def _analyze_trade_patterns(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze recent trades for patterns"""
        opportunities = []
        try:
            trades = market_data.get("recent_trades", [])
            if len(trades) >= 10:
                # Calculate average trade size
                avg_size = sum(trade.get("size", 0) for trade in trades) / len(trades)
                
                # Look for large trades
                large_trades = [t for t in trades if t.get("size", 0) > avg_size * 2]
                
                if large_trades:
                    # Analyze direction of large trades
                    buy_count = sum(1 for t in large_trades if t.get("side") == "buy")
                    sell_count = sum(1 for t in large_trades if t.get("side") == "sell")
                    
                    if buy_count > sell_count:
                        opportunities.append({
                            "type": "pattern",
                            "action": "buy",
                            "confidence": buy_count / len(large_trades),
                            "potential_profit": 0.015  # 1.5% potential
                        })
                    elif sell_count > buy_count:
                        opportunities.append({
                            "type": "pattern",
                            "action": "sell",
                            "confidence": sell_count / len(large_trades),
                            "potential_profit": 0.015  # 1.5% potential
                        })
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {str(e)}")
        return opportunities
    
    async def _execute_rapid_trade(self, opportunity: Dict[str, Any]) -> None:
        """Execute a rapid trade"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(opportunity)
            
            # Execute trade through GOAT
            result = self.goat.execute_tool(
                "payments",
                {
                    "type": "trade",
                    "opportunity": opportunity,
                    "position_size": position_size,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                    "priority": "high"
                }
            )
            
            if result.get("status") == "success":
                self.trades.append({
                    "timestamp": datetime.now().isoformat(),
                    "opportunity": opportunity,
                    "result": result
                })
                
        except Exception as e:
            logger.error(f"Error executing rapid trade: {str(e)}")
    
    def _calculate_position_size(self, opportunity: Dict[str, Any]) -> float:
        """Calculate position size for a rapid trade"""
        try:
            # Get current portfolio value
            portfolio = self.goat.execute_tool(
                "insights",
                {"type": "portfolio_value"}
            )
            
            if portfolio.get("status") != "success":
                return 0.0
            
            # Calculate position size based on opportunity type
            value = float(portfolio.get("result", 0))
            if opportunity["type"] == "arbitrage":
                return min(value * 0.3, self.max_position_size)  # 30% for arbitrage
            elif opportunity["type"] == "order_book":
                return min(value * 0.2, self.max_position_size)  # 20% for order book
            else:  # pattern
                return min(value * 0.15, self.max_position_size)  # 15% for patterns
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    async def _update_metrics(self) -> None:
        """Update performance metrics"""
        try:
            # Get current portfolio value
            portfolio = self.goat.execute_tool(
                "insights",
                {"type": "portfolio_value"}
            )
            
            # Calculate ROI
            if portfolio.get("status") == "success":
                initial_value = float(self.config.get("initial_investment", 0))
                current_value = float(portfolio.get("result", 0))
                
                self.performance_metrics = {
                    "current_value": current_value,
                    "roi": (current_value - initial_value) / initial_value,
                    "total_trades": len(self.trades),
                    "hourly_trades": self.hourly_trades,
                    "last_update": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}") 