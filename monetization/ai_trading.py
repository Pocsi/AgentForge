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

class AITrader:
    """AI-powered trading system using MCP and GOAT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp = MCPConnector(config.get("mcp", {}))
        self.goat = GOATConnector(config.get("goat", {}))
        self.time_series = TimeSeriesAnalyzer(config.get("analytics", {}))
        self.forecaster = Forecaster(config.get("forecasting", {}))
        
        # Trading parameters
        self.min_profit_threshold = 0.002  # 0.2%
        self.max_position_size = 0.1  # ETH
        self.stop_loss = 0.05  # 5%
        self.take_profit = 0.1  # 10%
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {}
    
    async def start_trading(self) -> None:
        """Start the AI trading system"""
        while True:
            try:
                # Get market analysis from MCP
                market_analysis = await self._get_market_analysis()
                
                # Get trading opportunities from GOAT
                opportunities = await self._get_trading_opportunities()
                
                # Analyze opportunities with AI
                valid_opportunities = await self._analyze_opportunities(
                    market_analysis,
                    opportunities
                )
                
                # Execute trades
                for opportunity in valid_opportunities:
                    await self._execute_trade(opportunity)
                
                # Update performance metrics
                await self._update_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _get_market_analysis(self) -> Dict[str, Any]:
        """Get market analysis from MCP"""
        try:
            # Get market sentiment
            sentiment = self.mcp.query("Analyze current market sentiment")
            
            # Get technical analysis
            technical = self.mcp.query("Generate technical analysis for current market")
            
            # Get on-chain metrics
            onchain = self.mcp.query("Analyze current on-chain metrics")
            
            return {
                "sentiment": sentiment,
                "technical": technical,
                "onchain": onchain
            }
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return {}
    
    async def _get_trading_opportunities(self) -> List[Dict[str, Any]]:
        """Get trading opportunities from GOAT"""
        try:
            # Get arbitrage opportunities
            arb_opportunities = self.goat.execute_tool(
                "investments",
                {"type": "arbitrage", "min_profit": self.min_profit_threshold}
            )
            
            # Get yield farming opportunities
            yield_opportunities = self.goat.execute_tool(
                "investments",
                {"type": "yield", "min_apy": 0.05}
            )
            
            # Get trading opportunities
            trading_opportunities = self.goat.execute_tool(
                "investments",
                {"type": "trading", "min_roi": 0.02}
            )
            
            return [
                *arb_opportunities.get("result", []),
                *yield_opportunities.get("result", []),
                *trading_opportunities.get("result", [])
            ]
        except Exception as e:
            logger.error(f"Error getting trading opportunities: {str(e)}")
            return []
    
    async def _analyze_opportunities(
        self,
        market_analysis: Dict[str, Any],
        opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze opportunities using AI"""
        valid_opportunities = []
        
        for opportunity in opportunities:
            try:
                # Get AI analysis
                analysis = self.mcp.query(
                    f"Analyze trading opportunity: {opportunity}"
                )
                
                # Check if opportunity is valid
                if self._is_valid_opportunity(analysis, market_analysis):
                    valid_opportunities.append({
                        **opportunity,
                        "analysis": analysis
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing opportunity: {str(e)}")
                continue
        
        return valid_opportunities
    
    async def _execute_trade(self, opportunity: Dict[str, Any]) -> None:
        """Execute a trade"""
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
                    "take_profit": self.take_profit
                }
            )
            
            if result.get("status") == "success":
                self.trades.append({
                    "timestamp": datetime.now().isoformat(),
                    "opportunity": opportunity,
                    "result": result
                })
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
    
    def _is_valid_opportunity(
        self,
        analysis: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> bool:
        """Check if an opportunity is valid"""
        try:
            # Check market conditions
            if market_analysis.get("sentiment", {}).get("status") != "success":
                return False
            
            # Check technical analysis
            if market_analysis.get("technical", {}).get("status") != "success":
                return False
            
            # Check on-chain metrics
            if market_analysis.get("onchain", {}).get("status") != "success":
                return False
            
            # Check opportunity analysis
            if analysis.get("status") != "success":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating opportunity: {str(e)}")
            return False
    
    def _calculate_position_size(self, opportunity: Dict[str, Any]) -> float:
        """Calculate position size for a trade"""
        try:
            # Get current portfolio value
            portfolio = self.goat.execute_tool(
                "insights",
                {"type": "portfolio_value"}
            )
            
            if portfolio.get("status") != "success":
                return 0.0
            
            # Calculate position size based on risk management
            value = float(portfolio.get("result", 0))
            return min(
                value * 0.2,  # 20% of portfolio
                self.max_position_size  # Maximum position size
            )
            
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
                    "last_update": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}") 