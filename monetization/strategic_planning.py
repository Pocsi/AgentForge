import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from analytics.time_series import TimeSeriesAnalyzer
from analytics.forecasting import Forecaster
from integrations.mcp_connector import MCPConnector
from integrations.goat_connector import GOATConnector

logger = logging.getLogger(__name__)

class StrategicPlanner:
    """Strategic planning system using foresight and time series analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp = MCPConnector(config.get("mcp", {}))
        self.goat = GOATConnector(config.get("goat", {}))
        self.time_series = TimeSeriesAnalyzer(config.get("analytics", {}))
        self.forecaster = Forecaster(config.get("forecasting", {}))
        
        # Goal parameters
        self.target_amount = 1000  # PC cost
        self.initial_capital = 3    # Starting amount
        self.timeframe = 30         # Days
        
        # Analysis results
        self.opportunities = []
        self.risks = []
        self.recommendations = []
    
    async def analyze_opportunities(self) -> Dict[str, Any]:
        """Analyze various opportunities using AI and time series"""
        try:
            # Get market trends
            market_trends = await self._analyze_market_trends()
            
            # Get skill demand analysis
            skill_demand = await self._analyze_skill_demand()
            
            # Get gig economy opportunities
            gig_opportunities = await self._analyze_gig_opportunities()
            
            # Get digital product trends
            product_trends = await self._analyze_product_trends()
            
            # Combine and analyze all data
            analysis = self._combine_analysis(
                market_trends,
                skill_demand,
                gig_opportunities,
                product_trends
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            
            return {
                "analysis": analysis,
                "recommendations": recommendations,
                "timeline": self._generate_timeline(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing opportunities: {str(e)}")
            return {}
    
    async def _analyze_market_trends(self) -> Dict[str, Any]:
        """Analyze current market trends"""
        try:
            # Get market data from GOAT
            market_data = self.goat.execute_tool(
                "insights",
                {"type": "market_trends", "timeframe": "30d"}
            )
            
            # Analyze trends using time series
            trend_analysis = self.time_series.analyze(
                "Analyze market trends and identify opportunities"
            )
            
            # Get future predictions
            forecast = self.forecaster.forecast(
                "Predict market trends for next 30 days"
            )
            
            return {
                "current_trends": market_data.get("result", {}),
                "analysis": trend_analysis,
                "forecast": forecast
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {str(e)}")
            return {}
    
    async def _analyze_skill_demand(self) -> Dict[str, Any]:
        """Analyze in-demand skills"""
        try:
            # Get skill demand data
            skill_data = self.mcp.query(
                "Analyze current in-demand skills and their market value"
            )
            
            # Get learning opportunities
            learning_data = self.mcp.query(
                "Find available learning resources and their costs"
            )
            
            return {
                "in_demand_skills": skill_data.get("result", []),
                "learning_opportunities": learning_data.get("result", [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill demand: {str(e)}")
            return {}
    
    async def _analyze_gig_opportunities(self) -> Dict[str, Any]:
        """Analyze gig economy opportunities"""
        try:
            # Get gig platform data
            gig_data = self.mcp.query(
                "Analyze current gig economy opportunities and earnings potential"
            )
            
            # Get platform comparisons
            platform_data = self.mcp.query(
                "Compare different gig platforms and their earning potential"
            )
            
            return {
                "opportunities": gig_data.get("result", []),
                "platform_comparison": platform_data.get("result", [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gig opportunities: {str(e)}")
            return {}
    
    async def _analyze_product_trends(self) -> Dict[str, Any]:
        """Analyze digital product trends"""
        try:
            # Get product trend data
            product_data = self.mcp.query(
                "Analyze current digital product trends and market opportunities"
            )
            
            # Get competition analysis
            competition_data = self.mcp.query(
                "Analyze competition in digital product markets"
            )
            
            return {
                "trends": product_data.get("result", []),
                "competition": competition_data.get("result", [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing product trends: {str(e)}")
            return {}
    
    def _combine_analysis(
        self,
        market_trends: Dict[str, Any],
        skill_demand: Dict[str, Any],
        gig_opportunities: Dict[str, Any],
        product_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all analysis results"""
        return {
            "market_trends": market_trends,
            "skill_demand": skill_demand,
            "gig_opportunities": gig_opportunities,
            "product_trends": product_trends,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        try:
            # Analyze skill-based opportunities
            skills = analysis.get("skill_demand", {}).get("in_demand_skills", [])
            for skill in skills:
                recommendations.append({
                    "type": "skill_development",
                    "skill": skill.get("name"),
                    "earning_potential": skill.get("earning_potential"),
                    "learning_cost": skill.get("learning_cost"),
                    "time_to_proficiency": skill.get("time_to_proficiency"),
                    "risk_level": "low"
                })
            
            # Analyze gig opportunities
            gigs = analysis.get("gig_opportunities", {}).get("opportunities", [])
            for gig in gigs:
                recommendations.append({
                    "type": "gig_work",
                    "platform": gig.get("platform"),
                    "earning_potential": gig.get("earning_potential"),
                    "startup_cost": gig.get("startup_cost"),
                    "time_to_first_earnings": gig.get("time_to_first_earnings"),
                    "risk_level": "medium"
                })
            
            # Analyze product opportunities
            products = analysis.get("product_trends", {}).get("trends", [])
            for product in products:
                recommendations.append({
                    "type": "digital_product",
                    "category": product.get("category"),
                    "earning_potential": product.get("earning_potential"),
                    "development_cost": product.get("development_cost"),
                    "time_to_market": product.get("time_to_market"),
                    "risk_level": "high"
                })
            
            # Sort by earning potential
            recommendations.sort(
                key=lambda x: x.get("earning_potential", 0),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_timeline(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate timeline for achieving goal"""
        timeline = {
            "phases": [],
            "total_duration": 0,
            "expected_earnings": 0
        }
        
        try:
            current_capital = self.initial_capital
            current_time = 0
            
            for rec in recommendations:
                phase = {
                    "type": rec["type"],
                    "duration": rec.get("time_to_first_earnings", 0),
                    "cost": rec.get("learning_cost", 0) or rec.get("startup_cost", 0),
                    "expected_earnings": rec.get("earning_potential", 0),
                    "risk_level": rec.get("risk_level", "medium")
                }
                
                # Check if we can afford this phase
                if current_capital >= phase["cost"]:
                    timeline["phases"].append(phase)
                    current_capital -= phase["cost"]
                    current_time += phase["duration"]
                    timeline["expected_earnings"] += phase["expected_earnings"]
            
            timeline["total_duration"] = current_time
            
        except Exception as e:
            logger.error(f"Error generating timeline: {str(e)}")
        
        return timeline 