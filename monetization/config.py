import os
from typing import Dict, Any
from dotenv import load_dotenv

def get_monetization_config() -> Dict[str, Any]:
    """Get configuration for monetization system"""
    # Load environment variables
    load_dotenv()
    
    return {
        "mcp": {
            "api_key": os.getenv("MCP_API_KEY", ""),
            "endpoint": os.getenv("MCP_ENDPOINT", "https://api.mcp.example.com"),
            "timeout": 30
        },
        "goat": {
            "api_key": os.getenv("GOAT_API_KEY", ""),
            "endpoint": os.getenv("GOAT_ENDPOINT", "https://api.goat.example.com"),
            "timeout": 30
        },
        "analytics": {
            "time_series_window": 30,
            "forecast_horizon": 7,
            "confidence_threshold": 0.8
        },
        "edge": {
            "max_nodes": 10,
            "min_performance": 0.7,
            "resource_types": ["cpu", "gpu", "memory", "network"]
        },
        "network": {
            "topology_type": "mesh",
            "connection_timeout": 5,
            "retry_attempts": 3,
            "optimization_interval": 300  # 5 minutes
        },
        "forecasting": {
            "model_type": "lstm",
            "sequence_length": 24,
            "prediction_window": 7,
            "update_frequency": 3600  # 1 hour
        }
    } 