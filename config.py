import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AppConfig:
    """Configuration manager for the AI Companion application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {
            # Default configuration
            "agent": {
                "name": "AI Companion",
                "personality": "helpful, informative, and insightful",
                "memory_size": 50,
                "compression_enabled": True
            },
            "distributed": {
                "enabled": True,
                "scheduler_address": None,  # Auto-discover
                "worker_threads": os.cpu_count(),
                "memory_limit": "4GB"
            },
            "mcp": {
                "server_url": os.getenv("MCP_SERVER_URL", "http://localhost:8080"),
                "api_key": os.getenv("MCP_API_KEY", "")
            },
            "goat": {
                "api_key": os.getenv("GOAT_API_KEY", ""),
                "enabled_tools": ["payments", "investments", "insights"]
            },
            "analytics": {
                "window_size": 30,
                "resample_frequency": "1D",
                "decomposition_method": "additive"
            },
            "forecasting": {
                "prediction_horizon": 7,
                "confidence_interval": 0.95,
                "methods": ["arima", "prophet", "lstm"]
            },
            "signal": {
                "sampling_rate": 100,
                "filter_type": "butterworth",
                "cutoff_frequency": 20,
                "iot_enabled": True
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self._update_nested_dict(self.config, loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.get("agent", {})
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """Get distributed computation configuration"""
        return self.config.get("distributed", {})
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get Model Context Protocol configuration"""
        return self.config.get("mcp", {})
    
    def get_goat_config(self) -> Dict[str, Any]:
        """Get GOAT SDK configuration"""
        return self.config.get("goat", {})
    
    def get_analytics_config(self) -> Dict[str, Any]:
        """Get analytics configuration"""
        return self.config.get("analytics", {})
    
    def get_forecasting_config(self) -> Dict[str, Any]:
        """Get forecasting configuration"""
        return self.config.get("forecasting", {})
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get signal processing configuration"""
        return self.config.get("signal", {})
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section in self.config:
            self.config[section][key] = value
            logger.info(f"Updated config {section}.{key} = {value}")
        else:
            logger.warning(f"Config section {section} does not exist")
    
    def save_config(self, config_path: str) -> bool:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            return False
