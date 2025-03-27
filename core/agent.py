import logging
from typing import Dict, Any, List, Tuple, Optional
import json
import time
from datetime import datetime

from core.memory import MemoryManager
from core.compression import TextCompressor

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Main agent manager that coordinates all the components
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mcp_connector=None,
        goat_connector=None,
        time_series_analyzer=None,
        forecaster=None,
        signal_processor=None,
        distributed_compute=None
    ):
        """
        Initialize the agent manager
        
        Args:
            config: Configuration dictionary
            mcp_connector: MCP connector instance
            goat_connector: GOAT connector instance
            time_series_analyzer: Time series analyzer instance
            forecaster: Forecaster instance
            signal_processor: Signal processor instance
            distributed_compute: Distributed compute instance
        """
        self.config = config
        self.name = config.get("name", "AI Companion")
        self.personality = config.get("personality", "helpful, informative, and insightful")
        
        # Initialize memory
        memory_size = config.get("memory_size", 50)
        self.memory = MemoryManager(max_size=memory_size)
        
        # Initialize text compressor
        compression_enabled = config.get("compression_enabled", True)
        if compression_enabled:
            self.compressor = TextCompressor()
        else:
            self.compressor = None
        
        # Store components
        self.mcp_connector = mcp_connector
        self.goat_connector = goat_connector
        self.time_series_analyzer = time_series_analyzer
        self.forecaster = forecaster
        self.signal_processor = signal_processor
        self.distributed_compute = distributed_compute
        
        logger.info(f"Agent '{self.name}' initialized")
    
    def process_message(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a message from the user
        
        Args:
            message: User message
            
        Returns:
            Tuple of (response, metadata)
        """
        # Add user message to memory
        self.memory.add_message({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log the received message
        logger.info(f"Processing message: {message[:50]}...")
        
        # Process with appropriate component based on content
        if "forecast" in message.lower() or "predict" in message.lower():
            if self.forecaster:
                # Handle forecasting requests
                results = self.forecaster.forecast(message)
                response = f"I've analyzed your forecasting request. Here are the results:\n\n{json.dumps(results, indent=2)}"
                metadata = {"type": "forecast", "results": results}
            else:
                response = "I'm sorry, forecasting capabilities are not available."
                metadata = {"type": "error", "component": "forecaster"}
        
        elif "analyze" in message.lower() or "time series" in message.lower():
            if self.time_series_analyzer:
                # Handle time series analysis requests
                results = self.time_series_analyzer.analyze(message)
                response = f"I've analyzed your time series request. Here are the results:\n\n{json.dumps(results, indent=2)}"
                metadata = {"type": "time_series", "results": results}
            else:
                response = "I'm sorry, time series analysis capabilities are not available."
                metadata = {"type": "error", "component": "time_series_analyzer"}
        
        elif "signal" in message.lower() or "filter" in message.lower() or "frequency" in message.lower():
            if self.signal_processor:
                # Handle signal processing requests
                results = self.signal_processor.process(message)
                response = f"I've processed your signal request. Here are the results:\n\n{json.dumps(results, indent=2)}"
                metadata = {"type": "signal", "results": results}
            else:
                response = "I'm sorry, signal processing capabilities are not available."
                metadata = {"type": "error", "component": "signal_processor"}
        
        else:
            # General response
            response = self._generate_general_response(message)
            metadata = {"type": "general"}
        
        # Add agent message to memory
        self.memory.add_message({
            "role": "agent",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
        
        # Compress memory if needed
        if self.compressor and self.memory.should_compress():
            self.memory.compress(self.compressor)
        
        return response, metadata
    
    def _generate_general_response(self, message: str) -> str:
        """
        Generate a general response for messages that don't match specific components
        
        Args:
            message: User message
            
        Returns:
            Response string
        """
        # Simple response for now
        if "hello" in message.lower() or "hi" in message.lower():
            return f"Hello! I'm {self.name}, your AI Companion. How can I assist you today?"
        
        if "help" in message.lower() or "capabilities" in message.lower():
            capabilities = [
                "Time Series Analysis - Analyze trends, patterns, and anomalies in time series data",
                "Forecasting - Predict future values based on historical time series data",
                "Signal Processing - Filter, transform, and analyze signal data from various sources",
                "IoT Device Management - Connect to and collect data from IoT devices",
                "Distributed Computing - Execute compute-intensive tasks across multiple worker nodes"
            ]
            
            capabilities_str = "\n".join([f"- {c}" for c in capabilities])
            return f"I'm {self.name}, a high-agency AI Companion. Here are my main capabilities:\n\n{capabilities_str}\n\nYou can interact with me through natural language, or use the specific interfaces in the sidebar tabs."
        
        if "thank" in message.lower():
            return "You're welcome! Let me know if you need assistance with anything else."
        
        # Default response
        return f"I understand your message. Is there a specific analysis, forecast, or signal processing task you'd like me to help with?"
    
    def get_agent_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent and all its components
        
        Returns:
            Dictionary with agent state
        """
        # Basic agent info
        state = {
            "name": self.name,
            "personality": self.personality,
            "timestamp": datetime.now().isoformat()
        }
        
        # Memory usage
        if self.memory:
            state["memory_usage"] = {
                "current_size": len(self.memory.messages),
                "max_size": self.memory.max_size,
                "usage_percentage": int((len(self.memory.messages) / self.memory.max_size) * 100),
                "last_compression": self.memory.last_compression.isoformat() if self.memory.last_compression else None
            }
        
        # Component states
        state["components"] = {}
        
        # MCP state
        if self.mcp_connector:
            state["components"]["mcp"] = self.mcp_connector.get_status()
        
        # GOAT state
        if self.goat_connector:
            state["components"]["goat"] = self.goat_connector.get_status()
        
        # Distributed compute state
        if self.distributed_compute:
            state["components"]["distributed_compute"] = self.distributed_compute.get_status()
        
        # Time series analyzer state
        if self.time_series_analyzer:
            state["components"]["time_series_analyzer"] = self.time_series_analyzer.get_status()
        
        # Forecaster state
        if self.forecaster:
            state["components"]["forecaster"] = self.forecaster.get_status()
        
        # Signal processor state
        if self.signal_processor:
            state["components"]["signal_processor"] = self.signal_processor.get_status()
        
        return state