import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime

from core.memory import MemoryManager
from core.compression import TextCompressor
from integrations.mcp_connector import MCPConnector
from integrations.goat_connector import GOATConnector
from analytics.time_series import TimeSeriesAnalyzer
from analytics.forecasting import Forecaster
from signal_processing.processor import SignalProcessor
from core.distributed import DistributedCompute

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Main agent manager that coordinates all the components and provides
    a high-agency AI companion experience
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mcp_connector: MCPConnector,
        goat_connector: GOATConnector,
        time_series_analyzer: TimeSeriesAnalyzer,
        forecaster: Forecaster,
        signal_processor: SignalProcessor,
        distributed_compute: DistributedCompute
    ):
        """
        Initialize the agent manager
        
        Args:
            config: Agent configuration
            mcp_connector: Model Context Protocol connector
            goat_connector: GOAT SDK connector
            time_series_analyzer: Time series analyzer
            forecaster: Forecasting module
            signal_processor: Signal processor
            distributed_compute: Distributed computation engine
        """
        self.config = config
        self.name = config.get("name", "AI Companion")
        self.personality = config.get("personality", "helpful, informative, and insightful")
        
        # Connect components
        self.mcp_connector = mcp_connector
        self.goat_connector = goat_connector
        self.time_series_analyzer = time_series_analyzer
        self.forecaster = forecaster
        self.signal_processor = signal_processor
        self.distributed_compute = distributed_compute
        
        # Initialize memory and compression
        memory_size = config.get("memory_size", 50)
        compression_enabled = config.get("compression_enabled", True)
        
        self.memory = MemoryManager(memory_size)
        self.compressor = TextCompressor() if compression_enabled else None
        
        logger.info(f"Agent '{self.name}' initialized")
    
    def process_message(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user message and generate a response
        
        Args:
            message: User message
            
        Returns:
            Tuple of response message and metadata
        """
        logger.info(f"Processing message: {message[:50]}...")
        start_time = time.time()
        
        # Add message to memory
        self.memory.add_user_message(message)
        
        # Determine required tools and context
        required_tools = self._determine_required_tools(message)
        
        # Gather context from MCP
        context = self._gather_context(message)
        
        # Execute tools if needed
        tool_results = self._execute_tools(message, required_tools)
        
        # Prepare response using MCP
        response, metadata = self._generate_response(message, context, tool_results)
        
        # Add response to memory
        self.memory.add_agent_message(response)
        
        # Compress memory if needed
        if self.compressor and self.memory.should_compress():
            self.memory.compress_memory(self.compressor)
        
        processing_time = time.time() - start_time
        logger.info(f"Message processed in {processing_time:.2f}s")
        
        return response, metadata
    
    def _determine_required_tools(self, message: str) -> List[str]:
        """
        Determine which tools might be needed for this message
        
        Args:
            message: User message
            
        Returns:
            List of tool identifiers that might be needed
        """
        # Use MCP to determine which tools might be needed
        tools_query = {
            "message": message,
            "available_tools": [
                "time_series", "forecasting", "signal_processing",
                "payments", "investments", "insights", "database"
            ]
        }
        
        try:
            # Distributed execution of tool selection
            future = self.distributed_compute.submit(
                self.mcp_connector.query,
                "tool_selection",
                tools_query
            )
            result = future.result()
            
            # Parse result
            if isinstance(result, dict) and "selected_tools" in result:
                return result["selected_tools"]
            return []
            
        except Exception as e:
            logger.error(f"Error determining required tools: {str(e)}")
            return []
    
    def _gather_context(self, message: str) -> Dict[str, Any]:
        """
        Gather context from various sources using MCP
        
        Args:
            message: User message
            
        Returns:
            Dictionary of context information
        """
        try:
            # Chat history context
            history_context = {
                "chat_history": self.memory.get_recent_messages(10)
            }
            
            # Get additional context from MCP
            mcp_context = self.mcp_connector.get_context(message, history_context)
            
            return {
                **history_context,
                **mcp_context
            }
            
        except Exception as e:
            logger.error(f"Error gathering context: {str(e)}")
            return {"chat_history": self.memory.get_recent_messages(10)}
    
    def _execute_tools(self, message: str, tools: List[str]) -> Dict[str, Any]:
        """
        Execute needed tools based on the message
        
        Args:
            message: User message
            tools: List of tools to execute
            
        Returns:
            Dictionary of tool results
        """
        results = {}
        
        try:
            futures = []
            
            # Process time series analysis if needed
            if "time_series" in tools:
                futures.append(
                    (
                        "time_series", 
                        self.distributed_compute.submit(
                            self.time_series_analyzer.analyze,
                            message
                        )
                    )
                )
            
            # Process forecasting if needed
            if "forecasting" in tools:
                futures.append(
                    (
                        "forecasting", 
                        self.distributed_compute.submit(
                            self.forecaster.forecast,
                            message
                        )
                    )
                )
            
            # Process signal analysis if needed
            if "signal_processing" in tools:
                futures.append(
                    (
                        "signal_processing", 
                        self.distributed_compute.submit(
                            self.signal_processor.process,
                            message
                        )
                    )
                )
            
            # Process GOAT tools if needed
            goat_tools = set(tools) & {"payments", "investments", "insights"}
            if goat_tools:
                futures.append(
                    (
                        "goat", 
                        self.distributed_compute.submit(
                            self.goat_connector.execute_tools,
                            message,
                            list(goat_tools)
                        )
                    )
                )
            
            # Collect results
            for name, future in futures:
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Error executing tool {name}: {str(e)}")
                    results[name] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing tools: {str(e)}")
            return {"error": str(e)}
    
    def _generate_response(
        self, 
        message: str, 
        context: Dict[str, Any], 
        tool_results: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an agent response
        
        Args:
            message: User message
            context: Context information
            tool_results: Results from various tools
            
        Returns:
            Tuple of response message and metadata
        """
        try:
            # Prepare response data
            response_data = {
                "message": message,
                "context": context,
                "tool_results": tool_results,
                "personality": self.personality,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get response from MCP
            response = self.mcp_connector.generate_response(response_data)
            
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "tools_used": list(tool_results.keys()),
                "context_sources": list(context.keys() if isinstance(context, dict) else [])
            }
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}", {"error": str(e)}
    
    def get_agent_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent
        
        Returns:
            Dictionary containing agent state information
        """
        return {
            "name": self.name,
            "personality": self.personality,
            "memory_usage": self.memory.get_usage_stats(),
            "components": {
                "mcp": self.mcp_connector.get_status(),
                "goat": self.goat_connector.get_status(),
                "time_series": self.time_series_analyzer.get_status(),
                "forecasting": self.forecaster.get_status(),
                "signal_processor": self.signal_processor.get_status(),
                "distributed_compute": self.distributed_compute.get_status()
            }
        }
    
    def update_agent_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update agent configuration
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
                
                # Update specific attributes
                if key == "name":
                    self.name = value
                elif key == "personality":
                    self.personality = value
                elif key == "memory_size":
                    self.memory.resize(value)
                elif key == "compression_enabled":
                    if value and self.compressor is None:
                        self.compressor = TextCompressor()
                    elif not value:
                        self.compressor = None
        
        logger.info(f"Agent configuration updated: {config_updates}")
