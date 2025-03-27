import logging
import json
import time
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# The following import might fail if the SDK is not installed
try:
    import mcp  # Model Context Protocol SDK
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP SDK not available, using stub implementation")
    MCP_AVAILABLE = False

class MCPConnector:
    """
    Connector for the Model Context Protocol (MCP)
    that enables seamless integration with external context sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MCP connector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.server_url = config.get("server_url", "http://localhost:8080")
        self.api_key = config.get("api_key", "")
        
        self.client = None
        self.status = {
            "connected": False,
            "last_error": None,
            "last_connection_attempt": None
        }
        
        # Try to initialize the MCP client
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the MCP client"""
        if not MCP_AVAILABLE:
            logger.warning("Cannot initialize MCP client: SDK not available")
            self.status["last_error"] = "MCP SDK not installed"
            return
        
        try:
            self.status["last_connection_attempt"] = time.time()
            
            # Initialize MCP client
            self.client = mcp.Client(
                server_url=self.server_url,
                api_key=self.api_key
            )
            
            # Test connection by getting server info
            server_info = self.client.get_server_info()
            logger.info(f"Connected to MCP server: {server_info.get('name', 'Unknown')}, version: {server_info.get('version', 'Unknown')}")
            
            self.status["connected"] = True
            self.status["last_error"] = None
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {str(e)}")
            self.status["connected"] = False
            self.status["last_error"] = str(e)
            
            # Create a stub client for graceful fallback
            self._create_stub_client()
    
    def _create_stub_client(self) -> None:
        """Create a stub client for graceful fallback"""
        class StubClient:
            def get_context(self, *args, **kwargs):
                return {}
            
            def generate_response(self, *args, **kwargs):
                return "I'm currently unable to access my context capabilities. Please try again later."
            
            def query(self, *args, **kwargs):
                return {}
        
        self.client = StubClient()
        logger.info("Created stub MCP client for graceful fallback")
    
    def get_context(self, message: str, existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get context for a message
        
        Args:
            message: The message to get context for
            existing_context: Existing context to augment
            
        Returns:
            Dictionary of context information
        """
        if not self.client:
            logger.warning("MCP client not initialized")
            return {}
        
        try:
            # Prepare context request
            request = {
                "message": message,
                "existing_context": existing_context or {}
            }
            
            # Get context from MCP
            context = self.client.get_context(request)
            
            logger.info(f"Retrieved context with {len(context)} items")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context from MCP: {str(e)}")
            self.status["last_error"] = str(e)
            
            # Try to reconnect
            self._try_reconnect()
            
            return {}
    
    def generate_response(self, data: Dict[str, Any]) -> str:
        """
        Generate a response using MCP
        
        Args:
            data: Data for response generation
            
        Returns:
            Generated response
        """
        if not self.client:
            logger.warning("MCP client not initialized")
            return "I'm currently experiencing connection issues. Please try again later."
        
        try:
            # Generate response
            response = self.client.generate_response(data)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with MCP: {str(e)}")
            self.status["last_error"] = str(e)
            
            # Try to reconnect
            self._try_reconnect()
            
            return f"I'm having trouble generating a response right now. Error: {str(e)}"
    
    def query(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query a specific MCP endpoint
        
        Args:
            endpoint: Endpoint to query
            data: Query data
            
        Returns:
            Query result
        """
        if not self.client:
            logger.warning("MCP client not initialized")
            return {}
        
        try:
            # Query endpoint
            result = self.client.query(endpoint, data)
            return result
            
        except Exception as e:
            logger.error(f"Error querying MCP endpoint {endpoint}: {str(e)}")
            self.status["last_error"] = str(e)
            
            # Try to reconnect
            self._try_reconnect()
            
            return {}
    
    def _try_reconnect(self) -> None:
        """Try to reconnect to the MCP server"""
        # Only attempt reconnection if some time has passed
        current_time = time.time()
        last_attempt = self.status.get("last_connection_attempt", 0)
        
        if current_time - last_attempt > 60:  # Wait at least 60 seconds between attempts
            logger.info("Attempting to reconnect to MCP server")
            self._initialize()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the MCP connector
        
        Returns:
            Dictionary with status information
        """
        return {
            "connected": self.status["connected"],
            "server_url": self.server_url,
            "api_key_configured": bool(self.api_key),
            "last_error": self.status["last_error"],
            "sdk_available": MCP_AVAILABLE
        }
