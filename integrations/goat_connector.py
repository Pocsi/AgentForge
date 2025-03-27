import logging
import json
import time
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# The following import might fail if the SDK is not installed
try:
    import goat  # GOAT SDK
    GOAT_AVAILABLE = True
except ImportError:
    logger.warning("GOAT SDK not available, using stub implementation")
    GOAT_AVAILABLE = False

class GOATConnector:
    """
    Connector for the GOAT SDK that enables AI agents
    to have financial and economic capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GOAT connector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get("api_key", "")
        self.enabled_tools = config.get("enabled_tools", [])
        
        self.client = None
        self.wallet = None
        self.available_tools = {}
        self.status = {
            "initialized": False,
            "last_error": None,
            "wallet_connected": False
        }
        
        # Try to initialize GOAT
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the GOAT SDK"""
        if not GOAT_AVAILABLE:
            logger.warning("Cannot initialize GOAT: SDK not available")
            self.status["last_error"] = "GOAT SDK not installed"
            return
        
        try:
            # Initialize GOAT client
            self.client = goat.Client(api_key=self.api_key)
            
            # Initialize wallet (if not already initialized)
            if not self.wallet:
                # Use in-memory wallet for now, can be changed later
                self.wallet = self.client.create_wallet("memory")
                self.status["wallet_connected"] = True
            
            # Load available tools based on enabled_tools config
            self._load_tools()
            
            self.status["initialized"] = True
            logger.info(f"GOAT initialized with {len(self.available_tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize GOAT: {str(e)}")
            self.status["initialized"] = False
            self.status["last_error"] = str(e)
            
            # Create a stub client for graceful fallback
            self._create_stub_client()
    
    def _create_stub_client(self) -> None:
        """Create a stub client for graceful fallback"""
        class StubClient:
            def execute_tool(self, *args, **kwargs):
                return {"status": "error", "message": "GOAT SDK not available"}
        
        self.client = StubClient()
        self.available_tools = {
            "payments": {"status": "unavailable"},
            "investments": {"status": "unavailable"},
            "insights": {"status": "unavailable"}
        }
        logger.info("Created stub GOAT client for graceful fallback")
    
    def _load_tools(self) -> None:
        """Load the tools specified in the configuration"""
        if not self.client:
            return
        
        tool_mapping = {
            "payments": ["send", "receive", "balance"],
            "investments": ["yield", "assets", "markets"],
            "insights": ["quotes", "metrics", "reports"]
        }
        
        # Load requested tools
        for tool_category in self.enabled_tools:
            if tool_category in tool_mapping:
                self.available_tools[tool_category] = {}
                
                for tool_name in tool_mapping[tool_category]:
                    try:
                        # Load the tool
                        tool = self.client.load_tool(f"{tool_category}.{tool_name}")
                        self.available_tools[tool_category][tool_name] = tool
                    except Exception as e:
                        logger.warning(f"Failed to load tool {tool_category}.{tool_name}: {str(e)}")
                        self.available_tools[tool_category][tool_name] = {"error": str(e)}
    
    def execute_tools(self, message: str, tools: List[str]) -> Dict[str, Any]:
        """
        Execute GOAT tools based on the message
        
        Args:
            message: User message
            tools: List of tools to execute
            
        Returns:
            Dictionary of tool results
        """
        if not self.client or not self.status["initialized"]:
            return {"status": "error", "message": "GOAT not initialized"}
        
        results = {}
        
        for tool_category in tools:
            if tool_category not in self.available_tools:
                results[tool_category] = {"status": "error", "message": f"Tool category {tool_category} not available"}
                continue
            
            category_results = {}
            
            # Determine which specific tools to use based on the message
            relevant_tools = self._determine_relevant_tools(message, tool_category)
            
            for tool_name in relevant_tools:
                try:
                    # Get the tool
                    tool = self.available_tools[tool_category].get(tool_name)
                    
                    if not tool:
                        category_results[tool_name] = {"status": "error", "message": f"Tool {tool_name} not available"}
                        continue
                    
                    # Execute the tool
                    params = self._extract_tool_params(message, tool_category, tool_name)
                    result = self.client.execute_tool(
                        f"{tool_category}.{tool_name}",
                        wallet=self.wallet,
                        **params
                    )
                    
                    category_results[tool_name] = {
                        "status": "success",
                        "result": result
                    }
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_category}.{tool_name}: {str(e)}")
                    category_results[tool_name] = {
                        "status": "error",
                        "message": str(e)
                    }
            
            results[tool_category] = category_results
        
        return results
    
    def _determine_relevant_tools(self, message: str, category: str) -> List[str]:
        """
        Determine which specific tools to use based on the message
        
        Args:
            message: User message
            category: Tool category
            
        Returns:
            List of relevant tool names
        """
        message = message.lower()
        
        # Simple keyword matching for now
        if category == "payments":
            if "send" in message or "transfer" in message or "pay" in message:
                return ["send"]
            elif "receive" in message or "request" in message:
                return ["receive"]
            else:
                return ["balance"]
        
        elif category == "investments":
            if "yield" in message or "interest" in message or "earn" in message:
                return ["yield"]
            elif "market" in message or "price" in message:
                return ["markets"]
            else:
                return ["assets"]
        
        elif category == "insights":
            if "quote" in message or "price" in message:
                return ["quotes"]
            elif "report" in message or "analysis" in message:
                return ["reports"]
            else:
                return ["metrics"]
        
        # Default: return all tools in the category
        return list(self.available_tools.get(category, {}).keys())
    
    def _extract_tool_params(self, message: str, category: str, tool: str) -> Dict[str, Any]:
        """
        Extract parameters for a specific tool from the message
        
        Args:
            message: User message
            category: Tool category
            tool: Tool name
            
        Returns:
            Dictionary of tool parameters
        """
        # This is a very basic implementation that would need to be improved
        # in a real system with NLP or more sophisticated parsing
        
        params = {}
        message = message.lower()
        
        # Example for payments.send
        if category == "payments" and tool == "send":
            # Try to extract amount and recipient
            import re
            
            # Look for dollar amounts
            amount_match = re.search(r'\$(\d+(?:\.\d+)?)', message)
            if amount_match:
                params["amount"] = float(amount_match.group(1))
            
            # Look for "to <recipient>"
            recipient_match = re.search(r'to\s+([a-zA-Z0-9_]+)', message)
            if recipient_match:
                params["recipient"] = recipient_match.group(1)
        
        # Example for investments.yield
        elif category == "investments" and tool == "yield":
            # Try to extract amount to invest
            import re
            
            # Look for dollar amounts
            amount_match = re.search(r'\$(\d+(?:\.\d+)?)', message)
            if amount_match:
                params["amount"] = float(amount_match.group(1))
            
            # Look for time period
            if "day" in message or "daily" in message:
                params["period"] = "daily"
            elif "week" in message or "weekly" in message:
                params["period"] = "weekly"
            elif "month" in message or "monthly" in message:
                params["period"] = "monthly"
            elif "year" in message or "yearly" in message or "annual" in message:
                params["period"] = "yearly"
        
        return params
    
    def get_wallet_info(self) -> Dict[str, Any]:
        """
        Get information about the connected wallet
        
        Returns:
            Dictionary with wallet information
        """
        if not self.client or not self.wallet or not self.status["wallet_connected"]:
            return {"status": "error", "message": "Wallet not connected"}
        
        try:
            # Get wallet info
            info = self.wallet.get_info()
            return {
                "status": "success",
                "info": info
            }
            
        except Exception as e:
            logger.error(f"Error getting wallet info: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the GOAT connector
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self.status["initialized"],
            "wallet_connected": self.status["wallet_connected"],
            "api_key_configured": bool(self.api_key),
            "last_error": self.status["last_error"],
            "enabled_tools": self.enabled_tools,
            "available_tools": list(self.available_tools.keys()),
            "sdk_available": GOAT_AVAILABLE
        }
