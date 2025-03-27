import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import time
from datetime import datetime
from core.compression import TextCompressor

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the agent's memory, including conversation history
    and compression capabilities
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize the memory manager
        
        Args:
            max_size: Maximum number of messages to store
        """
        self.max_size = max_size
        self.messages = deque(maxlen=max_size)
        self.last_compression_time = time.time()
        self.compression_interval = 3600  # 1 hour
        
        logger.info(f"Memory manager initialized with max size {max_size}")
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to memory
        
        Args:
            message: User message
        """
        self._add_message("user", message)
    
    def add_agent_message(self, message: str) -> None:
        """
        Add an agent message to memory
        
        Args:
            message: Agent message
        """
        self._add_message("agent", message)
    
    def _add_message(self, role: str, content: str) -> None:
        """
        Add a message to memory
        
        Args:
            role: Message role (user or agent)
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
    
    def get_all_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in memory
        
        Returns:
            List of all messages
        """
        return list(self.messages)
    
    def get_recent_messages(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the n most recent messages
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        return list(self.messages)[-n:] if n < len(self.messages) else list(self.messages)
    
    def clear_memory(self) -> None:
        """Clear all messages from memory"""
        self.messages.clear()
        logger.info("Memory cleared")
    
    def resize(self, new_size: int) -> None:
        """
        Resize the memory
        
        Args:
            new_size: New maximum size
        """
        if new_size <= 0:
            logger.warning(f"Invalid memory size: {new_size}, must be positive")
            return
            
        # Create a new deque with the new size
        old_messages = list(self.messages)
        self.messages = deque(old_messages[-new_size:] if new_size < len(old_messages) else old_messages, maxlen=new_size)
        self.max_size = new_size
        
        logger.info(f"Memory resized to {new_size}")
    
    def should_compress(self) -> bool:
        """
        Check if memory should be compressed based on time or size
        
        Returns:
            True if compression is recommended, False otherwise
        """
        # Check if enough time has passed since last compression
        time_based = (time.time() - self.last_compression_time) > self.compression_interval
        
        # Check if we're using more than 80% of capacity
        size_based = len(self.messages) > (self.max_size * 0.8)
        
        return time_based or size_based
    
    def compress_memory(self, compressor: TextCompressor) -> None:
        """
        Compress memory to save space
        
        Args:
            compressor: Text compressor to use
        """
        if len(self.messages) < 10:
            # Not enough messages to compress
            return
        
        try:
            # Get user-agent conversation pairs
            conversation_pairs = []
            current_pair = {}
            
            for msg in list(self.messages)[:-5]:  # Keep the most recent 5 messages uncompressed
                if msg["role"] == "user":
                    current_pair = {"user": msg["content"]}
                elif msg["role"] == "agent" and "user" in current_pair:
                    current_pair["agent"] = msg["content"]
                    conversation_pairs.append(current_pair)
                    current_pair = {}
            
            # Compress conversation pairs
            if conversation_pairs:
                compressed_summary = compressor.compress_conversations(conversation_pairs)
                
                # Replace the old messages with the compressed summary
                old_messages = list(self.messages)
                recent_messages = old_messages[-5:]  # Keep the 5 most recent messages
                
                self.messages.clear()
                
                # Add the compressed summary as a system message
                self._add_message("system", f"MEMORY SUMMARY: {compressed_summary}")
                
                # Add back the recent messages
                for msg in recent_messages:
                    self.messages.append(msg)
                
                self.last_compression_time = time.time()
                logger.info(f"Memory compressed, {len(conversation_pairs)} conversation pairs summarized")
        
        except Exception as e:
            logger.error(f"Error compressing memory: {str(e)}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics
        
        Returns:
            Dictionary of usage statistics
        """
        return {
            "current_size": len(self.messages),
            "max_size": self.max_size,
            "usage_percentage": (len(self.messages) / self.max_size) * 100 if self.max_size > 0 else 0,
            "last_compression": datetime.fromtimestamp(self.last_compression_time).isoformat() if self.last_compression_time else None
        }
