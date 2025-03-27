import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages chat history and memory for the AI Companion
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize the memory manager
        
        Args:
            max_size: Maximum number of messages to store
        """
        self.max_size = max_size
        self.messages = []
        self.last_compression = None
        
        logger.info(f"Memory manager initialized with max size {max_size}")
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to memory
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        
        # Trim if necessary
        if len(self.messages) > self.max_size:
            self._trim_memory()
    
    def _trim_memory(self) -> None:
        """Trim memory to max size by removing oldest messages"""
        excess = len(self.messages) - self.max_size
        if excess > 0:
            self.messages = self.messages[excess:]
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent messages
        
        Args:
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of messages
        """
        if limit and limit < len(self.messages):
            return self.messages[-limit:]
        return self.messages
    
    def clear(self) -> None:
        """Clear all messages from memory"""
        self.messages = []
    
    def should_compress(self) -> bool:
        """
        Check if memory should be compressed
        
        Returns:
            True if compression is needed, False otherwise
        """
        # Check if we're at 80% capacity
        return len(self.messages) >= int(0.8 * self.max_size)
    
    def compress(self, compressor) -> None:
        """
        Compress memory using the provided compressor
        
        Args:
            compressor: Text compressor object
        """
        if not self.messages:
            return
        
        # Only compress user messages for now
        user_messages = [msg for msg in self.messages if msg.get("role") == "user"]
        
        if len(user_messages) > 3:
            # Get text from messages
            texts = [msg.get("content", "") for msg in user_messages[:-2]]
            
            # Compress
            compressed_text = compressor.compress_text("\n".join(texts))
            
            # Create a new compressed message
            compressed_message = {
                "role": "system",
                "content": compressed_text,
                "timestamp": datetime.now().isoformat(),
                "compressed": True,
                "original_count": len(texts)
            }
            
            # Replace the old messages with the compressed one
            # Keep the most recent messages
            new_messages = [msg for msg in self.messages if msg.get("role") != "user" or msg in user_messages[-2:]]
            
            # Insert compressed message at the beginning
            new_messages.insert(0, compressed_message)
            
            self.messages = new_messages
            self.last_compression = datetime.now()
            
            logger.info(f"Compressed {len(texts)} messages into one")
        else:
            logger.info("Not enough messages to compress")