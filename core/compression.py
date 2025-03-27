import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

class TextCompressor:
    """
    Compresses text using a simple summarization approach
    """
    
    def __init__(self, max_length: int = 500):
        """
        Initialize the text compressor
        
        Args:
            max_length: Maximum length of compressed text
        """
        self.max_length = max_length
        logger.info(f"Text compressor initialized with max length {max_length}")
    
    def compress_text(self, text: str) -> str:
        """
        Compress text by summarizing it
        
        Args:
            text: Text to compress
            
        Returns:
            Compressed text
        """
        if not text or len(text) <= self.max_length:
            return text
        
        # Simple compression: keep first and last parts
        first_part = text[:self.max_length // 2]
        last_part = text[-(self.max_length // 2):]
        
        compressed = f"{first_part}...[Compressed: {len(text)} characters]...{last_part}"
        
        return compressed