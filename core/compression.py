import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)

class TextCompressor:
    """
    Handles compression of text to reduce memory usage,
    particularly for summarizing conversation history
    """
    
    def __init__(self, max_length: int = 500):
        """
        Initialize the text compressor
        
        Args:
            max_length: Maximum length of compressed text
        """
        self.max_length = max_length
        logger.info(f"Text compressor initialized with max length {max_length}")
    
    def compress_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """
        Compress a list of conversation pairs
        
        Args:
            conversations: List of conversation dictionaries with 'user' and 'agent' keys
            
        Returns:
            Compressed summary of the conversations
        """
        if not conversations:
            return ""
        
        # Group conversations by topic using simple heuristics
        topics = self._identify_topics(conversations)
        
        # Generate summaries for each topic
        topic_summaries = []
        for topic_name, topic_conversations in topics.items():
            summary = self._summarize_topic(topic_name, topic_conversations)
            topic_summaries.append(summary)
        
        # Combine topic summaries
        combined_summary = " | ".join(topic_summaries)
        
        # Ensure the summary doesn't exceed the maximum length
        if len(combined_summary) > self.max_length:
            combined_summary = combined_summary[:self.max_length - 3] + "..."
        
        logger.info(f"Compressed {len(conversations)} conversations into {len(combined_summary)} characters")
        return combined_summary
    
    def _identify_topics(self, conversations: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Group conversations by topic using simple keyword matching
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Dictionary mapping topic names to lists of conversations
        """
        # Simple keyword-based topic identification
        topics = {
            "finance": [],
            "time_series": [],
            "forecasting": [],
            "signal_processing": [],
            "general": []
        }
        
        # Keywords for each topic
        topic_keywords = {
            "finance": ["money", "payment", "invest", "finance", "bank", "transaction", "crypto", "wallet"],
            "time_series": ["time series", "data analysis", "trend", "pattern", "historical data"],
            "forecasting": ["forecast", "predict", "future", "projection", "model"],
            "signal_processing": ["signal", "iot", "sensor", "frequency", "filter", "edge"]
        }
        
        for conv in conversations:
            # Combine user and agent messages for topic detection
            combined_text = (conv.get("user", "") + " " + conv.get("agent", "")).lower()
            
            assigned = False
            for topic, keywords in topic_keywords.items():
                if any(keyword in combined_text for keyword in keywords):
                    topics[topic].append(conv)
                    assigned = True
                    break
            
            # If no specific topic matched, add to general
            if not assigned:
                topics["general"].append(conv)
        
        # Remove empty topics
        return {k: v for k, v in topics.items() if v}
    
    def _summarize_topic(self, topic: str, conversations: List[Dict[str, str]]) -> str:
        """
        Create a summary for conversations in a specific topic
        
        Args:
            topic: Topic name
            conversations: List of conversation dictionaries for this topic
            
        Returns:
            Summary string for this topic
        """
        # Extract key information based on topic
        if topic == "finance":
            return self._summarize_finance_conversations(conversations)
        elif topic in ["time_series", "forecasting"]:
            return self._summarize_data_analysis_conversations(conversations)
        elif topic == "signal_processing":
            return self._summarize_signal_conversations(conversations)
        else:
            return self._summarize_general_conversations(conversations)
    
    def _summarize_finance_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """Summarize finance-related conversations"""
        # Look for amounts, transactions, etc.
        amounts = []
        transactions = []
        
        for conv in conversations:
            user_msg = conv.get("user", "").lower()
            
            # Extract dollar amounts
            amount_matches = re.findall(r'\$\d+(?:\.\d+)?|\d+\s+dollars', user_msg)
            amounts.extend(amount_matches)
            
            # Extract transaction keywords
            if any(word in user_msg for word in ["send", "transfer", "pay", "transaction"]):
                transactions.append(user_msg)
        
        # Create summary
        summary = f"Finance: "
        if amounts:
            summary += f"Discussed amounts {', '.join(amounts[:3])}"
            if len(amounts) > 3:
                summary += f" and {len(amounts) - 3} more"
            summary += ". "
        
        if transactions:
            summary += f"Discussed {len(transactions)} transactions. "
        
        summary += f"Total {len(conversations)} finance conversations."
        return summary
    
    def _summarize_data_analysis_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """Summarize data analysis-related conversations"""
        analysis_types = set()
        data_sources = set()
        
        for conv in conversations:
            combined_text = (conv.get("user", "") + " " + conv.get("agent", "")).lower()
            
            # Extract analysis types
            if "forecast" in combined_text:
                analysis_types.add("forecasting")
            if "trend" in combined_text:
                analysis_types.add("trend analysis")
            if "pattern" in combined_text:
                analysis_types.add("pattern recognition")
            if "anomaly" in combined_text:
                analysis_types.add("anomaly detection")
            
            # Extract data sources
            for source in ["database", "csv", "excel", "api", "sensor"]:
                if source in combined_text:
                    data_sources.add(source)
        
        # Create summary
        summary = f"Data Analysis: "
        if analysis_types:
            summary += f"Types: {', '.join(list(analysis_types)[:3])}"
            if len(analysis_types) > 3:
                summary += f" and {len(analysis_types) - 3} more"
            summary += ". "
        
        if data_sources:
            summary += f"Sources: {', '.join(list(data_sources)[:3])}"
            if len(data_sources) > 3:
                summary += f" and {len(data_sources) - 3} more"
            summary += ". "
        
        summary += f"Total {len(conversations)} data analysis conversations."
        return summary
    
    def _summarize_signal_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """Summarize signal processing-related conversations"""
        signal_types = set()
        processing_methods = set()
        
        for conv in conversations:
            combined_text = (conv.get("user", "") + " " + conv.get("agent", "")).lower()
            
            # Extract signal types
            for signal_type in ["audio", "sensor", "iot", "rf", "radio", "wireless"]:
                if signal_type in combined_text:
                    signal_types.add(signal_type)
            
            # Extract processing methods
            for method in ["filter", "transform", "fft", "wavelet", "compression"]:
                if method in combined_text:
                    processing_methods.add(method)
        
        # Create summary
        summary = f"Signal Processing: "
        if signal_types:
            summary += f"Types: {', '.join(list(signal_types)[:3])}"
            if len(signal_types) > 3:
                summary += f" and {len(signal_types) - 3} more"
            summary += ". "
        
        if processing_methods:
            summary += f"Methods: {', '.join(list(processing_methods)[:3])}"
            if len(processing_methods) > 3:
                summary += f" and {len(processing_methods) - 3} more"
            summary += ". "
        
        summary += f"Total {len(conversations)} signal processing conversations."
        return summary
    
    def _summarize_general_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """Summarize general conversations"""
        # Extract frequent words (simple approach)
        word_count = {}
        
        for conv in conversations:
            combined_text = (conv.get("user", "") + " " + conv.get("agent", "")).lower()
            words = combined_text.split()
            
            for word in words:
                # Skip short and common words
                if len(word) <= 3 or word in ["the", "and", "for", "this", "that", "with", "you"]:
                    continue
                
                word_count[word] = word_count.get(word, 0) + 1
        
        # Get most frequent words
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create summary
        summary = f"General: "
        if top_words:
            summary += f"Topics: {', '.join([word for word, _ in top_words])}. "
        
        summary += f"Total {len(conversations)} general conversations."
        return summary
