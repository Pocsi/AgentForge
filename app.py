import streamlit as st
import os
import logging
from datetime import datetime

from config import AppConfig
from core.agent import AgentManager
from core.distributed import DistributedCompute
from integrations.mcp_connector import MCPConnector
from integrations.goat_connector import GOATConnector
from analytics.time_series import TimeSeriesAnalyzer
from analytics.forecasting import Forecaster
from signal.processor import SignalProcessor
from ui.components import sidebar, header, footer
from ui.pages import (
    dashboard_page,
    conversation_page,
    analytics_page,
    signal_page,
    settings_page
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "conversation"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'distributed_compute' not in st.session_state:
        st.session_state.distributed_compute = None
    if 'mcp_connector' not in st.session_state:
        st.session_state.mcp_connector = None
    if 'goat_connector' not in st.session_state:
        st.session_state.goat_connector = None
    if 'time_series_analyzer' not in st.session_state:
        st.session_state.time_series_analyzer = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'signal_processor' not in st.session_state:
        st.session_state.signal_processor = None
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False

def initialize_components():
    """Initialize all the components of the application"""
    try:
        # Load config
        config = AppConfig()
        
        # Initialize distributed compute
        st.session_state.distributed_compute = DistributedCompute(
            config.get_distributed_config()
        )
        
        # Initialize integrations
        st.session_state.mcp_connector = MCPConnector(
            config.get_mcp_config()
        )
        
        st.session_state.goat_connector = GOATConnector(
            config.get_goat_config()
        )
        
        # Initialize analytics
        st.session_state.time_series_analyzer = TimeSeriesAnalyzer(
            config.get_analytics_config()
        )
        
        st.session_state.forecaster = Forecaster(
            config.get_forecasting_config(),
            st.session_state.distributed_compute
        )
        
        # Initialize signal processor
        st.session_state.signal_processor = SignalProcessor(
            config.get_signal_config(),
            st.session_state.distributed_compute
        )
        
        # Initialize agent
        st.session_state.agent = AgentManager(
            config.get_agent_config(),
            st.session_state.mcp_connector,
            st.session_state.goat_connector,
            st.session_state.time_series_analyzer,
            st.session_state.forecaster,
            st.session_state.signal_processor,
            st.session_state.distributed_compute
        )
        
        st.session_state.is_initialized = True
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="AI Companion",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize components if not already done
    if not st.session_state.is_initialized:
        with st.spinner("Initializing components..."):
            initialize_components()
    
    # Setup UI
    header()
    sidebar()
    
    # Render current page
    if st.session_state.current_page == "dashboard":
        dashboard_page()
    elif st.session_state.current_page == "conversation":
        conversation_page()
    elif st.session_state.current_page == "analytics":
        analytics_page()
    elif st.session_state.current_page == "signal":
        signal_page()
    elif st.session_state.current_page == "settings":
        settings_page()
    
    footer()

if __name__ == "__main__":
    main()
