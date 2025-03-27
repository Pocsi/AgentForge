import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

def sidebar():
    """Render the sidebar with navigation"""
    with st.sidebar:
        st.title("AI Companion")
        
        # Navigation
        st.subheader("Navigation")
        
        # Use buttons for navigation
        if st.button("💬 Conversation", use_container_width=True):
            st.session_state.current_page = "conversation"
            st.rerun()
            
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
            
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.current_page = "analytics"
            st.rerun()
            
        if st.button("🔊 Signal Processing", use_container_width=True):
            st.session_state.current_page = "signal"
            st.rerun()
            
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
        
        # System status indicator
        st.markdown("---")
        st.subheader("System Status")
        
        if st.session_state.is_initialized:
            st.success("All systems operational")
            
            # Show some basic metrics if agent is initialized
            if st.session_state.agent:
                agent_state = st.session_state.agent.get_agent_state()
                memory_usage = agent_state.get("memory_usage", {})
                
                if memory_usage:
                    memory_percent = memory_usage.get("usage_percentage", 0)
                    st.caption(f"Memory: {memory_percent}%")
                    
                components = agent_state.get("components", {})
                
                # MCP Status
                mcp_status = components.get("mcp", {})
                if mcp_status.get("connected", False):
                    st.caption("MCP: Connected")
                else:
                    st.caption("MCP: Disconnected")
                
                # Distributed status
                dc_status = components.get("distributed_compute", {})
                if dc_status.get("enabled", False):
                    workers = dc_status.get("connected_workers", 0)
                    st.caption(f"Distributed: {workers} workers")
                else:
                    st.caption("Distributed: Local only")
        else:
            st.error("System not initialized")
            if st.button("Initialize"):
                with st.spinner("Initializing..."):
                    st.session_state.is_initialized = False  # Reset to trigger initialization
                    st.rerun()

def header():
    """Render the application header"""
    current_page = st.session_state.current_page.capitalize()
    st.title(f"{current_page}")
    
    # Current time display
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Current time: {current_time}")

def footer():
    """Render the application footer"""
    st.markdown("---")
    st.caption("AI Companion - High Agency Partner for Advanced Analytics, Signal Processing, and Time Series Forecasting")

def display_chat_message(message: str, is_user: bool):
    """Display a chat message with the appropriate styling"""
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #2b313e; border-radius: 10px; padding: 10px; max-width: 80%;">
                <p style="color: white; margin: 0;">{message}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #0e1117; border: 1px solid #262730; border-radius: 10px; padding: 10px; max-width: 80%;">
                <p style="color: white; margin: 0;">{message}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_time_series(data: List[Dict[str, Any]], title: str = "Time Series", height: int = 400):
    """Display a time series chart using Plotly"""
    # Extract time and value from data
    times = [point.get("time") for point in data]
    values = [point.get("value") for point in data]
    
    # Create figure
    fig = go.Figure()
    
    # Add time series
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines',
        name='Value'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)

def display_frequency_spectrum(data: List[Dict[str, Any]], title: str = "Frequency Spectrum", height: int = 400):
    """Display a frequency spectrum chart using Plotly"""
    # Extract frequency and amplitude from data
    frequencies = [point.get("frequency") for point in data]
    amplitudes = [point.get("amplitude") for point in data]
    
    # Create figure
    fig = go.Figure()
    
    # Add frequency spectrum
    fig.add_trace(go.Bar(
        x=frequencies,
        y=amplitudes,
        name='Amplitude'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)

def display_comparison_chart(datasets: List[Dict[str, Any]], title: str = "Comparison", height: int = 400):
    """Display a comparison chart for multiple datasets using Plotly"""
    # Create figure
    fig = go.Figure()
    
    # Add each dataset
    for dataset in datasets:
        name = dataset.get("name", "Unknown")
        x_data = dataset.get("x", [])
        y_data = dataset.get("y", [])
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=name
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)

def display_forecast_chart(
    historical_data: List[Dict[str, Any]], 
    forecast_data: List[Dict[str, Any]],
    lower_bound: Optional[List[Dict[str, Any]]] = None,
    upper_bound: Optional[List[Dict[str, Any]]] = None,
    title: str = "Forecast", 
    height: int = 400
):
    """Display a forecast chart with historical data and predictions"""
    # Create figure
    fig = go.Figure()
    
    # Extract data
    hist_times = [point.get("time") for point in historical_data]
    hist_values = [point.get("value") for point in historical_data]
    
    forecast_times = [point.get("time") for point in forecast_data]
    forecast_values = [point.get("value") for point in forecast_data]
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=hist_times,
        y=hist_values,
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence interval if provided
    if lower_bound and upper_bound:
        lower_times = [point.get("time") for point in lower_bound]
        lower_values = [point.get("value") for point in lower_bound]
        
        upper_times = [point.get("time") for point in upper_bound]
        upper_values = [point.get("value") for point in upper_bound]
        
        fig.add_trace(go.Scatter(
            x=lower_times + upper_times[::-1],
            y=lower_values + upper_values[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)

def display_system_metrics(metrics: Dict[str, Any], title: str = "System Metrics"):
    """Display system metrics in a clean layout"""
    st.subheader(title)
    
    # Create columns for metrics
    columns = st.columns(3)
    
    # Display each metric in a column
    for i, (key, value) in enumerate(metrics.items()):
        col_idx = i % 3
        with columns[col_idx]:
            if isinstance(value, (int, float)):
                # Format number
                if value > 1000000:
                    formatted_value = f"{value/1000000:.2f}M"
                elif value > 1000:
                    formatted_value = f"{value/1000:.2f}K"
                else:
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                
                st.metric(key, formatted_value)
            else:
                # Just show as text
                st.metric(key, str(value))

def display_json_data(data: Dict[str, Any], expanded: bool = False):
    """Display JSON data in a nice format"""
    st.json(data, expanded=expanded)