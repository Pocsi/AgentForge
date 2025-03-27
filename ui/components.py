import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json

def sidebar():
    """
    Display the application sidebar with navigation and settings
    """
    with st.sidebar:
        st.title("AI Companion")
        
        # Profile section
        st.subheader("Profile")
        if st.session_state.agent:
            st.write(f"**Name:** {st.session_state.agent.name}")
            st.write(f"**Personality:** {st.session_state.agent.personality}")
        else:
            st.write("Agent not initialized")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        # Define the pages
        pages = {
            "conversation": "💬 Conversation",
            "dashboard": "📊 Dashboard",
            "analytics": "📈 Analytics",
            "signal": "📡 Signal Processing",
            "settings": "⚙️ Settings"
        }
        
        # Create navigation buttons
        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}"):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown("---")
        
        # Connection status
        st.subheader("Component Status")
        
        if st.session_state.agent:
            agent_state = st.session_state.agent.get_agent_state()
            components = agent_state.get("components", {})
            
            # Display status of major components
            mcp_status = components.get("mcp", {})
            mcp_connected = mcp_status.get("connected", False)
            st.write(f"MCP: {'✅' if mcp_connected else '❌'}")
            
            goat_status = components.get("goat", {})
            goat_initialized = goat_status.get("initialized", False)
            st.write(f"GOAT: {'✅' if goat_initialized else '❌'}")
            
            # Distributed compute status
            dc_status = components.get("distributed_compute", {})
            dc_engine = dc_status.get("engine", "none")
            if dc_engine != "none":
                st.write(f"Compute: ✅ ({dc_engine})")
            else:
                st.write("Compute: ❌")
            
            # Memory usage
            if "memory_usage" in agent_state:
                memory = agent_state["memory_usage"]
                st.progress(memory["usage_percentage"] / 100, 
                            f"Memory: {memory['current_size']}/{memory['max_size']}")
        
        st.markdown("---")
        
        # Footer
        st.caption("© 2025 AI Companion")
        st.caption("Version 1.0.0")

def header():
    """
    Display the application header
    """
    # Simple clean header
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.current_page == "conversation":
            st.title("AI Companion")
        elif st.session_state.current_page == "dashboard":
            st.title("System Dashboard")
        elif st.session_state.current_page == "analytics":
            st.title("Time Series Analytics")
        elif st.session_state.current_page == "signal":
            st.title("Signal Processing")
        elif st.session_state.current_page == "settings":
            st.title("Settings")
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write(f"Time: {current_time}")

def footer():
    """
    Display the application footer
    """
    st.markdown("---")
    st.caption("AI Companion powered by MCP and GOAT")

def display_chat_message(message, is_user=True):
    """
    Display a chat message with appropriate styling
    
    Args:
        message: Message content
        is_user: Whether the message is from the user (True) or agent (False)
    """
    if is_user:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background-color: #E8F0FE; border-radius: 10px; padding: 10px; max-width: 80%;">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background-color: #F1F3F4; border-radius: 10px; padding: 10px; max-width: 80%;">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_time_series(data, title, height=400):
    """
    Display a time series chart using Plotly
    
    Args:
        data: Time series data
        title: Chart title
        height: Chart height
    """
    # Create figure
    fig = go.Figure()
    
    # Convert data to proper format if needed
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # Extract timestamps and values
        timestamps = [item.get("timestamp") for item in data]
        values = [item.get("value") for item in data]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Value'
        ))
    elif isinstance(data, pd.DataFrame):
        # Use DataFrame directly
        if "timestamp" in data.columns and "value" in data.columns:
            fig.add_trace(go.Scatter(
                x=data["timestamp"],
                y=data["value"],
                mode='lines',
                name='Value'
            ))
        elif data.index.name == "timestamp" or isinstance(data.index, pd.DatetimeIndex):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.iloc[:, 0],
                mode='lines',
                name=data.columns[0]
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def display_frequency_spectrum(data, title, height=400):
    """
    Display a frequency spectrum chart using Plotly
    
    Args:
        data: Spectrum data
        title: Chart title
        height: Chart height
    """
    # Create figure
    fig = go.Figure()
    
    # Convert data to proper format if needed
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # Extract frequencies and amplitudes
        frequencies = [item.get("frequency") for item in data]
        amplitudes = [item.get("amplitude") for item in data]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=amplitudes,
            mode='lines',
            name='Amplitude'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        template="plotly_white"
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def display_comparison_chart(original_data, processed_data, title, height=400):
    """
    Display a comparison chart of original and processed data
    
    Args:
        original_data: Original data
        processed_data: Processed data
        title: Chart title
        height: Chart height
    """
    # Create figure
    fig = go.Figure()
    
    # Convert data to proper format if needed
    if (isinstance(original_data, list) and len(original_data) > 0 and
        isinstance(original_data[0], dict) and
        isinstance(processed_data, list) and len(processed_data) > 0 and
        isinstance(processed_data[0], dict)):
        
        # Extract time and values
        original_times = [item.get("time") for item in original_data]
        original_values = [item.get("value") for item in original_data]
        
        processed_times = [item.get("time") for item in processed_data]
        processed_values = [item.get("value") for item in processed_data]
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=original_times,
            y=original_values,
            mode='lines',
            name='Original'
        ))
        
        fig.add_trace(go.Scatter(
            x=processed_times,
            y=processed_values,
            mode='lines',
            name='Processed',
            line=dict(color='red')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def display_forecast_chart(historical_data, forecast_data, methods, title, height=500):
    """
    Display a forecast chart with historical data and predictions
    
    Args:
        historical_data: Historical time series data
        forecast_data: Dictionary of forecast data by method
        methods: List of forecast methods to display
        title: Chart title
        height: Chart height
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    if isinstance(historical_data, list) and len(historical_data) > 0:
        timestamps = [item.get("timestamp") for item in historical_data]
        values = [item.get("value") for item in historical_data]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
    
    # Add forecast data for each method
    colors = ['red', 'green', 'orange', 'purple', 'teal']
    
    for i, method in enumerate(methods):
        if method in forecast_data:
            method_data = forecast_data[method]
            
            # Extract data
            timestamps = [item.get("timestamp") for item in method_data]
            values = [item.get("value") for item in method_data]
            lower_bounds = [item.get("lower_bound") for item in method_data]
            upper_bounds = [item.get("upper_bound") for item in method_data]
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=f'{method.capitalize()}',
                line=dict(color=colors[i % len(colors)])
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=timestamps + timestamps[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor=colors[i % len(colors)],
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.2
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def display_system_metrics(metrics):
    """
    Display system metrics in a clean format
    
    Args:
        metrics: Dictionary of system metrics
    """
    cols = st.columns(len(metrics))
    
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label=label, value=value)

def display_json_data(data, title="Data", expanded=False):
    """
    Display JSON data in an expandable section
    
    Args:
        data: Data to display
        title: Section title
        expanded: Whether to show expanded by default
    """
    with st.expander(title, expanded=expanded):
        st.json(data)
