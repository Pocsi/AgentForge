import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from ui.components import (
    display_chat_message, 
    display_time_series, 
    display_frequency_spectrum,
    display_comparison_chart,
    display_forecast_chart,
    display_system_metrics,
    display_json_data
)

# Add the missing pages for signal processing and settings

def conversation_page():
    """
    Render the main conversation page
    """
    st.subheader("Chat with your AI Companion")
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(
                message["content"],
                is_user=(message["role"] == "user")
            )
    
    # Input for new message
    with st.form(key="message_form", clear_on_submit=True):
        user_input = st.text_area("Message:", height=100, key="user_input")
        cols = st.columns([1, 1, 6])
        with cols[0]:
            submit_button = st.form_submit_button("Send")
        with cols[1]:
            clear_button = st.form_submit_button("Clear Chat")
    
    # Process the message when submitted
    if submit_button and user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            if st.session_state.agent:
                # Process with the agent
                response, metadata = st.session_state.agent.process_message(user_input)
                
                # Add agent response to chat history
                st.session_state.chat_history.append({
                    "role": "agent",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata
                })
            else:
                # Fallback if agent not initialized
                time.sleep(1)
                st.session_state.chat_history.append({
                    "role": "agent",
                    "content": "I'm sorry, but the agent hasn't been properly initialized yet. Please check the system status in the settings page.",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Force refresh to show new messages
        st.rerun()
    
    # Clear chat history if requested
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

def dashboard_page():
    """
    Render the system dashboard page
    """
    st.subheader("System Dashboard")
    
    # Check if agent is initialized
    if not st.session_state.agent:
        st.warning("Agent not initialized. Please check the settings.")
        return
    
    # Display system metrics
    agent_state = st.session_state.agent.get_agent_state()
    
    # Layout with 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Agent Status")
        
        # Basic agent info
        st.write(f"**Name:** {agent_state.get('name', 'Unknown')}")
        st.write(f"**Personality:** {agent_state.get('personality', 'Unknown')}")
        
        # Memory usage
        memory_usage = agent_state.get("memory_usage", {})
        if memory_usage:
            st.write("**Memory Usage:**")
            st.progress(
                memory_usage.get("usage_percentage", 0) / 100,
                text=f"{memory_usage.get('current_size', 0)}/{memory_usage.get('max_size', 0)} messages"
            )
            
            # Last compression time
            last_compression = memory_usage.get("last_compression")
            if last_compression:
                st.write(f"**Last Compression:** {last_compression}")
    
    with col2:
        st.subheader("Integrations")
        
        # MCP Status
        components = agent_state.get("components", {})
        mcp_status = components.get("mcp", {})
        
        st.write("**MCP Connector:**")
        if mcp_status.get("connected", False):
            st.success("Connected to MCP")
            st.write(f"Server: {mcp_status.get('server_url', 'Unknown')}")
        else:
            st.error("Not connected to MCP")
            if mcp_status.get("last_error"):
                st.write(f"Error: {mcp_status.get('last_error')}")
        
        # GOAT Status
        goat_status = components.get("goat", {})
        
        st.write("**GOAT Connector:**")
        if goat_status.get("initialized", False):
            st.success("GOAT SDK Initialized")
            if goat_status.get("wallet_connected", False):
                st.write("Wallet connected")
            
            # Available tools
            if "available_tools" in goat_status:
                tools = goat_status.get("available_tools", [])
                if tools:
                    st.write(f"Available tools: {', '.join(tools)}")
        else:
            st.error("GOAT SDK Not Initialized")
            if goat_status.get("last_error"):
                st.write(f"Error: {goat_status.get('last_error')}")
    
    with col3:
        st.subheader("Computational Resources")
        
        # Distributed Computing Status
        dc_status = components.get("distributed_compute", {})
        
        st.write("**Distributed Computing:**")
        if dc_status.get("enabled", False) and dc_status.get("engine", "none") != "none":
            st.success(f"Enabled ({dc_status.get('engine', 'Unknown').capitalize()})")
            
            # Worker information
            if "connected_workers" in dc_status:
                st.write(f"Workers: {dc_status.get('connected_workers', 0)}")
            else:
                st.write(f"Workers: {dc_status.get('workers', 0)}")
            
            # Memory usage if available
            if "total_memory" in dc_status and "used_memory" in dc_status:
                total_gb = dc_status.get("total_memory", 0) / 1e9
                used_gb = dc_status.get("used_memory", 0) / 1e9
                memory_percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0
                
                st.progress(
                    memory_percent / 100,
                    text=f"Memory: {used_gb:.1f}GB / {total_gb:.1f}GB"
                )
        else:
            st.warning("Local execution only")
    
    # Horizontal line
    st.markdown("---")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Use most recent messages from chat history
    if "chat_history" in st.session_state and st.session_state.chat_history:
        recent_messages = st.session_state.chat_history[-5:]
        
        for message in recent_messages:
            timestamp = datetime.fromisoformat(message.get("timestamp", datetime.now().isoformat()))
            formatted_time = timestamp.strftime("%H:%M:%S")
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."
            
            st.write(f"**{formatted_time}** - *{role.capitalize()}*: {content}")
    else:
        st.write("No recent activity")
    
    # Horizontal line
    st.markdown("---")
    
    # IoT Devices Section
    st.subheader("Connected IoT Devices")
    
    # Check if signal processor is initialized and has IoT manager
    signal_status = components.get("signal_processor", {})
    if signal_status.get("iot_enabled", False):
        iot_status = signal_status.get("iot", {})
        
        if iot_status:
            device_count = iot_status.get("device_count", 0)
            connected_devices = iot_status.get("connected_devices", 0)
            
            if device_count > 0:
                st.write(f"{connected_devices} connected out of {device_count} known devices")
                
                # We could show more device details here if available
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Devices", device_count)
                
                with col2:
                    st.metric("Connected", connected_devices)
            else:
                st.write("No IoT devices detected")
                
                # Add a refresh button
                if st.button("Discover Devices"):
                    st.session_state.signal_processor._force_iot_discovery()
                    st.success("Device discovery initiated")
        else:
            st.write("IoT management not available")
    else:
        st.write("IoT functionality is disabled")

def analytics_page():
    """
    Render the time series analytics page
    """
    st.subheader("Time Series Analytics & Forecasting")
    
    # Check if required components are initialized
    if not st.session_state.time_series_analyzer:
        st.warning("Time series analyzer not initialized. Please check the settings.")
        return
    
    if not st.session_state.forecaster:
        st.warning("Forecaster not initialized. Please check the settings.")
        return
    
    # Tab selection
    tab1, tab2 = st.tabs(["Time Series Analysis", "Forecasting"])
    
    # Time Series Analysis Tab
    with tab1:
        st.subheader("Analyze Time Series Data")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Trend Analysis", "Seasonal Decomposition", "Anomaly Detection", "Feature Extraction"],
            key="analysis_type"
        )
        
        # Input for time series data source
        source_type = st.radio(
            "Data Source:",
            ["Sample Data", "Upload CSV", "Database"],
            horizontal=True,
            key="ts_source_type"
        )
        
        # Parameters based on source type
        if source_type == "Sample Data":
            st.write("Using generated sample data")
            
            # Sample data parameters
            col1, col2 = st.columns(2)
            with col1:
                duration = st.slider("Duration (days):", 7, 90, 30, key="ts_duration")
            with col2:
                pattern = st.selectbox(
                    "Pattern:",
                    ["Trending", "Seasonal", "Cyclical", "Random"],
                    key="ts_pattern"
                )
            
        elif source_type == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file with time series data", type="csv")
            if uploaded_file is not None:
                st.success("File uploaded successfully")
            else:
                st.info("Please upload a CSV file with time series data")
                
        elif source_type == "Database":
            st.write("Database connection")
            col1, col2 = st.columns(2)
            with col1:
                db_name = st.text_input("Database Name:", key="ts_db_name")
            with col2:
                table_name = st.text_input("Table/Collection:", key="ts_table_name")
        
        # Field selection
        field_name = st.text_input("Field/Metric Name:", "value", key="ts_field")
        
        # Analysis button
        if st.button("Analyze", key="analyze_button"):
            with st.spinner("Analyzing data..."):
                # Build analysis query
                if analysis_type == "Trend Analysis":
                    query = f"analyze trend for {field_name}"
                elif analysis_type == "Seasonal Decomposition":
                    query = f"decompose time series for {field_name}"
                elif analysis_type == "Anomaly Detection":
                    query = f"detect anomalies in {field_name}"
                elif analysis_type == "Feature Extraction":
                    query = f"extract features from {field_name}"
                
                # Add source details
                if source_type == "Sample Data":
                    query += f" using sample data for last {duration} days with {pattern.lower()} pattern"
                elif source_type == "Upload CSV":
                    query += " from csv file"
                elif source_type == "Database":
                    query += f" from database {db_name} table {table_name}"
                
                # Execute analysis
                result = st.session_state.time_series_analyzer.analyze(query)
                
                # Display results
                if result.get("status") == "success":
                    st.success("Analysis completed successfully")
                    
                    # Display based on analysis type
                    if analysis_type == "Trend Analysis" and "time_series" in result:
                        # Display trend analysis results
                        display_time_series(
                            result["time_series"],
                            f"Trend Analysis: {field_name}",
                            height=400
                        )
                        
                        # Display trend statistics
                        if "trend" in result:
                            trend_data = result["trend"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Trend Direction", trend_data.get("direction", "Unknown"))
                            with col2:
                                st.metric("Start Value", f"{trend_data.get('start_value', 0):.2f}")
                            with col3:
                                st.metric("End Value", f"{trend_data.get('end_value', 0):.2f}")
                            
                            # Percent change
                            st.metric("Percent Change", f"{trend_data.get('percent_change', 0):.2f}%")
                        
                        # Display statistics
                        if "stats" in result:
                            stats = result["stats"]
                            st.subheader("Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats.get('mean', 0):.2f}")
                            with col2:
                                st.metric("Std Dev", f"{stats.get('std', 0):.2f}")
                            with col3:
                                st.metric("Min", f"{stats.get('min', 0):.2f}")
                            with col4:
                                st.metric("Max", f"{stats.get('max', 0):.2f}")
                    
                    elif analysis_type == "Seasonal Decomposition" and "components" in result:
                        # Display decomposition components
                        components = result["components"]
                        
                        # Original data
                        if "trend" in components:
                            display_time_series(
                                components["trend"],
                                "Trend Component",
                                height=250
                            )
                        
                        # Seasonal component
                        if "seasonal" in components:
                            display_time_series(
                                components["seasonal"],
                                "Seasonal Component",
                                height=250
                            )
                        
                        # Residual component
                        if "resid" in components:
                            display_time_series(
                                components["resid"],
                                "Residual Component",
                                height=250
                            )
                        
                        # Method and period info
                        st.info(f"Decomposition method: {result.get('method', 'Unknown')}, Period: {result.get('period', 'Unknown')}")
                    
                    elif analysis_type == "Anomaly Detection" and "time_series" in result:
                        # Display anomaly detection results
                        display_time_series(
                            result["time_series"],
                            f"Anomaly Detection: {field_name}",
                            height=400
                        )
                        
                        # Anomaly statistics
                        if "anomalies" in result:
                            anomalies = result["anomalies"]
                            st.subheader("Anomalies")
                            st.metric("Anomalies Detected", anomalies.get("count", 0))
                            
                            # Display anomaly list
                            if "items" in anomalies and anomalies["items"]:
                                st.write("Anomaly details:")
                                anomaly_df = pd.DataFrame(anomalies["items"])
                                st.dataframe(anomaly_df)
                        
                        # Display statistics
                        if "stats" in result:
                            stats = result["stats"]
                            st.subheader("Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean", f"{stats.get('mean', 0):.2f}")
                            with col2:
                                st.metric("Std Dev", f"{stats.get('std', 0):.2f}")
                            with col3:
                                st.metric("Threshold", f"{stats.get('threshold', 0):.2f} σ")
                    
                    elif analysis_type == "Feature Extraction" and "features" in result:
                        # Display extracted features
                        st.subheader("Extracted Features")
                        
                        # Create dataframe from features
                        features = result["features"]
                        feature_df = pd.DataFrame(features)
                        
                        # Display features table
                        st.dataframe(feature_df, use_container_width=True)
                        
                        # Display feature count
                        st.metric("Total Features", result.get("count", 0))
                        
                        # Note about extraction method
                        if "note" in result:
                            st.info(result["note"])
                    
                    # Raw data viewer
                    with st.expander("View raw analysis data"):
                        st.json(result)
                else:
                    st.error(f"Analysis failed: {result.get('message', 'Unknown error')}")
    
    # Forecasting Tab
    with tab2:
        st.subheader("Generate Forecasts")
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Forecast Horizon (days):", 1, 90, 30, key="fc_horizon")
        with col2:
            confidence = st.slider("Confidence Interval:", 0.7, 0.99, 0.95, step=0.01, key="fc_confidence")
        
        # Forecast method selection
        methods = st.multiselect(
            "Forecast Methods:",
            ["ARIMA", "Prophet", "LSTM", "Ensemble"],
            default=["ARIMA", "Prophet", "Ensemble"],
            key="fc_methods"
        )
        
        # Input for time series data source
        source_type = st.radio(
            "Data Source:",
            ["Sample Data", "Upload CSV", "Database"],
            horizontal=True,
            key="fc_source_type"
        )
        
        # Parameters based on source type
        if source_type == "Sample Data":
            st.write("Using generated sample data")
            
            # Sample data parameters
            col1, col2 = st.columns(2)
            with col1:
                history_duration = st.slider("Historical Data (days):", 30, 365, 90, key="fc_history")
            with col2:
                pattern = st.selectbox(
                    "Pattern:",
                    ["Trending", "Seasonal", "Cyclical", "Random"],
                    key="fc_pattern"
                )
            
        elif source_type == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file with time series data", type="csv", key="fc_upload")
            if uploaded_file is not None:
                st.success("File uploaded successfully")
            else:
                st.info("Please upload a CSV file with time series data")
                
        elif source_type == "Database":
            st.write("Database connection")
            col1, col2 = st.columns(2)
            with col1:
                db_name = st.text_input("Database Name:", key="fc_db_name")
            with col2:
                table_name = st.text_input("Table/Collection:", key="fc_table_name")
        
        # Field selection
        field_name = st.text_input("Field/Metric Name:", "value", key="fc_field")
        
        # Forecast button
        if st.button("Generate Forecast", key="forecast_button"):
            with st.spinner("Generating forecast..."):
                # Build forecast query
                query = f"predict next {horizon} days for {field_name}"
                
                # Add methods
                methods_str = ", ".join([m.lower() for m in methods])
                query += f" using {methods_str}"
                
                # Add source details
                if source_type == "Sample Data":
                    query += f" with {history_duration} days of historical {pattern.lower()} data"
                elif source_type == "Upload CSV":
                    query += " from csv file"
                elif source_type == "Database":
                    query += f" from database {db_name} table {table_name}"
                
                # Execute forecast
                result = st.session_state.forecaster.forecast(query)
                
                # Display results
                if result.get("status") == "success":
                    st.success("Forecast generated successfully")
                    
                    # Display forecast chart
                    if "historical_data" in result and "forecast_data" in result:
                        display_forecast_chart(
                            result["historical_data"],
                            result["forecast_data"],
                            result.get("methods", []),
                            f"Forecast for {field_name} - Next {horizon} Days",
                            height=500
                        )
                        
                        # Forecast statistics
                        st.subheader("Forecast Details")
                        st.write(f"Field: {result.get('field', 'Unknown')}")
                        st.write(f"Horizon: {result.get('horizon', 0)} days")
                        st.write(f"Confidence Interval: {result.get('confidence_interval', 0) * 100:.0f}%")
                        st.write(f"Methods: {', '.join(result.get('methods', []))}")
                        
                        # Raw data viewer
                        with st.expander("View raw forecast data"):
                            st.json(result)
                else:
                    st.error(f"Forecast failed: {result.get('message', 'Unknown error')}")

def signal_page():
    """
    Render the signal processing page
    """
    st.subheader("Signal Processing")
    
    # Check if signal processor is initialized
    if not st.session_state.signal_processor:
        st.warning("Signal processor not initialized. Please check the settings.")
        return
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["Signal Processing", "FFT Analysis", "IoT Devices"])
    
    # Signal Processing Tab
    with tab1:
        st.subheader("Process Signal Data")
        
        # Processing type selection
        processing_type = st.selectbox(
            "Select Processing Type:",
            ["Filtering", "Peak Detection", "Wavelet Transform", "Compression"],
            key="proc_type"
        )
        
        # Input for signal data source
        source_type = st.radio(
            "Data Source:",
            ["Sample Signal", "IoT Device", "Upload File"],
            horizontal=True,
            key="sig_source_type"
        )
        
        # Parameters based on source type
        if source_type == "Sample Signal":
            st.write("Using generated sample signal")
            
            # Sample signal parameters
            col1, col2 = st.columns(2)
            with col1:
                frequency = st.slider("Base Frequency (Hz):", 1, 50, 10, key="sig_frequency")
            with col2:
                duration = st.slider("Duration (seconds):", 1, 30, 5, key="sig_duration")
            
        elif source_type == "IoT Device":
            # Get devices from session state if available
            devices = []
            if st.session_state.signal_processor and hasattr(st.session_state.signal_processor, "iot_manager"):
                try:
                    devices = st.session_state.signal_processor.iot_manager.list_devices()
                except:
                    devices = []
            
            if devices:
                device_options = [f"{d['name']} ({d['id']})" for d in devices]
                selected_device = st.selectbox(
                    "Select IoT Device:",
                    device_options,
                    key="sig_device"
                )
                # Extract device ID from selection
                selected_device_id = selected_device.split("(")[-1].split(")")[0]
            else:
                st.info("No IoT devices available")
                if st.button("Discover Devices"):
                    with st.spinner("Discovering devices..."):
                        try:
                            devices = st.session_state.signal_processor.iot_manager.discover_devices()
                            st.success(f"Discovered {len(devices)} new devices")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error discovering devices: {str(e)}")
                selected_device_id = None
                
        elif source_type == "Upload File":
            uploaded_file = st.file_uploader("Upload a signal data file", type=["csv", "json", "txt"])
            if uploaded_file is not None:
                st.success("File uploaded successfully")
            else:
                st.info("Please upload a signal data file")
        
        # Processing parameters based on type
        if processing_type == "Filtering":
            st.subheader("Filter Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.selectbox(
                    "Filter Type:",
                    ["Lowpass", "Highpass", "Bandpass", "Notch"],
                    key="filter_type"
                )
            with col2:
                cutoff_freq = st.slider(
                    "Cutoff Frequency (Hz):",
                    0.1, 100.0, 20.0,
                    key="cutoff_freq"
                )
                
            # For bandpass filter, need high cutoff
            if filter_type == "Bandpass":
                high_cutoff = st.slider(
                    "High Cutoff Frequency (Hz):",
                    cutoff_freq + 0.1, 100.0, min(cutoff_freq * 2, 100.0),
                    key="high_cutoff"
                )
            
            # For notch filter, need Q factor
            if filter_type == "Notch":
                q_factor = st.slider(
                    "Q Factor:",
                    1.0, 100.0, 30.0,
                    key="q_factor"
                )
                
        elif processing_type == "Peak Detection":
            st.subheader("Peak Detection Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Detection Threshold:",
                    0.1, 5.0, 1.0,
                    key="peak_threshold"
                )
            with col2:
                min_distance = st.slider(
                    "Minimum Distance (samples):",
                    1, 100, 30,
                    key="min_distance"
                )
                
        elif processing_type == "Wavelet Transform":
            st.subheader("Wavelet Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                wavelet_type = st.selectbox(
                    "Wavelet Type:",
                    ["db4", "haar", "sym5", "coif3", "bior2.2"],
                    key="wavelet_type"
                )
            with col2:
                level = st.slider(
                    "Decomposition Level:",
                    1, 10, 5,
                    key="decomp_level"
                )
                
        elif processing_type == "Compression":
            st.subheader("Compression Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                compression_method = st.selectbox(
                    "Compression Method:",
                    ["Threshold", "Wavelet", "Sampling"],
                    key="compression_method"
                )
            with col2:
                threshold = st.slider(
                    "Threshold:",
                    0.01, 0.5, 0.1,
                    key="compression_threshold"
                )
        
        # Process button
        if st.button("Process Signal", key="process_button"):
            with st.spinner("Processing signal..."):
                # Build processing query
                if processing_type == "Filtering":
                    query = f"filter signal using {filter_type.lower()} filter at {cutoff_freq} hz"
                    
                    # Add filter-specific parameters
                    if filter_type == "Bandpass":
                        query += f" to {high_cutoff} hz"
                    elif filter_type == "Notch":
                        query += f" with q factor {q_factor}"
                        
                elif processing_type == "Peak Detection":
                    query = f"detect peaks in signal with threshold {threshold}"
                    query += f" and minimum distance {min_distance}"
                    
                elif processing_type == "Wavelet Transform":
                    query = f"wavelet transform signal using {wavelet_type} with level {level}"
                    
                elif processing_type == "Compression":
                    query = f"compress signal using {compression_method.lower()} method with threshold {threshold}"
                
                # Add source details
                if source_type == "Sample Signal":
                    query += f" using sample signal at {frequency} hz for {duration} seconds"
                elif source_type == "IoT Device" and selected_device_id:
                    query += f" from device {selected_device_id}"
                elif source_type == "Upload File":
                    query += " from file"
                
                # Execute processing
                result = st.session_state.signal_processor.process(query)
                
                # Display results
                if result.get("status") == "success":
                    st.success("Signal processing completed successfully")
                    
                    # Display based on processing type
                    if processing_type == "Filtering" and "filtered_signal" in result:
                        # Display filtering results
                        display_comparison_chart(
                            result["original_signal"],
                            result["filtered_signal"],
                            f"Filtering Results: {result.get('filter_type', 'Unknown')}",
                            height=400
                        )
                        
                        # Filter info
                        st.info(f"Filter: {result.get('filter_type', 'Unknown')}, Cutoff: {result.get('cutoff_frequency', 0)} Hz")
                        
                        # Filter statistics
                        if "stats" in result:
                            stats = result["stats"]
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Original Mean", f"{stats.get('original_mean', 0):.2f}")
                            with col2:
                                st.metric("Filtered Mean", f"{stats.get('filtered_mean', 0):.2f}")
                            with col3:
                                st.metric("Original Std", f"{stats.get('original_std', 0):.2f}")
                            with col4:
                                st.metric("Filtered Std", f"{stats.get('filtered_std', 0):.2f}")
                    
                    elif processing_type == "Peak Detection" and "signal" in result:
                        # Display peak detection results
                        display_time_series(
                            result["signal"],
                            "Signal with Detected Peaks",
                            height=400
                        )
                        
                        # Peak statistics
                        st.subheader("Detected Peaks")
                        st.metric("Peak Count", result.get("peak_count", 0))
                        
                        # Display peak list
                        if "peaks" in result and result["peaks"]:
                            peak_df = pd.DataFrame(result["peaks"])
                            st.dataframe(peak_df)
                            
                            # Peak statistics
                            if "stats" in result:
                                stats = result["stats"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Peak Height", f"{stats.get('mean_peak_height', 0):.2f}")
                                with col2:
                                    st.metric("Max Peak Height", f"{stats.get('max_peak_height', 0):.2f}")
                                with col3:
                                    st.metric("Mean Peak Distance", f"{stats.get('mean_peak_distance', 0):.4f} s")
                    
                    elif processing_type == "Wavelet Transform":
                        # Display wavelet transform results
                        st.subheader(f"Wavelet Transform: {result.get('wavelet_type', 'Unknown')}")
                        
                        # Display approximation
                        if "approximation" in result:
                            st.write("Approximation Coefficients")
                            approx_df = pd.DataFrame(result["approximation"])
                            
                            # Create a simple line chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=approx_df["index"],
                                y=approx_df["value"],
                                mode='lines',
                                name='Approximation'
                            ))
                            fig.update_layout(
                                height=300,
                                margin=dict(l=10, r=10, t=40, b=10),
                                title="Approximation Coefficients",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display details for each level
                        if "details" in result:
                            st.write("Detail Coefficients by Level")
                            
                            # Create tabs for each level
                            detail_tabs = st.tabs([f"Level {d['level']}" for d in result["details"]])
                            
                            for i, detail in enumerate(result["details"]):
                                with detail_tabs[i]:
                                    detail_df = pd.DataFrame(detail["data"])
                                    
                                    # Create a simple line chart
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=detail_df["index"],
                                        y=detail_df["value"],
                                        mode='lines',
                                        name=f'Level {detail["level"]}'
                                    ))
                                    fig.update_layout(
                                        height=250,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        title=f"Detail Coefficients - Level {detail['level']}",
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Wavelet energy statistics
                        if "stats" in result:
                            stats = result["stats"]
                            st.subheader("Energy Distribution")
                            
                            # Create energy bar chart
                            energy_data = {
                                "Component": ["Approximation"] + [f"Detail {i+1}" for i in range(len(stats.get("details_energy", [])))],
                                "Energy": [stats.get("approximation_energy", 0)] + stats.get("details_energy", [])
                            }
                            energy_df = pd.DataFrame(energy_data)
                            
                            fig = go.Figure(go.Bar(
                                x=energy_df["Component"],
                                y=energy_df["Energy"],
                                marker_color='blue'
                            ))
                            fig.update_layout(
                                height=300,
                                title="Energy by Component",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif processing_type == "Compression" and "reconstructed_signal" in result:
                        # Display compression results
                        display_comparison_chart(
                            result["original_signal"],
                            result["reconstructed_signal"],
                            f"Compression Results: {result.get('method', 'Unknown').capitalize()}",
                            height=400
                        )
                        
                        # Compression statistics
                        st.subheader("Compression Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Size", result.get("original_size", 0))
                        with col2:
                            st.metric("Compressed Size", result.get("compressed_size", 0))
                        with col3:
                            st.metric("Compression Ratio", f"{result.get('compression_ratio', 1):.2f}x")
                        
                        # Quality metrics
                        if "stats" in result:
                            stats = result["stats"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Square Error", f"{stats.get('mse', 0):.6f}")
                            with col2:
                                st.metric("Max Error", f"{stats.get('max_error', 0):.6f}")
                            with col3:
                                st.metric("SNR (dB)", f"{stats.get('snr', 0):.2f}")
                    
                    # Raw data viewer
                    with st.expander("View raw processing data"):
                        st.json(result)
                else:
                    st.error(f"Processing failed: {result.get('message', 'Unknown error')}")
    
    # FFT Analysis Tab
    with tab2:
        st.subheader("Frequency Domain Analysis")
        
        # Input for signal data source
        source_type = st.radio(
            "Data Source:",
            ["Sample Signal", "IoT Device", "Upload File"],
            horizontal=True,
            key="fft_source_type"
        )
        
        # Parameters based on source type
        if source_type == "Sample Signal":
            st.write("Using generated sample signal")
            
            # Sample signal parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                base_freq = st.slider("Base Frequency (Hz):", 1, 50, 10, key="fft_base_freq")
            with col2:
                secondary_freq = st.slider("Secondary Frequency (Hz):", 1, 100, 25, key="fft_secondary_freq")
            with col3:
                duration = st.slider("Duration (seconds):", 1, 30, 5, key="fft_duration")
            
            # Additional parameter for noise
            noise_level = st.slider("Noise Level:", 0.0, 1.0, 0.1, key="fft_noise")
            
        elif source_type == "IoT Device":
            # Get devices from session state if available
            devices = []
            if st.session_state.signal_processor and hasattr(st.session_state.signal_processor, "iot_manager"):
                try:
                    devices = st.session_state.signal_processor.iot_manager.list_devices()
                except:
                    devices = []
            
            if devices:
                device_options = [f"{d['name']} ({d['id']})" for d in devices]
                selected_device = st.selectbox(
                    "Select IoT Device:",
                    device_options,
                    key="fft_device"
                )
                # Extract device ID from selection
                selected_device_id = selected_device.split("(")[-1].split(")")[0]
            else:
                st.info("No IoT devices available")
                if st.button("Discover Devices", key="fft_discover"):
                    with st.spinner("Discovering devices..."):
                        try:
                            devices = st.session_state.signal_processor.iot_manager.discover_devices()
                            st.success(f"Discovered {len(devices)} new devices")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error discovering devices: {str(e)}")
                selected_device_id = None
                
        elif source_type == "Upload File":
            uploaded_file = st.file_uploader("Upload a signal data file", type=["csv", "json", "txt"], key="fft_upload")
            if uploaded_file is not None:
                st.success("File uploaded successfully")
            else:
                st.info("Please upload a signal data file")
        
        # FFT Parameters
        st.subheader("FFT Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            sampling_rate = st.slider("Sampling Rate (Hz):", 100, 10000, 1000, key="fft_rate")
        with col2:
            window_function = st.selectbox(
                "Window Function:",
                ["None", "Hanning", "Hamming", "Blackman"],
                key="fft_window"
            )
        
        # Analyze button
        if st.button("Analyze Frequency Spectrum", key="fft_button"):
            with st.spinner("Analyzing frequency spectrum..."):
                # Build FFT query
                query = f"analyze frequency spectrum with sampling rate {sampling_rate} hz"
                
                # Add window function
                if window_function != "None":
                    query += f" using {window_function.lower()} window"
                
                # Add source details
                if source_type == "Sample Signal":
                    query += f" for sample signal with base frequency {base_freq} hz"
                    query += f" and secondary frequency {secondary_freq} hz"
                    query += f" with {duration} seconds duration and {noise_level} noise level"
                elif source_type == "IoT Device" and selected_device_id:
                    query += f" from device {selected_device_id}"
                elif source_type == "Upload File":
                    query += " from file"
                
                # Execute FFT analysis
                result = st.session_state.signal_processor.process(query)
                
                # Display results
                if result.get("status") == "success" and result.get("processing_type") == "fft":
                    st.success("Frequency analysis completed successfully")
                    
                    # Display frequency spectrum
                    if "spectrum" in result:
                        display_frequency_spectrum(
                            result["spectrum"],
                            "Frequency Spectrum Analysis",
                            height=400
                        )
                        
                        # Info on sampling rate
                        st.info(f"Sampling Rate: {result.get('sampling_rate', 0)} Hz")
                        
                        # Dominant frequencies
                        if "dominant_frequencies" in result:
                            st.subheader("Dominant Frequencies")
                            dom_freq_df = pd.DataFrame(result["dominant_frequencies"])
                            st.dataframe(dom_freq_df)
                        
                        # Spectrum statistics
                        if "stats" in result:
                            stats = result["stats"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Max Amplitude", f"{stats.get('max_amplitude', 0):.4f}")
                            with col2:
                                st.metric("Mean Amplitude", f"{stats.get('mean_amplitude', 0):.4f}")
                            with col3:
                                st.metric("Signal Power", f"{stats.get('signal_power', 0):.4f}")
                    
                    # Raw data viewer
                    with st.expander("View raw FFT data"):
                        st.json(result)
                else:
                    st.error(f"FFT analysis failed: {result.get('message', 'Unknown error')}")
    
    # IoT Devices Tab
    with tab3:
        st.subheader("IoT Device Management")
        
        # Check if IoT manager is available
        iot_available = (st.session_state.signal_processor and 
                        hasattr(st.session_state.signal_processor, "iot_manager") and
                        st.session_state.signal_processor.iot_manager is not None)
        
        if not iot_available:
            st.warning("IoT functionality is not available or enabled.")
        else:
            # Get IoT status
            try:
                iot_status = st.session_state.signal_processor.iot_manager.get_status()
                
                # Display status info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Devices", iot_status.get("device_count", 0))
                with col2:
                    st.metric("Connected Devices", iot_status.get("connected_devices", 0))
                with col3:
                    last_discovery = iot_status.get("last_discovery", "Never")
                    if isinstance(last_discovery, str) and last_discovery != "Never":
                        try:
                            discovery_time = datetime.fromisoformat(last_discovery)
                            last_discovery = discovery_time.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    st.metric("Last Discovery", last_discovery)
                
                # Device discovery button
                if st.button("Discover Devices", key="device_discovery"):
                    with st.spinner("Discovering devices..."):
                        try:
                            new_devices = st.session_state.signal_processor.iot_manager.discover_devices()
                            st.success(f"Discovered {len(new_devices)} new devices")
                        except Exception as e:
                            st.error(f"Error discovering devices: {str(e)}")
                
                # Display device list
                st.subheader("Available Devices")
                
                try:
                    devices = st.session_state.signal_processor.iot_manager.list_devices()
                    
                    if not devices:
                        st.info("No devices found. Click 'Discover Devices' to search for available devices.")
                    else:
                        # Create a dataframe for display
                        device_data = []
                        for device in devices:
                            device_data.append({
                                "ID": device.get("id", "Unknown"),
                                "Name": device.get("name", "Unknown"),
                                "Type": device.get("type", "Unknown"),
                                "Status": device.get("status", "Unknown"),
                                "Last Seen": device.get("last_seen", "Never"),
                                "Capabilities": ", ".join(device.get("capabilities", []))
                            })
                        
                        device_df = pd.DataFrame(device_data)
                        st.dataframe(device_df, use_container_width=True)
                        
                        # Device selection for details and commands
                        selected_device_id = st.selectbox(
                            "Select a device for details and commands:",
                            [d["ID"] for d in device_data],
                            key="device_select"
                        )
                        
                        if selected_device_id:
                            # Get device details
                            device = next((d for d in devices if d.get("id") == selected_device_id), None)
                            
                            if device:
                                # Display device details
                                st.subheader(f"Device Details: {device.get('name', 'Unknown')}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**ID:** {device.get('id', 'Unknown')}")
                                    st.write(f"**Type:** {device.get('type', 'Unknown')}")
                                    st.write(f"**Protocol:** {device.get('protocol', 'Unknown')}")
                                with col2:
                                    st.write(f"**Status:** {device.get('status', 'Unknown')}")
                                    st.write(f"**Address:** {device.get('address', 'Unknown')}")
                                    st.write(f"**Capabilities:** {', '.join(device.get('capabilities', []))}")
                                
                                # Command section
                                st.subheader("Send Command")
                                
                                # Command selection based on capabilities
                                capabilities = device.get("capabilities", [])
                                
                                if "temperature" in capabilities:
                                    commands = ["Read Temperature", "Set Alert Threshold"]
                                elif "vibration" in capabilities:
                                    commands = ["Read Vibration", "Set Sensitivity"]
                                elif "voltage" in capabilities:
                                    commands = ["Read Voltage", "Set Measurement Mode"]
                                else:
                                    commands = ["Read Data", "Configure Device"]
                                
                                command = st.selectbox("Command:", commands, key="device_command")
                                
                                # Parameter input based on command
                                parameters = {}
                                if command.startswith("Set"):
                                    param_value = st.slider("Value:", 0.0, 100.0, 50.0, key="command_param")
                                    parameters["value"] = param_value
                                
                                # Send command button
                                if st.button("Send Command", key="send_command"):
                                    with st.spinner(f"Sending {command} to {device.get('name', 'Unknown')}..."):
                                        try:
                                            result = st.session_state.signal_processor.iot_manager.send_command(
                                                selected_device_id,
                                                command,
                                                parameters
                                            )
                                            
                                            if result.get("status") == "success":
                                                st.success(f"Command sent successfully!")
                                                st.json(result)
                                            else:
                                                st.error(f"Command failed: {result.get('message', 'Unknown error')}")
                                        except Exception as e:
                                            st.error(f"Error sending command: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error retrieving devices: {str(e)}")
            
            except Exception as e:
                st.error(f"Error accessing IoT manager: {str(e)}")

def settings_page():
    """
    Render the settings page
    """
    st.subheader("Settings")
    
    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Agent Settings", 
        "Integration Settings", 
        "Analytics Settings",
        "System Settings"
    ])
    
    # Agent Settings
    with tab1:
        st.subheader("Agent Configuration")
        
        # Check if agent is initialized
        if not st.session_state.agent:
            st.warning("Agent not initialized.")
        else:
            # Get current agent state
            agent_state = st.session_state.agent.get_agent_state()
            
            # Agent name and personality
            col1, col2 = st.columns(2)
            with col1:
                agent_name = st.text_input(
                    "Agent Name:",
                    value=agent_state.get("name", "AI Companion")
                )
            with col2:
                agent_personality = st.text_input(
                    "Agent Personality:",
                    value=agent_state.get("personality", "helpful, informative, and insightful")
                )
            
            # Memory settings
            memory_usage = agent_state.get("memory_usage", {})
            memory_size = st.slider(
                "Memory Size (messages):",
                10, 200, memory_usage.get("max_size", 50)
            )
            
            compression_enabled = st.checkbox(
                "Enable Memory Compression",
                value=True
            )
            
            # Update button
            if st.button("Update Agent Settings"):
                with st.spinner("Updating agent settings..."):
                    # Prepare updates
                    updates = {
                        "name": agent_name,
                        "personality": agent_personality,
                        "memory_size": memory_size,
                        "compression_enabled": compression_enabled
                    }
                    
                    # Update agent
                    st.session_state.agent.update_agent_config(updates)
                    st.success("Agent settings updated successfully!")
    
    # Integration Settings
    with tab2:
        st.subheader("Integration Settings")
        
        # MCP settings
        st.subheader("Model Context Protocol (MCP)")
        
        # Check if MCP connector is initialized
        if not st.session_state.mcp_connector:
            st.warning("MCP connector not initialized.")
        else:
            # Get current status
            mcp_status = st.session_state.mcp_connector.get_status()
            
            # Display current status
            if mcp_status.get("connected", False):
                st.success("Connected to MCP server")
            else:
                st.error("Not connected to MCP server")
                if mcp_status.get("last_error"):
                    st.write(f"Error: {mcp_status.get('last_error')}")
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                mcp_server_url = st.text_input(
                    "MCP Server URL:",
                    value=mcp_status.get("server_url", "http://localhost:8080")
                )
            with col2:
                mcp_api_key = st.text_input(
                    "MCP API Key:",
                    value="",
                    type="password"
                )
                if not mcp_api_key and mcp_status.get("api_key_configured", False):
                    st.info("API key already configured")
        
        # Horizontal divider
        st.markdown("---")
        
        # GOAT settings
        st.subheader("GOAT SDK")
        
        # Check if GOAT connector is initialized
        if not st.session_state.goat_connector:
            st.warning("GOAT connector not initialized.")
        else:
            # Get current status
            goat_status = st.session_state.goat_connector.get_status()
            
            # Display current status
            if goat_status.get("initialized", False):
                st.success("GOAT SDK initialized")
                
                # Wallet status
                if goat_status.get("wallet_connected", False):
                    st.info("Wallet connected")
                else:
                    st.warning("No wallet connected")
            else:
                st.error("GOAT SDK not initialized")
                if goat_status.get("last_error"):
                    st.write(f"Error: {goat_status.get('last_error')}")
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                goat_api_key = st.text_input(
                    "GOAT API Key:",
                    value="",
                    type="password"
                )
                if not goat_api_key and goat_status.get("api_key_configured", False):
                    st.info("API key already configured")
            
            with col2:
                # Tool selection
                available_tools = [
                    "payments", "investments", "insights", 
                    "commerce", "assets", "nfts"
                ]
                enabled_tools = goat_status.get("enabled_tools", ["payments", "investments", "insights"])
                
                selected_tools = st.multiselect(
                    "Enabled Tools:",
                    available_tools,
                    default=enabled_tools
                )
        
        # Update integrations button
        if st.button("Update Integration Settings"):
            st.warning("Integration settings update not implemented in this demo")
            st.info("In a production application, this would update the MCP and GOAT settings")
    
    # Analytics Settings
    with tab3:
        st.subheader("Analytics Configuration")
        
        # Time Series settings
        st.subheader("Time Series Analysis")
        
        # Check if time series analyzer is initialized
        if not st.session_state.time_series_analyzer:
            st.warning("Time series analyzer not initialized.")
        else:
            # Get current status
            ts_status = st.session_state.time_series_analyzer.get_status()
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider(
                    "Window Size (days):",
                    1, 90, ts_status.get("window_size", 30)
                )
            with col2:
                resample_options = ["1min", "5min", "15min", "1H", "6H", "12H", "1D", "1W"]
                resample_frequency = st.selectbox(
                    "Resample Frequency:",
                    resample_options,
                    index=resample_options.index(ts_status.get("resample_frequency", "1D"))
                )
            
            decomposition_method = st.selectbox(
                "Decomposition Method:",
                ["additive", "multiplicative"],
                index=0 if ts_status.get("decomposition_method", "additive") == "additive" else 1
            )
        
        # Horizontal divider
        st.markdown("---")
        
        # Forecasting settings
        st.subheader("Forecasting")
        
        # Check if forecaster is initialized
        if not st.session_state.forecaster:
            st.warning("Forecaster not initialized.")
        else:
            # Get current status
            fc_status = st.session_state.forecaster.get_status()
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                prediction_horizon = st.slider(
                    "Default Prediction Horizon (days):",
                    1, 90, fc_status.get("prediction_horizon", 7)
                )
            with col2:
                confidence_interval = st.slider(
                    "Confidence Interval:",
                    0.7, 0.99, fc_status.get("confidence_interval", 0.95)
                )
            
            # Method selection
            available_methods = ["arima", "prophet", "lstm"]
            enabled_methods = fc_status.get("methods", ["arima", "prophet", "lstm"])
            
            selected_methods = st.multiselect(
                "Enabled Forecast Methods:",
                available_methods,
                default=enabled_methods
            )
        
        # Horizontal divider
        st.markdown("---")
        
        # Signal processing settings
        st.subheader("Signal Processing")
        
        # Check if signal processor is initialized
        if not st.session_state.signal_processor:
            st.warning("Signal processor not initialized.")
        else:
            # Get current status
            sig_status = st.session_state.signal_processor.get_status()
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                sampling_rate = st.slider(
                    "Default Sampling Rate (Hz):",
                    10, 1000, sig_status.get("sampling_rate", 100)
                )
            with col2:
                filter_options = ["butterworth", "chebyshev", "bessel", "elliptic"]
                filter_type = st.selectbox(
                    "Default Filter Type:",
                    filter_options,
                    index=filter_options.index(sig_status.get("filter_type", "butterworth"))
                )
            
            col1, col2 = st.columns(2)
            with col1:
                cutoff_frequency = st.slider(
                    "Default Cutoff Frequency (Hz):",
                    1, 100, sig_status.get("cutoff_frequency", 20)
                )
            with col2:
                iot_enabled = st.checkbox(
                    "Enable IoT Functionality",
                    value=sig_status.get("iot_enabled", True)
                )
        
        # Update analytics button
        if st.button("Update Analytics Settings"):
            st.warning("Analytics settings update not implemented in this demo")
            st.info("In a production application, this would update the time series, forecasting, and signal processing settings")
    
    # System Settings
    with tab4:
        st.subheader("System Configuration")
        
        # Distributed Computing Settings
        st.subheader("Distributed Computing")
        
        # Check if distributed compute is initialized
        if not st.session_state.distributed_compute:
            st.warning("Distributed computing not initialized.")
        else:
            # Get current status
            dc_status = st.session_state.distributed_compute.get_status()
            
            # Display current status
            engine_type = dc_status.get("engine", "none")
            if dc_status.get("enabled", False) and engine_type != "none":
                st.success(f"Distributed computing enabled ({engine_type})")
                
                # Additional status info
                if engine_type == "dask":
                    st.write(f"Connected workers: {dc_status.get('connected_workers', 0)}")
                elif engine_type == "ray":
                    st.write(f"CPUs: {dc_status.get('available_cpus', 0)}/{dc_status.get('total_cpus', 0)}")
            else:
                st.warning("Distributed computing disabled or unavailable")
            
            # Settings
            enabled = st.checkbox(
                "Enable Distributed Computing",
                value=dc_status.get("enabled", True)
            )
            
            col1, col2 = st.columns(2)
            with col1:
                engine = st.selectbox(
                    "Preferred Engine:",
                    ["auto", "dask", "ray", "none"],
                    index=0
                )
            with col2:
                worker_threads = st.slider(
                    "Worker Threads:",
                    1, 16, dc_status.get("workers", 4)
                )
            
            memory_options = ["1GB", "2GB", "4GB", "8GB", "16GB"]
            memory_limit = st.selectbox(
                "Memory Limit:",
                memory_options,
                index=memory_options.index(dc_status.get("memory_limit", "4GB")) if dc_status.get("memory_limit", "4GB") in memory_options else 2
            )
        
        # Horizontal divider
        st.markdown("---")
        
        # Application Settings
        st.subheader("Application Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            log_level = st.selectbox(
                "Log Level:",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
        with col2:
            theme = st.selectbox(
                "Theme:",
                ["Light", "Dark"],
                index=0
            )
        
        # Data storage settings
        st.subheader("Data Storage")
        
        storage_location = st.text_input(
            "Storage Location:",
            value="./data"
        )
        
        auto_backup = st.checkbox(
            "Enable Automatic Backups",
            value=True
        )
        
        # Update system button
        if st.button("Update System Settings"):
            st.warning("System settings update not implemented in this demo")
            st.info("In a production application, this would update the distributed computing and application settings")
        
        # Horizontal divider
        st.markdown("---")
        
        # Danger Zone
        st.subheader("Danger Zone")
        
        with st.expander("Advanced Options"):
            st.warning("These actions can cause data loss or disrupt the application")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset Application", type="primary"):
                    st.error("This would reset the application in a production environment")
            
            with col2:
                if st.button("Clear All Data", type="primary"):
                    st.error("This would clear all data in a production environment")
