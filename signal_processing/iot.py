import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import json
import time
from datetime import datetime
import re
import threading

logger = logging.getLogger(__name__)

class IoTManager:
    """
    Manages IoT device connections and data retrieval
    for signal processing from edge devices and sensors
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the IoT manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.iot_enabled = config.get("iot_enabled", True)
        self.discovery_enabled = config.get("discovery_enabled", True)
        self.auto_connect = config.get("auto_connect", True)
        
        # Device storage
        self.devices = {}
        self.connected_devices = set()
        
        # Last discovery time
        self.last_discovery = 0
        self.discovery_interval = 300  # 5 minutes
        
        # Thread for background device discovery
        self.discovery_thread = None
        
        # Initialize and discover devices if enabled
        if self.iot_enabled:
            self._initialize_devices()
            
            if self.discovery_enabled:
                self._start_discovery_thread()
        
        logger.info(f"IoT manager initialized with {len(self.devices)} devices")
    
    def _initialize_devices(self) -> None:
        """Initialize devices from configuration"""
        # Load any pre-configured devices
        preconfigured_devices = self.config.get("devices", [])
        
        for device_config in preconfigured_devices:
            device_id = device_config.get("id")
            if device_id:
                self.devices[device_id] = {
                    "id": device_id,
                    "name": device_config.get("name", f"Device {device_id}"),
                    "type": device_config.get("type", "sensor"),
                    "protocol": device_config.get("protocol", "mqtt"),
                    "address": device_config.get("address", None),
                    "credentials": device_config.get("credentials", None),
                    "status": "registered",
                    "last_seen": None,
                    "capabilities": device_config.get("capabilities", [])
                }
        
        # If auto connect is enabled, try to connect to pre-configured devices
        if self.auto_connect:
            for device_id in self.devices:
                try:
                    self._connect_device(device_id)
                except Exception as e:
                    logger.error(f"Error connecting to device {device_id}: {str(e)}")
    
    def _start_discovery_thread(self) -> None:
        """Start the background discovery thread"""
        if self.discovery_thread is None or not self.discovery_thread.is_alive():
            self.discovery_thread = threading.Thread(
                target=self._discovery_loop,
                daemon=True
            )
            self.discovery_thread.start()
    
    def _discovery_loop(self) -> None:
        """Background loop for device discovery"""
        while self.discovery_enabled:
            try:
                self.discover_devices()
            except Exception as e:
                logger.error(f"Error in discovery loop: {str(e)}")
            
            # Sleep for the discovery interval
            time.sleep(self.discovery_interval)
    
    def discover_devices(self) -> List[Dict[str, Any]]:
        """
        Discover IoT devices on the network
        
        Returns:
            List of discovered devices
        """
        # Update last discovery time
        self.last_discovery = time.time()
        
        # In a real implementation, this would use various protocols to
        # discover devices (mDNS, UPnP, Bluetooth, etc.)
        # For now, we'll just simulate discovery
        
        # Simulate network discovery
        discovered_devices = []
        
        # Generate some virtual devices for demo purposes
        for i in range(1, 4):
            device_id = f"virtual_sensor_{i}"
            
            # Skip if already known
            if device_id in self.devices:
                continue
            
            # Create virtual device
            device = {
                "id": device_id,
                "name": f"Virtual Sensor {i}",
                "type": "sensor",
                "protocol": "virtual",
                "address": f"10.0.0.{100 + i}",
                "status": "discovered",
                "last_seen": datetime.now().isoformat(),
                "capabilities": ["temperature", "vibration", "voltage"]
            }
            
            self.devices[device_id] = device
            discovered_devices.append(device)
        
        logger.info(f"Discovered {len(discovered_devices)} new devices")
        return discovered_devices
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List all known IoT devices
        
        Returns:
            List of device information dictionaries
        """
        # Ensure discovery has run at least once
        if not self.devices and self.discovery_enabled:
            try:
                self.discover_devices()
            except Exception as e:
                logger.error(f"Error discovering devices: {str(e)}")
        
        return list(self.devices.values())
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific device
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device information dictionary or None if not found
        """
        return self.devices.get(device_id)
    
    def _connect_device(self, device_id: str) -> bool:
        """
        Connect to an IoT device
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if successful, False otherwise
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        # In a real implementation, this would establish a connection
        # to the actual device using the appropriate protocol
        
        # Simulate connection
        self.devices[device_id]["status"] = "connected"
        self.devices[device_id]["last_seen"] = datetime.now().isoformat()
        self.connected_devices.add(device_id)
        
        logger.info(f"Connected to device {device_id}")
        return True
    
    def _disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect from an IoT device
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if successful, False otherwise
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        # Simulate disconnection
        self.devices[device_id]["status"] = "disconnected"
        if device_id in self.connected_devices:
            self.connected_devices.remove(device_id)
        
        logger.info(f"Disconnected from device {device_id}")
        return True
    
    def get_device_data(self, device_id: str, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Get data from an IoT device
        
        Args:
            device_id: Device identifier
            parameters: Data retrieval parameters
            
        Returns:
            Numpy array with device data
        """
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        # Connect to device if not already connected
        if device_id not in self.connected_devices:
            if not self._connect_device(device_id):
                raise ConnectionError(f"Failed to connect to device {device_id}")
        
        # In a real implementation, this would get actual data from the device
        # For now, generate synthetic data based on device type and capabilities
        
        # Parameters
        parameters = parameters or {}
        sampling_rate = parameters.get("sampling_rate", 100)
        duration = parameters.get("duration", 5.0)  # seconds
        
        # Generate time base
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # Get device info
        device = self.devices[device_id]
        device_type = device.get("type", "sensor")
        capabilities = device.get("capabilities", [])
        
        # Generate different signal patterns based on device type and capabilities
        if "temperature" in capabilities:
            # Temperature signal: slow oscillation with noise
            base = 25.0  # 25°C baseline
            variation = 2.0  # +/- 2°C variation
            
            # Slow oscillation with period of several seconds
            oscillation = variation * np.sin(2 * np.pi * 0.1 * t)
            noise = 0.2 * np.random.normal(0, 1, size=len(t))
            
            signal_data = base + oscillation + noise
            
        elif "vibration" in capabilities:
            # Vibration signal: multiple frequency components
            base = 0.0
            primary = 1.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz primary vibration
            secondary = 0.3 * np.sin(2 * np.pi * 25 * t)  # 25 Hz secondary vibration
            noise = 0.1 * np.random.normal(0, 1, size=len(t))
            
            # Add some impulses
            impulses = np.zeros_like(t)
            impulse_times = np.random.choice(len(t), size=3, replace=False)
            impulses[impulse_times] = 2.0
            
            signal_data = base + primary + secondary + noise + impulses
            
        elif "voltage" in capabilities:
            # Voltage signal: stable with occasional dips
            base = 120.0  # 120V baseline
            noise = 0.5 * np.random.normal(0, 1, size=len(t))
            
            # Add voltage dips
            dips = np.zeros_like(t)
            dip_start = int(len(t) * 0.3)
            dip_end = int(len(t) * 0.35)
            dips[dip_start:dip_end] = -10.0
            
            signal_data = base + noise + dips
            
        else:
            # Generic sensor data: random walk
            steps = np.random.normal(0, 1, size=len(t))
            signal_data = np.cumsum(steps)
            
            # Normalize to reasonable range
            signal_data = 10 + 5 * (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
        
        # Update last seen timestamp
        self.devices[device_id]["last_seen"] = datetime.now().isoformat()
        
        return signal_data
    
    def send_command(self, device_id: str, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a command to an IoT device
        
        Args:
            device_id: Device identifier
            command: Command to send
            parameters: Command parameters
            
        Returns:
            Dictionary with command result
        """
        if device_id not in self.devices:
            return {
                "status": "error",
                "message": f"Device {device_id} not found"
            }
        
        # Connect to device if not already connected
        if device_id not in self.connected_devices:
            if not self._connect_device(device_id):
                return {
                    "status": "error",
                    "message": f"Failed to connect to device {device_id}"
                }
        
        # In a real implementation, this would send actual commands to the device
        # For now, just log the command and return a success response
        
        logger.info(f"Sending command '{command}' to device {device_id} with parameters: {parameters}")
        
        # Simulate command execution
        time.sleep(0.5)
        
        return {
            "status": "success",
            "device_id": device_id,
            "command": command,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the IoT manager
        
        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.iot_enabled,
            "discovery_enabled": self.discovery_enabled,
            "auto_connect": self.auto_connect,
            "last_discovery": datetime.fromtimestamp(self.last_discovery).isoformat() if self.last_discovery > 0 else None,
            "device_count": len(self.devices),
            "connected_devices": len(self.connected_devices)
        }
    
    def shutdown(self) -> None:
        """Shutdown the IoT manager and disconnect all devices"""
        logger.info("Shutting down IoT manager")
        
        # Stop discovery thread
        self.discovery_enabled = False
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=1.0)
        
        # Disconnect all devices
        for device_id in list(self.connected_devices):
            try:
                self._disconnect_device(device_id)
            except Exception as e:
                logger.error(f"Error disconnecting from device {device_id}: {str(e)}")
        
        self.iot_enabled = False