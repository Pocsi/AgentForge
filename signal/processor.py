import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import json

# Import signal processing libraries
try:
    from scipy import signal
    import scipy.fftpack as fftpack
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

from signal.iot import IoTManager

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Handles signal processing tasks including filtering, 
    transformation, and analysis of signal data from various sources
    """
    
    def __init__(self, config: Dict[str, Any], distributed_compute=None):
        """
        Initialize the signal processor
        
        Args:
            config: Configuration dictionary
            distributed_compute: Distributed computation engine
        """
        self.config = config
        self.sampling_rate = config.get("sampling_rate", 100)
        self.filter_type = config.get("filter_type", "butterworth")
        self.cutoff_frequency = config.get("cutoff_frequency", 20)
        self.iot_enabled = config.get("iot_enabled", True)
        
        # Distributed compute for parallel processing
        self.distributed_compute = distributed_compute
        
        # Initialize IoT manager
        if self.iot_enabled:
            try:
                self.iot_manager = IoTManager(config)
                logger.info("IoT manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize IoT manager: {str(e)}")
                self.iot_enabled = False
                self.iot_manager = None
        else:
            self.iot_manager = None
        
        # Cache for recent analyses
        self.signal_cache = {}
        self.max_cache_size = 10
        
        logger.info(f"Signal processor initialized with sampling rate {self.sampling_rate} Hz")
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process signal data based on a natural language query
        
        Args:
            query: Natural language query describing the signal processing needed
            
        Returns:
            Dictionary with signal processing results
        """
        logger.info(f"Processing signal with query: {query}")
        
        try:
            # Check if this query is in the cache
            cache_key = self._generate_cache_key(query)
            if cache_key in self.signal_cache:
                logger.info(f"Using cached signal processing for query: {query}")
                return self.signal_cache[cache_key]
            
            # Parse the query to determine what kind of processing is needed
            processing_type, signal_source, parameters = self._parse_query(query)
            
            # Get signal data from the appropriate source
            signal_data = self._get_signal_data(signal_source, parameters)
            
            if signal_data is None or len(signal_data) == 0:
                return {"status": "error", "message": "No signal data available for processing"}
            
            # Apply the requested processing
            results = self._apply_processing(signal_data, processing_type, parameters)
            
            # Cache the results
            self._cache_results(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key from a query"""
        # Simple hash function for the query
        return f"sig_{hash(query) % 1000000}"
    
    def _cache_results(self, key: str, results: Dict[str, Any]) -> None:
        """Add processing results to the cache"""
        # Add to cache
        self.signal_cache[key] = results
        
        # Remove oldest entries if cache is too large
        if len(self.signal_cache) > self.max_cache_size:
            oldest_key = next(iter(self.signal_cache))
            del self.signal_cache[oldest_key]
    
    def _parse_query(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse a natural language query to determine the signal processing needed
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (processing_type, signal_source, parameters)
        """
        query = query.lower()
        
        # Default values
        processing_type = "filter"
        signal_source = "sample"
        parameters = {}
        
        # Determine processing type
        if "filter" in query:
            processing_type = "filter"
            # Extract filter parameters
            if "low" in query and "pass" in query:
                parameters["filter_subtype"] = "lowpass"
            elif "high" in query and "pass" in query:
                parameters["filter_subtype"] = "highpass"
            elif "band" in query and "pass" in query:
                parameters["filter_subtype"] = "bandpass"
            elif "notch" in query:
                parameters["filter_subtype"] = "notch"
        elif "fft" in query or "frequency" in query or "spectrum" in query:
            processing_type = "fft"
        elif "wavelet" in query:
            processing_type = "wavelet"
        elif "detect" in query or "peak" in query:
            processing_type = "peak_detection"
        elif "compress" in query:
            processing_type = "compression"
        
        # Determine signal source
        if "database" in query or "db" in query:
            signal_source = "database"
        elif "file" in query:
            signal_source = "file"
        elif "iot" in query or "sensor" in query or "device" in query:
            signal_source = "iot"
            
            # Try to extract device ID
            device_match = re.search(r'device[:\s]+([a-zA-Z0-9_-]+)', query)
            if device_match:
                parameters["device_id"] = device_match.group(1)
        
        # Extract frequency parameters
        freq_match = re.search(r'(\d+(?:\.\d+)?)\s*hz', query)
        if freq_match:
            parameters["frequency"] = float(freq_match.group(1))
        
        # Extract window size for processing
        window_match = re.search(r'window\s+(\d+)', query)
        if window_match:
            parameters["window_size"] = int(window_match.group(1))
        
        return processing_type, signal_source, parameters
    
    def _get_signal_data(self, signal_source: str, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Get signal data from the specified source
        
        Args:
            signal_source: Signal source type
            parameters: Parameters for data retrieval
            
        Returns:
            Numpy array with signal data
        """
        if signal_source == "iot":
            # Get data from IoT device
            if not self.iot_enabled or self.iot_manager is None:
                logger.warning("IoT functionality is not enabled or available")
                return self._generate_sample_signal(parameters)
            
            device_id = parameters.get("device_id")
            if device_id:
                try:
                    return self.iot_manager.get_device_data(device_id, parameters)
                except Exception as e:
                    logger.error(f"Error getting data from IoT device {device_id}: {str(e)}")
                    return self._generate_sample_signal(parameters)
            else:
                # Try to get data from any available device
                try:
                    devices = self.iot_manager.list_devices()
                    if devices:
                        return self.iot_manager.get_device_data(devices[0]["id"], parameters)
                except Exception as e:
                    logger.error(f"Error getting data from IoT devices: {str(e)}")
                
                return self._generate_sample_signal(parameters)
        
        elif signal_source == "database" or signal_source == "file":
            # In a real implementation, this would connect to databases or files
            # For now, just generate sample data
            logger.info(f"Using sample signal data for {signal_source} source")
            return self._generate_sample_signal(parameters)
        
        else:
            # Generate sample signal
            return self._generate_sample_signal(parameters)
    
    def _generate_sample_signal(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Generate a sample signal for testing
        
        Args:
            parameters: Signal generation parameters
            
        Returns:
            Numpy array with sample signal data
        """
        # Signal parameters
        sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
        duration = parameters.get("duration", 5.0)  # seconds
        
        # Generate time base
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # Generate composite signal with multiple frequency components
        # Base frequency
        base_freq = parameters.get("frequency", 10.0)
        
        # Generate signal: base frequency + harmonic + noise
        signal_data = (
            np.sin(2 * np.pi * base_freq * t) +                          # Base frequency
            0.5 * np.sin(2 * np.pi * (2 * base_freq) * t) +              # First harmonic
            0.25 * np.sin(2 * np.pi * (3 * base_freq) * t) +             # Second harmonic
            0.1 * np.random.normal(0, 1, size=len(t))                    # Noise
        )
        
        # Add some artificial spikes/anomalies
        num_spikes = 3
        spike_indices = np.random.choice(len(signal_data), size=num_spikes, replace=False)
        for idx in spike_indices:
            signal_data[idx] += 2.0 * np.random.choice([-1, 1])
        
        return signal_data
    
    def _apply_processing(
        self, 
        signal_data: np.ndarray, 
        processing_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply the requested processing to the signal data
        
        Args:
            signal_data: Signal data to process
            processing_type: Type of processing to apply
            parameters: Processing parameters
            
        Returns:
            Dictionary with processing results
        """
        if not SCIPY_AVAILABLE:
            return {
                "status": "error",
                "message": "SciPy is required for signal processing"
            }
        
        if processing_type == "filter":
            return self._apply_filter(signal_data, parameters)
        elif processing_type == "fft":
            return self._apply_fft(signal_data, parameters)
        elif processing_type == "wavelet":
            return self._apply_wavelet(signal_data, parameters)
        elif processing_type == "peak_detection":
            return self._detect_peaks(signal_data, parameters)
        elif processing_type == "compression":
            return self._compress_signal(signal_data, parameters)
        else:
            return {
                "status": "error",
                "message": f"Unknown processing type: {processing_type}"
            }
    
    def _apply_filter(self, signal_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a filter to the signal data
        
        Args:
            signal_data: Signal data to filter
            parameters: Filter parameters
            
        Returns:
            Dictionary with filtering results
        """
        sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
        filter_subtype = parameters.get("filter_subtype", "lowpass")
        
        # Set cutoff frequency
        cutoff = parameters.get("frequency", self.cutoff_frequency)
        
        # Normalize the cutoff frequency
        nyquist = 0.5 * sampling_rate
        normalized_cutoff = cutoff / nyquist
        
        # Design the filter
        try:
            # Different filter types based on subtype
            if filter_subtype == "lowpass":
                b, a = signal.butter(4, normalized_cutoff, btype='low')
                filter_name = "Butterworth Lowpass"
            elif filter_subtype == "highpass":
                b, a = signal.butter(4, normalized_cutoff, btype='high')
                filter_name = "Butterworth Highpass"
            elif filter_subtype == "bandpass":
                # For bandpass, we need two cutoff frequencies
                high_cutoff = parameters.get("high_frequency", 2 * cutoff)
                normalized_high = high_cutoff / nyquist
                b, a = signal.butter(4, [normalized_cutoff, normalized_high], btype='band')
                filter_name = "Butterworth Bandpass"
            elif filter_subtype == "notch":
                Q = parameters.get("q_factor", 30.0)
                b, a = signal.iirnotch(cutoff, Q, sampling_rate)
                filter_name = "IIR Notch"
            else:
                # Default to lowpass
                b, a = signal.butter(4, normalized_cutoff, btype='low')
                filter_name = "Butterworth Lowpass (Default)"
            
            # Apply the filter
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            # Generate time array for plotting
            duration = len(signal_data) / sampling_rate
            t = np.linspace(0, duration, len(signal_data), endpoint=False)
            
            # Prepare data for response
            original_data = []
            filtered_data = []
            
            for i, (time_val, orig, filt) in enumerate(zip(t, signal_data, filtered_signal)):
                # Limit the number of points for efficiency
                if i % max(1, len(t) // 1000) == 0:
                    original_data.append({
                        "time": float(time_val),
                        "value": float(orig)
                    })
                    filtered_data.append({
                        "time": float(time_val),
                        "value": float(filt)
                    })
            
            return {
                "status": "success",
                "processing_type": "filter",
                "filter_type": filter_name,
                "cutoff_frequency": cutoff,
                "sampling_rate": sampling_rate,
                "original_signal": original_data,
                "filtered_signal": filtered_data,
                "stats": {
                    "original_mean": float(np.mean(signal_data)),
                    "original_std": float(np.std(signal_data)),
                    "filtered_mean": float(np.mean(filtered_signal)),
                    "filtered_std": float(np.std(filtered_signal))
                }
            }
            
        except Exception as e:
            logger.error(f"Error applying filter: {str(e)}")
            return {
                "status": "error",
                "message": f"Error applying filter: {str(e)}"
            }
    
    def _apply_fft(self, signal_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Fast Fourier Transform to the signal data
        
        Args:
            signal_data: Signal data to transform
            parameters: Transform parameters
            
        Returns:
            Dictionary with FFT results
        """
        sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
        
        try:
            # Apply FFT
            n = len(signal_data)
            fft_data = fftpack.fft(signal_data)
            
            # Compute frequency bins
            freq = fftpack.fftfreq(n, 1/sampling_rate)
            
            # Consider only the positive half of the spectrum
            half_n = n // 2
            freq = freq[:half_n]
            fft_amplitude = np.abs(fft_data[:half_n]) / n
            
            # Find the dominant frequencies
            # Get indices of top 5 amplitudes
            top_indices = np.argsort(fft_amplitude)[-5:][::-1]
            dominant_frequencies = []
            
            for idx in top_indices:
                if fft_amplitude[idx] > 0.01:  # Only include significant components
                    dominant_frequencies.append({
                        "frequency": float(freq[idx]),
                        "amplitude": float(fft_amplitude[idx])
                    })
            
            # Prepare spectrum data for response
            spectrum_data = []
            
            for i, (f, amp) in enumerate(zip(freq, fft_amplitude)):
                # Limit the number of points for efficiency
                if i % max(1, len(freq) // 500) == 0:
                    spectrum_data.append({
                        "frequency": float(f),
                        "amplitude": float(amp)
                    })
            
            return {
                "status": "success",
                "processing_type": "fft",
                "sampling_rate": sampling_rate,
                "spectrum": spectrum_data,
                "dominant_frequencies": dominant_frequencies,
                "stats": {
                    "max_amplitude": float(np.max(fft_amplitude)),
                    "mean_amplitude": float(np.mean(fft_amplitude)),
                    "signal_power": float(np.sum(fft_amplitude**2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error applying FFT: {str(e)}")
            return {
                "status": "error",
                "message": f"Error applying FFT: {str(e)}"
            }
    
    def _apply_wavelet(self, signal_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply wavelet transform to the signal data
        
        Args:
            signal_data: Signal data to transform
            parameters: Transform parameters
            
        Returns:
            Dictionary with wavelet transform results
        """
        if not PYWAVELETS_AVAILABLE:
            return {
                "status": "error",
                "message": "PyWavelets is required for wavelet transforms"
            }
        
        wavelet_type = parameters.get("wavelet", "db4")
        level = parameters.get("level", 5)
        
        try:
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet_type, level=level)
            
            # Process each level
            approximation = coeffs[0]
            details = coeffs[1:]
            
            # Prepare wavelet data for response
            approximation_data = []
            details_data = []
            
            # Scale the x-axis for each level for better visualization
            for i, val in enumerate(approximation):
                if i % max(1, len(approximation) // 500) == 0:
                    approximation_data.append({
                        "index": i,
                        "value": float(val)
                    })
            
            for level_idx, detail in enumerate(details):
                level_data = []
                for i, val in enumerate(detail):
                    if i % max(1, len(detail) // 500) == 0:
                        level_data.append({
                            "index": i,
                            "value": float(val)
                        })
                
                details_data.append({
                    "level": level_idx + 1,
                    "data": level_data
                })
            
            return {
                "status": "success",
                "processing_type": "wavelet",
                "wavelet_type": wavelet_type,
                "levels": level,
                "approximation": approximation_data,
                "details": details_data,
                "stats": {
                    "approximation_energy": float(np.sum(approximation**2)),
                    "details_energy": [float(np.sum(d**2)) for d in details]
                }
            }
            
        except Exception as e:
            logger.error(f"Error applying wavelet transform: {str(e)}")
            return {
                "status": "error",
                "message": f"Error applying wavelet transform: {str(e)}"
            }
    
    def _detect_peaks(self, signal_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect peaks in the signal data
        
        Args:
            signal_data: Signal data to analyze
            parameters: Detection parameters
            
        Returns:
            Dictionary with peak detection results
        """
        sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
        height = parameters.get("height", None)
        threshold = parameters.get("threshold", 0.5)
        distance = parameters.get("distance", sampling_rate // 10)  # Minimum distance between peaks
        
        try:
            # Auto-determine height if not specified
            if height is None:
                height = np.mean(signal_data) + threshold * np.std(signal_data)
            
            # Find peaks
            peaks, properties = signal.find_peaks(
                signal_data, 
                height=height, 
                distance=distance
            )
            
            # Generate time base
            duration = len(signal_data) / sampling_rate
            t = np.linspace(0, duration, len(signal_data), endpoint=False)
            
            # Prepare data for response
            peak_data = []
            signal_data_points = []
            
            for peak_idx in peaks:
                peak_data.append({
                    "time": float(t[peak_idx]),
                    "value": float(signal_data[peak_idx])
                })
            
            for i, (time_val, val) in enumerate(zip(t, signal_data)):
                # Limit the number of points for efficiency
                if i % max(1, len(t) // 1000) == 0:
                    signal_data_points.append({
                        "time": float(time_val),
                        "value": float(val)
                    })
            
            return {
                "status": "success",
                "processing_type": "peak_detection",
                "sampling_rate": sampling_rate,
                "peak_count": len(peaks),
                "peaks": peak_data,
                "signal": signal_data_points,
                "stats": {
                    "mean_peak_height": float(np.mean(signal_data[peaks])) if len(peaks) > 0 else 0,
                    "max_peak_height": float(np.max(signal_data[peaks])) if len(peaks) > 0 else 0,
                    "mean_peak_distance": float(np.mean(np.diff(peaks)) / sampling_rate) if len(peaks) > 1 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting peaks: {str(e)}")
            return {
                "status": "error",
                "message": f"Error detecting peaks: {str(e)}"
            }
    
    def _compress_signal(self, signal_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress the signal data
        
        Args:
            signal_data: Signal data to compress
            parameters: Compression parameters
            
        Returns:
            Dictionary with compression results
        """
        method = parameters.get("method", "threshold")
        threshold = parameters.get("threshold", 0.1)
        
        try:
            original_size = len(signal_data)
            
            if method == "threshold":
                # Simple threshold-based compression: keep only values above threshold
                # Calculate the absolute signal
                abs_signal = np.abs(signal_data)
                
                # Determine threshold value if relative
                threshold_value = threshold * np.max(abs_signal)
                
                # Keep indices where signal is above threshold
                keep_indices = np.where(abs_signal > threshold_value)[0]
                
                # Create compressed representation
                compressed_values = signal_data[keep_indices]
                compressed_indices = keep_indices
                
                # Reconstruct signal for comparison
                reconstructed = np.zeros_like(signal_data)
                reconstructed[compressed_indices] = compressed_values
                
            elif method == "wavelet" and PYWAVELETS_AVAILABLE:
                # Wavelet-based compression
                wavelet_type = parameters.get("wavelet", "db4")
                level = parameters.get("level", 5)
                
                # Decompose
                coeffs = pywt.wavedec(signal_data, wavelet_type, level=level)
                
                # Threshold coefficients
                modified_coeffs = []
                for c in coeffs:
                    max_coeff = np.max(np.abs(c))
                    threshold_value = threshold * max_coeff
                    modified_c = pywt.threshold(c, threshold_value, mode='hard')
                    modified_coeffs.append(modified_c)
                
                # Reconstruct
                reconstructed = pywt.waverec(modified_coeffs, wavelet_type)
                
                # Ensure reconstructed signal has same length as original
                if len(reconstructed) > len(signal_data):
                    reconstructed = reconstructed[:len(signal_data)]
                elif len(reconstructed) < len(signal_data):
                    padding = np.zeros(len(signal_data) - len(reconstructed))
                    reconstructed = np.concatenate([reconstructed, padding])
                
                # Count non-zero coefficients
                non_zero_count = sum(np.count_nonzero(c) for c in modified_coeffs)
                compressed_values = non_zero_count
                compressed_indices = None
                
            else:
                # Fallback to simple sampling
                sampling_factor = int(1 / threshold)
                compressed_indices = np.arange(0, original_size, sampling_factor)
                compressed_values = signal_data[compressed_indices]
                
                # Reconstruct signal for comparison
                reconstructed = np.interp(
                    np.arange(original_size),
                    compressed_indices,
                    compressed_values
                )
            
            # Calculate compression metrics
            if method == "wavelet":
                compression_ratio = original_size / compressed_values
            else:
                compression_ratio = original_size / len(compressed_values)
            
            mse = np.mean((signal_data - reconstructed)**2)
            max_error = np.max(np.abs(signal_data - reconstructed))
            
            # Prepare data for response
            sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
            duration = len(signal_data) / sampling_rate
            t = np.linspace(0, duration, len(signal_data), endpoint=False)
            
            original_data = []
            reconstructed_data = []
            
            for i, (time_val, orig, recon) in enumerate(zip(t, signal_data, reconstructed)):
                # Limit the number of points for efficiency
                if i % max(1, len(t) // 1000) == 0:
                    original_data.append({
                        "time": float(time_val),
                        "value": float(orig)
                    })
                    reconstructed_data.append({
                        "time": float(time_val),
                        "value": float(recon)
                    })
            
            return {
                "status": "success",
                "processing_type": "compression",
                "method": method,
                "threshold": threshold,
                "original_size": original_size,
                "compressed_size": len(compressed_values) if compressed_indices is not None else compressed_values,
                "compression_ratio": float(compression_ratio),
                "original_signal": original_data,
                "reconstructed_signal": reconstructed_data,
                "stats": {
                    "mse": float(mse),
                    "max_error": float(max_error),
                    "snr": float(10 * np.log10(np.sum(signal_data**2) / (mse * len(signal_data)))) if mse > 0 else float('inf')
                }
            }
            
        except Exception as e:
            logger.error(f"Error compressing signal: {str(e)}")
            return {
                "status": "error",
                "message": f"Error compressing signal: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the signal processor
        
        Returns:
            Dictionary with status information
        """
        status = {
            "sampling_rate": self.sampling_rate,
            "filter_type": self.filter_type,
            "cutoff_frequency": self.cutoff_frequency,
            "iot_enabled": self.iot_enabled,
            "cache_size": len(self.signal_cache),
            "max_cache_size": self.max_cache_size,
            "scipy_available": SCIPY_AVAILABLE,
            "pywavelets_available": PYWAVELETS_AVAILABLE
        }
        
        # Add IoT status if enabled
        if self.iot_enabled and self.iot_manager:
            status["iot"] = self.iot_manager.get_status()
        
        return status
