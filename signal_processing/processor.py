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

from signal_processing.iot import IoTManager

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
            fft_output = fftpack.fft(signal_data)
            fft_magnitude = np.abs(fft_output)
            
            # Generate frequency array
            n = len(signal_data)
            frequencies = fftpack.fftfreq(n, 1.0/sampling_rate)
            
            # Only take the positive frequencies
            pos_mask = frequencies >= 0
            pos_frequencies = frequencies[pos_mask]
            pos_magnitudes = fft_magnitude[pos_mask]
            
            # If there are too many points, downsample
            max_points = 1000
            if len(pos_frequencies) > max_points:
                step = len(pos_frequencies) // max_points
                pos_frequencies = pos_frequencies[::step]
                pos_magnitudes = pos_magnitudes[::step]
            
            # Prepare data for response
            spectrum_data = []
            for freq, mag in zip(pos_frequencies, pos_magnitudes):
                spectrum_data.append({
                    "frequency": float(freq),
                    "amplitude": float(mag / n)  # Normalize by signal length
                })
            
            # Find peak frequencies
            peak_threshold = parameters.get("peak_threshold", 0.1)
            normalized_magnitudes = pos_magnitudes / n
            peak_indices = signal.find_peaks(normalized_magnitudes, height=peak_threshold)[0]
            peaks = []
            
            for idx in peak_indices:
                if idx < len(pos_frequencies):
                    peaks.append({
                        "frequency": float(pos_frequencies[idx]),
                        "amplitude": float(normalized_magnitudes[idx])
                    })
            
            return {
                "status": "success",
                "processing_type": "fft",
                "sampling_rate": sampling_rate,
                "spectrum": spectrum_data,
                "peaks": peaks,
                "stats": {
                    "signal_mean": float(np.mean(signal_data)),
                    "signal_std": float(np.std(signal_data)),
                    "signal_rms": float(np.sqrt(np.mean(signal_data**2)))
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
        
        try:
            # Wavelet parameters
            wavelet = parameters.get("wavelet", "db4")
            level = parameters.get("level", 5)
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # Prepare data for response
            coeff_data = []
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    name = "Approximation"
                else:
                    name = f"Detail {i}"
                
                # If there are too many points, downsample
                max_points = 500
                if len(coeff) > max_points:
                    step = len(coeff) // max_points
                    coeff = coeff[::step]
                
                # Create data points
                points = []
                for j, value in enumerate(coeff):
                    points.append({
                        "index": j,
                        "value": float(value)
                    })
                
                coeff_data.append({
                    "name": name,
                    "data": points
                })
            
            return {
                "status": "success",
                "processing_type": "wavelet",
                "wavelet": wavelet,
                "level": level,
                "coefficients": coeff_data,
                "stats": {
                    "signal_mean": float(np.mean(signal_data)),
                    "signal_std": float(np.std(signal_data)),
                    "signal_energy": float(np.sum(signal_data**2))
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
        
        try:
            # Peak detection parameters
            height = parameters.get("height", 0.5)
            distance = parameters.get("distance", int(sampling_rate * 0.1))  # Min distance between peaks
            prominence = parameters.get("prominence", 0.2)
            
            # Detect peaks
            peaks, properties = signal.find_peaks(
                signal_data,
                height=height,
                distance=distance,
                prominence=prominence
            )
            
            # Generate time array
            duration = len(signal_data) / sampling_rate
            t = np.linspace(0, duration, len(signal_data), endpoint=False)
            
            # Prepare peak data
            peak_data = []
            for idx in peaks:
                peak_data.append({
                    "time": float(t[idx]),
                    "value": float(signal_data[idx])
                })
            
            # Prepare signal data for visualization
            # Downsample if needed
            signal_viz_data = []
            max_points = 1000
            if len(signal_data) > max_points:
                step = len(signal_data) // max_points
                for i in range(0, len(signal_data), step):
                    signal_viz_data.append({
                        "time": float(t[i]),
                        "value": float(signal_data[i])
                    })
            else:
                for i in range(len(signal_data)):
                    signal_viz_data.append({
                        "time": float(t[i]),
                        "value": float(signal_data[i])
                    })
            
            return {
                "status": "success",
                "processing_type": "peak_detection",
                "sampling_rate": sampling_rate,
                "signal_data": signal_viz_data,
                "peaks": peak_data,
                "peak_count": len(peaks),
                "stats": {
                    "signal_mean": float(np.mean(signal_data)),
                    "signal_std": float(np.std(signal_data)),
                    "mean_peak_height": float(np.mean(signal_data[peaks])) if len(peaks) > 0 else 0,
                    "max_peak_height": float(np.max(signal_data[peaks])) if len(peaks) > 0 else 0
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
        sampling_rate = parameters.get("sampling_rate", self.sampling_rate)
        
        try:
            # Compression parameters
            method = parameters.get("method", "pywt" if PYWAVELETS_AVAILABLE else "threshold")
            compression_ratio = parameters.get("compression_ratio", 0.1)
            threshold = parameters.get("threshold", 0.1)
            
            # Generate time array
            duration = len(signal_data) / sampling_rate
            t = np.linspace(0, duration, len(signal_data), endpoint=False)
            
            if method == "pywt" and PYWAVELETS_AVAILABLE:
                # Wavelet-based compression
                wavelet = parameters.get("wavelet", "db4")
                
                # Decompose signal
                coeffs = pywt.wavedec(signal_data, wavelet)
                
                # Determine threshold based on compression ratio
                all_coeffs = np.concatenate([c for c in coeffs])
                sorted_coeffs = np.sort(np.abs(all_coeffs))
                threshold_idx = int((1 - compression_ratio) * len(sorted_coeffs))
                threshold = sorted_coeffs[threshold_idx]
                
                # Apply thresholding
                coeffs_thresholded = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
                
                # Reconstruct signal
                compressed_signal = pywt.waverec(coeffs_thresholded, wavelet)
                
                # If reconstruction adds extra points, trim to original length
                if len(compressed_signal) > len(signal_data):
                    compressed_signal = compressed_signal[:len(signal_data)]
                
                # Calculate compression stats
                zeros_after = sum(1 for c in all_coeffs if abs(c) <= threshold)
                compression_achieved = zeros_after / len(all_coeffs)
                
            else:
                # Simple threshold-based compression
                # Keep only values above threshold
                mean_val = np.mean(np.abs(signal_data))
                threshold = mean_val * threshold
                
                compressed_signal = np.copy(signal_data)
                mask = np.abs(compressed_signal) < threshold
                compressed_signal[mask] = 0
                
                # Calculate compression stats
                zeros_after = np.sum(mask)
                compression_achieved = zeros_after / len(signal_data)
            
            # Calculate error metrics
            mse = np.mean((signal_data - compressed_signal[:len(signal_data)])**2)
            max_error = np.max(np.abs(signal_data - compressed_signal[:len(signal_data)]))
            
            # Prepare data for visualization
            # Downsample if needed
            original_data = []
            compressed_data = []
            
            max_points = 1000
            if len(signal_data) > max_points:
                step = len(signal_data) // max_points
                for i in range(0, len(signal_data), step):
                    if i < len(compressed_signal):
                        original_data.append({
                            "time": float(t[i]),
                            "value": float(signal_data[i])
                        })
                        compressed_data.append({
                            "time": float(t[i]),
                            "value": float(compressed_signal[i])
                        })
            else:
                for i in range(len(signal_data)):
                    if i < len(compressed_signal):
                        original_data.append({
                            "time": float(t[i]),
                            "value": float(signal_data[i])
                        })
                        compressed_data.append({
                            "time": float(t[i]),
                            "value": float(compressed_signal[i])
                        })
            
            return {
                "status": "success",
                "processing_type": "compression",
                "method": method,
                "sampling_rate": sampling_rate,
                "compression_target": compression_ratio,
                "compression_achieved": float(compression_achieved),
                "original_signal": original_data,
                "compressed_signal": compressed_data,
                "stats": {
                    "mean_squared_error": float(mse),
                    "max_error": float(max_error),
                    "original_size": len(signal_data),
                    "nonzero_coeffs": len(signal_data) - int(zeros_after)
                }
            }
            
        except Exception as e:
            logger.error(f"Error compressing signal: {str(e)}")
            return {
                "status": "error",
                "message": f"Error compressing signal: {str(e)}"
            }
    
    def _force_iot_discovery(self) -> List[Dict[str, Any]]:
        """
        Force IoT device discovery
        
        Returns:
            List of discovered devices
        """
        if self.iot_enabled and self.iot_manager:
            try:
                return self.iot_manager.discover_devices()
            except Exception as e:
                logger.error(f"Error during forced IoT discovery: {str(e)}")
                return []
        return []
    
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
            "scipy_available": SCIPY_AVAILABLE,
            "pywavelets_available": PYWAVELETS_AVAILABLE
        }
        
        # Add IoT status if enabled
        if self.iot_enabled and self.iot_manager:
            status["iot"] = self.iot_manager.get_status()
        
        return status