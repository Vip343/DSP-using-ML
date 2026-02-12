"""
Classical DSP filters for audio and signal denoising.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, rfft, irfft
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from .config import Config
from .data_loader import AudioData, SensorData


@dataclass
class FilterResult:
    """Container for filter output."""
    filtered_signal: np.ndarray
    method: str
    parameters: dict


class DSPFilters:
    """Classical DSP filters for noise reduction."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def wiener_filter(self, noisy_signal: np.ndarray, 
                      noise_estimate: Optional[np.ndarray] = None,
                      frame_length: Optional[int] = None,
                      hop_length: Optional[int] = None) -> FilterResult:
        """
        Apply Wiener filter for noise reduction.
        
        The Wiener filter minimizes the mean square error between the estimated
        and the true signal. It works in the frequency domain by estimating
        the signal-to-noise ratio and applying optimal filtering.
        
        Args:
            noisy_signal: Input noisy signal
            noise_estimate: Estimated noise signal (uses first 0.5s if not provided)
            frame_length: STFT frame length
            hop_length: STFT hop length
            
        Returns:
            FilterResult with filtered signal
        """
        frame_length = frame_length or self.config.wiener_frame_length
        hop_length = hop_length or self.config.wiener_hop_length
        
        # Estimate noise spectrum from noise signal or initial frames
        if noise_estimate is None:
            # Use first 0.5 seconds as noise estimate
            noise_frames = int(0.5 * self.config.sample_rate)
            noise_estimate = noisy_signal[:min(noise_frames, len(noisy_signal)//4)]
        
        # Compute STFT of noisy signal
        _, _, noisy_stft = signal.stft(noisy_signal, nperseg=frame_length, 
                                        noverlap=frame_length-hop_length)
        
        # Compute noise power spectrum
        _, _, noise_stft = signal.stft(noise_estimate, nperseg=frame_length,
                                        noverlap=frame_length-hop_length)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        
        # Compute noisy signal power spectrum
        noisy_power = np.abs(noisy_stft) ** 2
        
        # Wiener filter gain: H = max(0, 1 - noise_power / noisy_power)
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        wiener_gain = np.maximum(0, 1 - noise_power / (noisy_power + eps))
        
        # Apply Wiener filter
        filtered_stft = wiener_gain * noisy_stft
        
        # Inverse STFT
        _, filtered_signal = signal.istft(filtered_stft, nperseg=frame_length,
                                          noverlap=frame_length-hop_length)
        
        # Match length to original
        filtered_signal = self._match_length(filtered_signal, len(noisy_signal))
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='wiener',
            parameters={
                'frame_length': frame_length,
                'hop_length': hop_length
            }
        )
    
    def spectral_subtraction(self, noisy_signal: np.ndarray,
                             noise_estimate: Optional[np.ndarray] = None,
                             alpha: Optional[float] = None,
                             beta: Optional[float] = None,
                             frame_length: Optional[int] = None,
                             hop_length: Optional[int] = None) -> FilterResult:
        """
        Apply spectral subtraction for noise reduction.
        
        Spectral subtraction estimates the clean signal spectrum by subtracting
        an estimate of the noise spectrum from the noisy signal spectrum.
        
        Args:
            noisy_signal: Input noisy signal
            noise_estimate: Estimated noise signal (uses first 0.5s if not provided)
            alpha: Oversubtraction factor (default from config)
            beta: Spectral floor parameter (default from config)
            frame_length: STFT frame length
            hop_length: STFT hop length
            
        Returns:
            FilterResult with filtered signal
        """
        alpha = alpha if alpha is not None else self.config.spectral_sub_alpha
        beta = beta if beta is not None else self.config.spectral_sub_beta
        frame_length = frame_length or self.config.spectral_sub_frame_length
        hop_length = hop_length or self.config.spectral_sub_hop_length
        
        # Estimate noise spectrum
        if noise_estimate is None:
            noise_frames = int(0.5 * self.config.sample_rate)
            noise_estimate = noisy_signal[:min(noise_frames, len(noisy_signal)//4)]
        
        # Compute STFT
        _, _, noisy_stft = signal.stft(noisy_signal, nperseg=frame_length,
                                        noverlap=frame_length-hop_length)
        _, _, noise_stft = signal.stft(noise_estimate, nperseg=frame_length,
                                        noverlap=frame_length-hop_length)
        
        # Get magnitude and phase
        noisy_mag = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)
        
        # Estimate noise magnitude spectrum (averaged)
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Spectral subtraction with oversubtraction and spectral floor
        # |S|^2 = max(|Y|^2 - alpha * |N|^2, beta * |Y|^2)
        subtracted_power = noisy_mag ** 2 - alpha * noise_mag ** 2
        spectral_floor = beta * noisy_mag ** 2
        clean_power = np.maximum(subtracted_power, spectral_floor)
        clean_mag = np.sqrt(clean_power)
        
        # Reconstruct with original phase
        filtered_stft = clean_mag * np.exp(1j * noisy_phase)
        
        # Inverse STFT
        _, filtered_signal = signal.istft(filtered_stft, nperseg=frame_length,
                                          noverlap=frame_length-hop_length)
        
        filtered_signal = self._match_length(filtered_signal, len(noisy_signal))
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='spectral_subtraction',
            parameters={
                'alpha': alpha,
                'beta': beta,
                'frame_length': frame_length,
                'hop_length': hop_length
            }
        )
    
    def lowpass_filter(self, noisy_signal: np.ndarray,
                       cutoff_freq: Optional[float] = None,
                       sample_rate: Optional[int] = None,
                       order: Optional[int] = None) -> FilterResult:
        """
        Apply lowpass Butterworth filter.
        
        Args:
            noisy_signal: Input signal
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate of signal
            order: Filter order
            
        Returns:
            FilterResult with filtered signal
        """
        cutoff_freq = cutoff_freq or self.config.lowpass_cutoff
        sample_rate = sample_rate or self.config.sample_rate
        order = order or self.config.filter_order
        
        # Normalize cutoff frequency
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        normalized_cutoff = min(normalized_cutoff, 0.99)
        
        # Design Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        # Apply filter (forward-backward for zero phase)
        filtered_signal = signal.filtfilt(b, a, noisy_signal)
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='lowpass',
            parameters={
                'cutoff_freq': cutoff_freq,
                'sample_rate': sample_rate,
                'order': order
            }
        )
    
    def highpass_filter(self, noisy_signal: np.ndarray,
                        cutoff_freq: Optional[float] = None,
                        sample_rate: Optional[int] = None,
                        order: Optional[int] = None) -> FilterResult:
        """
        Apply highpass Butterworth filter.
        
        Args:
            noisy_signal: Input signal
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate of signal
            order: Filter order
            
        Returns:
            FilterResult with filtered signal
        """
        cutoff_freq = cutoff_freq or self.config.highpass_cutoff
        sample_rate = sample_rate or self.config.sample_rate
        order = order or self.config.filter_order
        
        # Normalize cutoff frequency
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        normalized_cutoff = max(normalized_cutoff, 0.01)
        
        # Design Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, noisy_signal)
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='highpass',
            parameters={
                'cutoff_freq': cutoff_freq,
                'sample_rate': sample_rate,
                'order': order
            }
        )
    
    def bandpass_filter(self, noisy_signal: np.ndarray,
                        low_cutoff: Optional[float] = None,
                        high_cutoff: Optional[float] = None,
                        sample_rate: Optional[int] = None,
                        order: Optional[int] = None) -> FilterResult:
        """
        Apply bandpass Butterworth filter.
        
        Args:
            noisy_signal: Input signal
            low_cutoff: Low cutoff frequency in Hz
            high_cutoff: High cutoff frequency in Hz
            sample_rate: Sample rate of signal
            order: Filter order
            
        Returns:
            FilterResult with filtered signal
        """
        low_cutoff = low_cutoff or self.config.highpass_cutoff
        high_cutoff = high_cutoff or self.config.lowpass_cutoff
        sample_rate = sample_rate or self.config.sample_rate
        order = order or self.config.filter_order
        
        # Normalize cutoff frequencies
        nyquist = sample_rate / 2
        low_normalized = max(low_cutoff / nyquist, 0.01)
        high_normalized = min(high_cutoff / nyquist, 0.99)
        
        # Design Butterworth filter
        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, noisy_signal)
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='bandpass',
            parameters={
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff,
                'sample_rate': sample_rate,
                'order': order
            }
        )
    
    def moving_average(self, noisy_signal: np.ndarray,
                       window_size: int = 5) -> FilterResult:
        """
        Apply moving average filter (simple smoothing).
        
        Useful for sensor data denoising.
        
        Args:
            noisy_signal: Input signal
            window_size: Size of moving average window
            
        Returns:
            FilterResult with filtered signal
        """
        kernel = np.ones(window_size) / window_size
        filtered_signal = np.convolve(noisy_signal, kernel, mode='same')
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='moving_average',
            parameters={'window_size': window_size}
        )
    
    def median_filter(self, noisy_signal: np.ndarray,
                      kernel_size: int = 5) -> FilterResult:
        """
        Apply median filter for impulse noise removal.
        
        Effective for salt-and-pepper noise in sensor data.
        
        Args:
            noisy_signal: Input signal
            kernel_size: Size of median filter kernel
            
        Returns:
            FilterResult with filtered signal
        """
        filtered_signal = signal.medfilt(noisy_signal, kernel_size=kernel_size)
        
        return FilterResult(
            filtered_signal=filtered_signal,
            method='median',
            parameters={'kernel_size': kernel_size}
        )
    
    def filter_audio(self, audio_data: AudioData, method: str = 'wiener',
                     noise_estimate: Optional[np.ndarray] = None,
                     **kwargs) -> Tuple[AudioData, FilterResult]:
        """
        Apply specified filter to audio data.
        
        Args:
            audio_data: Input audio data
            method: Filter method ('wiener', 'spectral_subtraction', 'lowpass', 
                    'highpass', 'bandpass')
            noise_estimate: Optional noise estimate for Wiener/spectral subtraction
            **kwargs: Additional parameters for the filter
            
        Returns:
            Tuple of (filtered AudioData, FilterResult)
        """
        filter_methods = {
            'wiener': self.wiener_filter,
            'spectral_subtraction': self.spectral_subtraction,
            'lowpass': self.lowpass_filter,
            'highpass': self.highpass_filter,
            'bandpass': self.bandpass_filter,
        }
        
        if method not in filter_methods:
            raise ValueError(f"Unknown filter method: {method}. "
                           f"Available: {list(filter_methods.keys())}")
        
        filter_func = filter_methods[method]
        
        # Prepare arguments
        if method in ['wiener', 'spectral_subtraction']:
            result = filter_func(audio_data.signal, noise_estimate=noise_estimate, **kwargs)
        else:
            kwargs['sample_rate'] = audio_data.sample_rate
            result = filter_func(audio_data.signal, **kwargs)
        
        filtered_audio = AudioData(
            signal=result.filtered_signal,
            sample_rate=audio_data.sample_rate,
            filename=f"{method}_{audio_data.filename}",
            duration=len(result.filtered_signal) / audio_data.sample_rate,
            channels=audio_data.channels
        )
        
        return filtered_audio, result
    
    def filter_sensor(self, sensor_data: SensorData, method: str = 'lowpass',
                      **kwargs) -> Tuple[SensorData, FilterResult]:
        """
        Apply specified filter to sensor data.
        
        Args:
            sensor_data: Input sensor data
            method: Filter method ('lowpass', 'highpass', 'bandpass', 
                    'moving_average', 'median', 'wiener')
            **kwargs: Additional parameters for the filter
            
        Returns:
            Tuple of (filtered SensorData, FilterResult)
        """
        filter_methods = {
            'wiener': self.wiener_filter,
            'spectral_subtraction': self.spectral_subtraction,
            'lowpass': self.lowpass_filter,
            'highpass': self.highpass_filter,
            'bandpass': self.bandpass_filter,
            'moving_average': self.moving_average,
            'median': self.median_filter,
        }
        
        if method not in filter_methods:
            raise ValueError(f"Unknown filter method: {method}")
        
        filter_func = filter_methods[method]
        
        # Filter each column
        filtered_values = np.zeros_like(sensor_data.values)
        
        if sensor_data.values.ndim == 1:
            values = sensor_data.values.reshape(-1, 1)
        else:
            values = sensor_data.values
        
        results = []
        for i in range(values.shape[1]):
            if method in ['lowpass', 'highpass', 'bandpass']:
                kwargs['sample_rate'] = sensor_data.sample_rate
            result = filter_func(values[:, i], **kwargs)
            filtered_values[:, i] = result.filtered_signal
            results.append(result)
        
        filtered_sensor = SensorData(
            time=sensor_data.time.copy(),
            values=filtered_values,
            filename=f"{method}_{sensor_data.filename}",
            sample_rate=sensor_data.sample_rate,
            duration=sensor_data.duration,
            columns=sensor_data.columns
        )
        
        return filtered_sensor, results[0]
    
    def _match_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Match signal length to target by trimming or padding."""
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            return np.pad(signal, (0, target_length - len(signal)))
        return signal
