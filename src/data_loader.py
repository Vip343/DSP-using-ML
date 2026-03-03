"""
Data loading utilities for audio and sensor data.
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from dataclasses import dataclass

from .config import Config


@dataclass
class AudioData:
    """Container for audio data."""
    signal: np.ndarray
    sample_rate: int
    filename: str
    duration: float
    channels: int
    
    @property
    def is_stereo(self) -> bool:
        return self.channels > 1


@dataclass
class SensorData:
    """Container for sensor data."""
    time: np.ndarray
    values: np.ndarray
    filename: str
    sample_rate: float  # Estimated from time series
    duration: float
    columns: List[str]


def sensor_to_audio(sensor_data: 'SensorData', column_idx: int = 0) -> AudioData:
    """Convert a single column of sensor data into an AudioData wrapper for AI processing."""
    if sensor_data.values.ndim > 1:
        signal = sensor_data.values[:, column_idx].astype(np.float32)
    else:
        signal = sensor_data.values.astype(np.float32)

    return AudioData(
        signal=signal,
        sample_rate=max(int(sensor_data.sample_rate), 1),
        filename=sensor_data.filename,
        duration=sensor_data.duration,
        channels=1,
    )


def audio_to_sensor(audio_data: AudioData, original_sensor: 'SensorData',
                    column_idx: int = 0) -> 'SensorData':
    """Convert AI-processed AudioData back to SensorData, preserving the original structure."""
    values = original_sensor.values.copy()
    filtered = audio_data.signal
    min_len = min(len(filtered), values.shape[0] if values.ndim > 1 else len(values))

    if values.ndim > 1:
        values[:min_len, column_idx] = filtered[:min_len]
    else:
        values[:min_len] = filtered[:min_len]

    return SensorData(
        time=original_sensor.time.copy(),
        values=values,
        filename=original_sensor.filename,
        sample_rate=original_sensor.sample_rate,
        duration=original_sensor.duration,
        columns=original_sensor.columns,
    )


class AudioLoader:
    """Loader for audio files with preprocessing capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.target_sr = config.sample_rate
    
    def load(self, filepath: Union[str, Path], target_sr: Optional[int] = None) -> AudioData:
        """
        Load an audio file and resample if necessary.
        
        Args:
            filepath: Path to audio file
            target_sr: Target sample rate (uses config if not specified)
            
        Returns:
            AudioData object containing the loaded audio
        """
        filepath = Path(filepath)
        target_sr = target_sr or self.target_sr
        
        # Load audio with librosa (automatically converts to mono and resamples)
        signal, sr = librosa.load(filepath, sr=target_sr, mono=False)
        
        # Determine channels
        if signal.ndim == 1:
            channels = 1
        else:
            channels = signal.shape[0]
            # Convert to mono for processing
            signal = librosa.to_mono(signal)
        
        duration = len(signal) / sr
        
        return AudioData(
            signal=signal,
            sample_rate=sr,
            filename=filepath.name,
            duration=duration,
            channels=channels
        )
    
    def load_all(self, directory: Optional[Path] = None) -> List[AudioData]:
        """
        Load all audio files from a directory.
        
        Args:
            directory: Directory to load from (uses config if not specified)
            
        Returns:
            List of AudioData objects
        """
        directory = directory or self.config.input_audio_dir
        audio_files = []
        
        for ext in self.config.audio_extensions:
            audio_files.extend(directory.glob(f'*{ext}'))
            audio_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return [self.load(f) for f in sorted(audio_files)]
    
    def save(self, audio_data: AudioData, filepath: Union[str, Path], 
             normalize: bool = True) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: AudioData object to save
            filepath: Output path
            normalize: Whether to normalize audio before saving
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        signal = audio_data.signal.copy()
        if normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val * 0.95
        
        sf.write(filepath, signal, audio_data.sample_rate)
    
    def add_noise(self, audio: AudioData, noise_type: str = 'white', 
                  snr_db: float = 10.0) -> Tuple[AudioData, AudioData]:
        """
        Add synthetic noise to audio signal.
        
        Args:
            audio: Clean audio data
            noise_type: Type of noise ('white', 'pink', 'babble')
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Tuple of (noisy_audio, noise_signal)
        """
        signal = audio.signal
        n_samples = len(signal)
        
        # Generate noise based on type
        if noise_type == 'white':
            noise = np.random.randn(n_samples)
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(n_samples)
        elif noise_type == 'babble':
            noise = self._generate_babble_noise(n_samples)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Calculate signal power and scale noise for desired SNR
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power_target = signal_power / 10^(SNR/10)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise_scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * noise_scale
        
        # Create noisy signal
        noisy_signal = signal + scaled_noise
        
        noisy_audio = AudioData(
            signal=noisy_signal,
            sample_rate=audio.sample_rate,
            filename=f"noisy_{noise_type}_{snr_db}dB_{audio.filename}",
            duration=audio.duration,
            channels=audio.channels
        )
        
        noise_audio = AudioData(
            signal=scaled_noise,
            sample_rate=audio.sample_rate,
            filename=f"noise_{noise_type}_{snr_db}dB.wav",
            duration=audio.duration,
            channels=1
        )
        
        return noisy_audio, noise_audio
    
    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
        # Simple approximation using filtered white noise
        white = np.random.randn(n_samples)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n_samples)
        
        return pink
    
    def _generate_babble_noise(self, n_samples: int) -> np.ndarray:
        """Generate babble-like noise (sum of multiple noise sources)."""
        # Simulate babble as sum of multiple filtered noise sources
        n_voices = 5
        babble = np.zeros(n_samples)
        
        for _ in range(n_voices):
            # Generate noise with speech-like spectral characteristics
            noise = np.random.randn(n_samples)
            # Apply bandpass-like filtering (simple moving average + high-pass)
            kernel_size = 50
            noise = np.convolve(noise, np.ones(kernel_size)/kernel_size, mode='same')
            babble += noise * np.random.uniform(0.5, 1.5)
        
        return babble / n_voices


class SensorLoader:
    """Loader for sensor data from CSV files."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load(self, filepath: Union[str, Path], 
             time_column: Optional[str] = None,
             value_columns: Optional[List[str]] = None) -> SensorData:
        """
        Load sensor data from CSV file.
        
        Args:
            filepath: Path to CSV file
            time_column: Name of time column (uses config if not specified)
            value_columns: Names of value columns (uses all non-time columns if not specified)
            
        Returns:
            SensorData object
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath)
        
        # Determine time column
        time_col = time_column or self.config.sensor_time_column
        if time_col not in df.columns:
            # Try to find a time-like column
            time_candidates = ['time', 't', 'timestamp', 'Time', 'T', 'Timestamp']
            for col in time_candidates:
                if col in df.columns:
                    time_col = col
                    break
            else:
                # Use index as time
                time = np.arange(len(df))
                time_col = None
        
        if time_col:
            time = df[time_col].values
        else:
            time = np.arange(len(df))
        
        # Determine value columns
        if value_columns:
            val_cols = value_columns
        else:
            # Use all numeric columns except time
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col and time_col in numeric_cols:
                numeric_cols.remove(time_col)
            val_cols = numeric_cols
        
        # Extract values
        values = df[val_cols].values
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        # Estimate sample rate
        if len(time) > 1:
            dt = np.median(np.diff(time))
            sample_rate = 1.0 / dt if dt > 0 else 1.0
        else:
            sample_rate = 1.0
        
        duration = time[-1] - time[0] if len(time) > 1 else 0
        
        return SensorData(
            time=time,
            values=values,
            filename=filepath.name,
            sample_rate=sample_rate,
            duration=duration,
            columns=val_cols
        )
    
    def load_all(self, directory: Optional[Path] = None) -> List[SensorData]:
        """
        Load all sensor files from a directory.
        
        Args:
            directory: Directory to load from (uses config if not specified)
            
        Returns:
            List of SensorData objects
        """
        directory = directory or self.config.input_sensor_dir
        sensor_files = []
        
        for ext in self.config.sensor_extensions:
            sensor_files.extend(directory.glob(f'*{ext}'))
        
        return [self.load(f) for f in sorted(sensor_files)]
    
    def save(self, sensor_data: SensorData, filepath: Union[str, Path]) -> None:
        """
        Save sensor data to CSV file.
        
        Args:
            sensor_data: SensorData object to save
            filepath: Output path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        data = {'time': sensor_data.time}
        for i, col in enumerate(sensor_data.columns):
            if sensor_data.values.ndim == 1:
                data[col] = sensor_data.values
            else:
                data[col] = sensor_data.values[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def add_noise(self, sensor_data: SensorData, noise_type: str = 'white',
                  snr_db: float = 10.0) -> Tuple[SensorData, np.ndarray]:
        """
        Add synthetic noise to sensor data.
        
        Args:
            sensor_data: Clean sensor data
            noise_type: Type of noise ('white', 'pink')
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Tuple of (noisy_data, noise_array)
        """
        values = sensor_data.values.copy()
        n_samples = len(values)
        
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        noisy_values = np.zeros_like(values)
        noise_array = np.zeros_like(values)
        
        for i in range(values.shape[1]):
            signal = values[:, i]
            
            # Generate noise
            if noise_type == 'white':
                noise = np.random.randn(n_samples)
            elif noise_type == 'pink':
                # Pink noise generation
                white = np.random.randn(n_samples)
                fft = np.fft.rfft(white)
                freqs = np.fft.rfftfreq(n_samples)
                freqs[0] = 1
                fft = fft / np.sqrt(freqs)
                noise = np.fft.irfft(fft, n_samples)
            else:
                noise = np.random.randn(n_samples)
            
            # Scale noise for desired SNR
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0:
                target_noise_power = signal_power / (10 ** (snr_db / 10))
                noise_scale = np.sqrt(target_noise_power / noise_power)
                scaled_noise = noise * noise_scale
            else:
                scaled_noise = noise
            
            noisy_values[:, i] = signal + scaled_noise
            noise_array[:, i] = scaled_noise
        
        noisy_data = SensorData(
            time=sensor_data.time.copy(),
            values=noisy_values,
            filename=f"noisy_{noise_type}_{snr_db}dB_{sensor_data.filename}",
            sample_rate=sensor_data.sample_rate,
            duration=sensor_data.duration,
            columns=sensor_data.columns
        )
        
        return noisy_data, noise_array
