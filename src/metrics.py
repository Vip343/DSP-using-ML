"""
Evaluation metrics for audio denoising quality assessment.

Implements:
- SNR (Signal-to-Noise Ratio)
- SDR (Signal-to-Distortion Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

from .config import Config
from .data_loader import AudioData, SensorData


@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    filename: str
    method: str
    snr_input: float = np.nan
    snr_output: float = np.nan
    snr_improvement: float = np.nan
    sdr: float = np.nan
    pesq: float = np.nan
    stoi: float = np.nan
    si_sdr: float = np.nan  # Scale-Invariant SDR
    
    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'method': self.method,
            'SNR_input_dB': self.snr_input,
            'SNR_output_dB': self.snr_output,
            'SNR_improvement_dB': self.snr_improvement,
            'SDR_dB': self.sdr,
            'SI-SDR_dB': self.si_sdr,
            'PESQ': self.pesq,
            'STOI': self.stoi,
        }


class MetricsCalculator:
    """Calculator for audio quality metrics."""
    
    def __init__(self, config: Config):
        self.config = config
        self._pesq_available = self._check_pesq()
        self._stoi_available = self._check_stoi()
        self._mir_eval_available = self._check_mir_eval()
    
    def _check_pesq(self) -> bool:
        try:
            from pesq import pesq
            return True
        except ImportError:
            warnings.warn("PESQ not available. Install with: pip install pesq")
            return False
    
    def _check_stoi(self) -> bool:
        try:
            from pystoi import stoi
            return True
        except ImportError:
            warnings.warn("STOI not available. Install with: pip install pystoi")
            return False
    
    def _check_mir_eval(self) -> bool:
        try:
            import mir_eval
            return True
        except ImportError:
            warnings.warn("mir_eval not available. Install with: pip install mir_eval")
            return False
    
    def calculate_snr(self, clean_signal: np.ndarray, 
                      noisy_signal: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.
        
        SNR = 10 * log10(signal_power / noise_power)
        
        Args:
            clean_signal: Reference clean signal
            noisy_signal: Noisy or processed signal
            
        Returns:
            SNR in dB
        """
        # Ensure same length
        min_len = min(len(clean_signal), len(noisy_signal))
        clean = clean_signal[:min_len]
        noisy = noisy_signal[:min_len]
        
        # Calculate noise as difference
        noise = noisy - clean
        
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_sdr(self, reference: np.ndarray, 
                      estimated: np.ndarray) -> float:
        """
        Calculate Signal-to-Distortion Ratio (SDR) using mir_eval.
        
        Args:
            reference: Reference clean signal
            estimated: Estimated/processed signal
            
        Returns:
            SDR in dB
        """
        if not self._mir_eval_available:
            return self._calculate_sdr_simple(reference, estimated)
        
        try:
            import mir_eval
            
            # Ensure same length
            min_len = min(len(reference), len(estimated))
            ref = reference[:min_len].reshape(1, -1)
            est = estimated[:min_len].reshape(1, -1)
            
            sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref, est)
            return float(sdr[0])
        except Exception as e:
            warnings.warn(f"mir_eval SDR failed: {e}")
            return self._calculate_sdr_simple(reference, estimated)
    
    def _calculate_sdr_simple(self, reference: np.ndarray, 
                               estimated: np.ndarray) -> float:
        """Simple SDR calculation without mir_eval."""
        min_len = min(len(reference), len(estimated))
        ref = reference[:min_len]
        est = estimated[:min_len]
        
        # Optimal scaling factor
        alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
        scaled_ref = alpha * ref
        
        noise = est - scaled_ref
        
        signal_power = np.sum(scaled_ref ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        sdr = 10 * np.log10(signal_power / noise_power)
        return sdr
    
    def calculate_si_sdr(self, reference: np.ndarray, 
                         estimated: np.ndarray) -> float:
        """
        Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
        
        SI-SDR is more robust to volume differences between signals.
        
        Args:
            reference: Reference clean signal
            estimated: Estimated/processed signal
            
        Returns:
            SI-SDR in dB
        """
        min_len = min(len(reference), len(estimated))
        ref = reference[:min_len]
        est = estimated[:min_len]
        
        # Remove mean
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        
        # Compute SI-SDR
        alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
        s_target = alpha * ref
        e_noise = est - s_target
        
        si_sdr = 10 * np.log10(
            np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-10)
        )
        
        return si_sdr
    
    def calculate_pesq(self, reference: np.ndarray, 
                       degraded: np.ndarray,
                       sample_rate: int) -> float:
        """
        Calculate PESQ (Perceptual Evaluation of Speech Quality).
        
        PESQ is an ITU standard for speech quality assessment.
        Score ranges from -0.5 to 4.5 (higher is better).
        
        Args:
            reference: Reference clean signal
            degraded: Degraded/processed signal
            sample_rate: Sample rate (8000 for narrowband, 16000 for wideband)
            
        Returns:
            PESQ score
        """
        if not self._pesq_available:
            warnings.warn("PESQ not available")
            return np.nan
        
        try:
            from pesq import pesq
            
            # Ensure same length
            min_len = min(len(reference), len(degraded))
            ref = reference[:min_len]
            deg = degraded[:min_len]
            
            # PESQ requires 8kHz or 16kHz
            mode = self.config.pesq_mode
            
            if sample_rate == 8000:
                mode = 'nb'
            elif sample_rate == 16000:
                mode = 'wb'
            else:
                # Resample to 16kHz
                import librosa
                ref = librosa.resample(ref, orig_sr=sample_rate, target_sr=16000)
                deg = librosa.resample(deg, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
                mode = 'wb'
            
            score = pesq(sample_rate, ref, deg, mode)
            return score
            
        except Exception as e:
            warnings.warn(f"PESQ calculation failed: {e}")
            return np.nan
    
    def calculate_stoi(self, reference: np.ndarray, 
                       degraded: np.ndarray,
                       sample_rate: int,
                       extended: bool = False) -> float:
        """
        Calculate STOI (Short-Time Objective Intelligibility).
        
        STOI predicts speech intelligibility. Score ranges from 0 to 1.
        
        Args:
            reference: Reference clean signal
            degraded: Degraded/processed signal
            sample_rate: Sample rate
            extended: Use extended STOI (better for processed speech)
            
        Returns:
            STOI score (0-1)
        """
        if not self._stoi_available:
            warnings.warn("STOI not available")
            return np.nan
        
        try:
            from pystoi import stoi
            
            # Ensure same length
            min_len = min(len(reference), len(degraded))
            ref = reference[:min_len]
            deg = degraded[:min_len]
            
            score = stoi(ref, deg, sample_rate, extended=extended)
            return score
            
        except Exception as e:
            warnings.warn(f"STOI calculation failed: {e}")
            return np.nan
    
    def calculate_all(self, clean_audio: AudioData,
                      noisy_audio: AudioData,
                      denoised_audio: AudioData,
                      method: str) -> MetricsResult:
        """
        Calculate all metrics for a denoised audio.
        
        Args:
            clean_audio: Reference clean audio
            noisy_audio: Noisy input audio
            denoised_audio: Denoised output audio
            method: Name of the denoising method
            
        Returns:
            MetricsResult with all computed metrics
        """
        clean = clean_audio.signal
        noisy = noisy_audio.signal
        denoised = denoised_audio.signal
        sr = clean_audio.sample_rate
        
        # Ensure same lengths
        min_len = min(len(clean), len(noisy), len(denoised))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        denoised = denoised[:min_len]
        
        # Calculate metrics
        snr_input = self.calculate_snr(clean, noisy)
        snr_output = self.calculate_snr(clean, denoised)
        snr_improvement = snr_output - snr_input
        
        sdr = self.calculate_sdr(clean, denoised)
        si_sdr = self.calculate_si_sdr(clean, denoised)
        pesq_score = self.calculate_pesq(clean, denoised, sr)
        stoi_score = self.calculate_stoi(clean, denoised, sr)
        
        return MetricsResult(
            filename=clean_audio.filename,
            method=method,
            snr_input=snr_input,
            snr_output=snr_output,
            snr_improvement=snr_improvement,
            sdr=sdr,
            si_sdr=si_sdr,
            pesq=pesq_score,
            stoi=stoi_score
        )
    
    def calculate_sensor_metrics(self, clean_data: SensorData,
                                 noisy_data: SensorData,
                                 filtered_data: SensorData,
                                 method: str) -> MetricsResult:
        """
        Calculate metrics for sensor data denoising.
        
        For sensor data, only SNR and SDR are applicable.
        
        Args:
            clean_data: Reference clean sensor data
            noisy_data: Noisy input data
            filtered_data: Filtered output data
            method: Name of the filtering method
            
        Returns:
            MetricsResult with applicable metrics
        """
        # Use first column for metrics
        clean = clean_data.values[:, 0] if clean_data.values.ndim > 1 else clean_data.values
        noisy = noisy_data.values[:, 0] if noisy_data.values.ndim > 1 else noisy_data.values
        filtered = filtered_data.values[:, 0] if filtered_data.values.ndim > 1 else filtered_data.values
        
        # Ensure same lengths
        min_len = min(len(clean), len(noisy), len(filtered))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        filtered = filtered[:min_len]
        
        snr_input = self.calculate_snr(clean, noisy)
        snr_output = self.calculate_snr(clean, filtered)
        snr_improvement = snr_output - snr_input
        
        sdr = self.calculate_sdr(clean, filtered)
        si_sdr = self.calculate_si_sdr(clean, filtered)
        
        return MetricsResult(
            filename=clean_data.filename,
            method=method,
            snr_input=snr_input,
            snr_output=snr_output,
            snr_improvement=snr_improvement,
            sdr=sdr,
            si_sdr=si_sdr,
            pesq=np.nan,  # Not applicable for sensor data
            stoi=np.nan   # Not applicable for sensor data
        )
    
    def results_to_dataframe(self, results: List[MetricsResult]) -> pd.DataFrame:
        """Convert list of MetricsResult to pandas DataFrame."""
        data = [r.to_dict() for r in results]
        df = pd.DataFrame(data)
        return df
    
    def save_results(self, results: List[MetricsResult], 
                     filepath: Optional[Path] = None,
                     format: str = 'csv') -> Path:
        """
        Save metrics results to file.
        
        Args:
            results: List of MetricsResult
            filepath: Output path (auto-generated if not provided)
            format: Output format ('csv' or 'json')
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.{format}"
            filepath = self.config.output_metrics_dir / filename
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.results_to_dataframe(results)
        
        if format == 'csv':
            df.to_csv(filepath, index=False, float_format='%.4f')
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Metrics saved to: {filepath}")
        return filepath
    
    def print_summary(self, results: List[MetricsResult]) -> None:
        """Print summary statistics of results."""
        df = self.results_to_dataframe(results)
        
        print("\n" + "="*80)
        print("DENOISING EVALUATION SUMMARY")
        print("="*80)
        
        # Group by method
        numeric_cols = ['SNR_input_dB', 'SNR_output_dB', 'SNR_improvement_dB', 
                       'SDR_dB', 'SI-SDR_dB', 'PESQ', 'STOI']
        
        summary = df.groupby('method')[numeric_cols].agg(['mean', 'std'])
        
        for method in df['method'].unique():
            print(f"\n{method.upper()}")
            print("-" * 40)
            method_data = df[df['method'] == method]
            for col in numeric_cols:
                values = method_data[col].dropna()
                if len(values) > 0:
                    print(f"  {col}: {values.mean():.4f} ± {values.std():.4f}")
        
        print("\n" + "="*80)
    
    def compare_methods(self, results: List[MetricsResult]) -> pd.DataFrame:
        """
        Create comparison table across methods.
        
        Returns DataFrame with methods as rows and metrics as columns.
        """
        df = self.results_to_dataframe(results)
        
        numeric_cols = ['SNR_improvement_dB', 'SDR_dB', 'SI-SDR_dB', 'PESQ', 'STOI']
        
        comparison = df.groupby('method')[numeric_cols].mean()
        comparison = comparison.round(4)
        
        return comparison
