"""
Visualization utilities for audio denoising analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from .config import Config
from .data_loader import AudioData, SensorData
from .metrics import MetricsResult


class Visualizer:
    """Visualization tools for audio and sensor data analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_waveform(self, audio_data: AudioData, 
                      ax: Optional[plt.Axes] = None,
                      title: Optional[str] = None,
                      color: str = '#2E86AB') -> plt.Axes:
        """
        Plot audio waveform.
        
        Args:
            audio_data: Audio data to plot
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            color: Waveform color
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        
        time = np.arange(len(audio_data.signal)) / audio_data.sample_rate
        ax.plot(time, audio_data.signal, color=color, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title or audio_data.filename)
        ax.set_xlim([0, time[-1]])
        
        return ax
    
    def plot_spectrogram(self, audio_data: AudioData,
                         ax: Optional[plt.Axes] = None,
                         title: Optional[str] = None,
                         n_fft: Optional[int] = None,
                         hop_length: Optional[int] = None,
                         cmap: str = 'magma') -> plt.Axes:
        """
        Plot audio spectrogram.
        
        Args:
            audio_data: Audio data to plot
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            n_fft: FFT window size
            hop_length: Hop length for STFT
            cmap: Colormap
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        n_fft = n_fft or self.config.spectrogram_n_fft
        hop_length = hop_length or self.config.spectrogram_hop_length
        
        # Compute spectrogram
        D = librosa.stft(audio_data.signal, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Plot
        img = librosa.display.specshow(
            S_db, 
            sr=audio_data.sample_rate, 
            hop_length=hop_length,
            x_axis='time', 
            y_axis='hz',
            ax=ax,
            cmap=cmap
        )
        
        ax.set_title(title or f"Spectrogram: {audio_data.filename}")
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return ax
    
    def plot_waveform_comparison(self, 
                                  clean: Optional[AudioData],
                                  noisy: AudioData,
                                  filtered_dict: Dict[str, AudioData],
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot waveform comparison across methods.
        
        Args:
            clean: Clean reference audio (optional)
            noisy: Noisy input audio
            filtered_dict: Dictionary mapping method name to filtered AudioData
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_plots = 1 + len(filtered_dict) + (1 if clean is not None else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_plots))
        idx = 0
        
        # Plot clean if available
        if clean is not None:
            self.plot_waveform(clean, ax=axes[idx], 
                             title="Clean Reference", color=colors[idx])
            idx += 1
        
        # Plot noisy
        self.plot_waveform(noisy, ax=axes[idx], 
                          title="Noisy Input", color=colors[idx])
        idx += 1
        
        # Plot each filtered version
        for method, audio in filtered_dict.items():
            self.plot_waveform(audio, ax=axes[idx], 
                             title=f"Filtered: {method}", color=colors[idx])
            idx += 1
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_spectrogram_comparison(self,
                                     clean: Optional[AudioData],
                                     noisy: AudioData,
                                     filtered_dict: Dict[str, AudioData],
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot spectrogram comparison in a grid.
        
        Args:
            clean: Clean reference audio (optional)
            noisy: Noisy input audio
            filtered_dict: Dictionary mapping method name to filtered AudioData
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_plots = 1 + len(filtered_dict) + (1 if clean is not None else 0)
        
        # Calculate grid dimensions
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        idx = 0
        
        # Plot clean if available
        if clean is not None:
            self.plot_spectrogram(clean, ax=axes[idx], title="Clean Reference")
            idx += 1
        
        # Plot noisy
        self.plot_spectrogram(noisy, ax=axes[idx], title="Noisy Input")
        idx += 1
        
        # Plot each filtered version
        for method, audio in filtered_dict.items():
            if idx < len(axes):
                self.plot_spectrogram(audio, ax=axes[idx], title=f"Filtered: {method}")
                idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_metrics_comparison(self, 
                                 results: List[MetricsResult],
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot bar chart comparing metrics across methods.
        
        Args:
            results: List of MetricsResult
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Create dataframe
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Metrics to plot
        metrics = ['SNR_improvement_dB', 'SDR_dB', 'PESQ', 'STOI']
        available_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
        
        if not available_metrics:
            print("No metrics available to plot")
            return None
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                figsize=(4 * len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(df['method'].unique()))
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Group by method
            grouped = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
            
            bars = ax.bar(grouped['method'], grouped['mean'], 
                         yerr=grouped['std'], capsize=5, color=colors)
            
            ax.set_xlabel('Method')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, grouped['mean']):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_metrics_boxplot(self,
                              results: List[MetricsResult],
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot boxplots comparing metrics distribution across methods.
        
        Args:
            results: List of MetricsResult
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        df = pd.DataFrame([r.to_dict() for r in results])
        
        metrics = ['SNR_improvement_dB', 'SDR_dB', 'PESQ', 'STOI']
        available_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
        
        if not available_metrics:
            return None
        
        fig, axes = plt.subplots(1, len(available_metrics),
                                figsize=(4 * len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            sns.boxplot(x='method', y=metric, data=df, ax=ax, palette='husl')
            ax.set_xlabel('Method')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Distribution')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_signal_and_filter_response(self,
                                         noisy: AudioData,
                                         filtered: AudioData,
                                         method: str,
                                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot signal before/after filtering with frequency response.
        
        Args:
            noisy: Noisy input audio
            filtered: Filtered audio
            method: Filter method name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Time domain - noisy
        ax1 = fig.add_subplot(gs[0, 0])
        time = np.arange(len(noisy.signal)) / noisy.sample_rate
        ax1.plot(time, noisy.signal, color='#E74C3C', linewidth=0.5, alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Noisy Signal (Time Domain)')
        
        # Time domain - filtered
        ax2 = fig.add_subplot(gs[0, 1])
        time_f = np.arange(len(filtered.signal)) / filtered.sample_rate
        ax2.plot(time_f, filtered.signal, color='#27AE60', linewidth=0.5, alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Filtered Signal ({method})')
        
        # Frequency domain - noisy
        ax3 = fig.add_subplot(gs[1, 0])
        freqs_n, psd_n = self._compute_psd(noisy.signal, noisy.sample_rate)
        ax3.semilogy(freqs_n, psd_n, color='#E74C3C', linewidth=0.8)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power Spectral Density')
        ax3.set_title('Noisy Signal (Frequency Domain)')
        ax3.set_xlim([0, noisy.sample_rate / 2])
        
        # Frequency domain - filtered
        ax4 = fig.add_subplot(gs[1, 1])
        freqs_f, psd_f = self._compute_psd(filtered.signal, filtered.sample_rate)
        ax4.semilogy(freqs_f, psd_f, color='#27AE60', linewidth=0.8)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Power Spectral Density')
        ax4.set_title(f'Filtered Signal ({method})')
        ax4.set_xlim([0, filtered.sample_rate / 2])
        
        # Spectrogram comparison
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_spectrogram(noisy, ax=ax5, title='Noisy Spectrogram')
        
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_spectrogram(filtered, ax=ax6, title=f'Filtered Spectrogram ({method})')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_sensor_data(self,
                          clean: Optional[SensorData],
                          noisy: SensorData,
                          filtered_dict: Dict[str, SensorData],
                          column_idx: int = 0,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot sensor data comparison.
        
        Args:
            clean: Clean reference data (optional)
            noisy: Noisy sensor data
            filtered_dict: Dictionary mapping method name to filtered data
            column_idx: Index of column to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_plots = 1 + len(filtered_dict) + (1 if clean is not None else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
        
        if n_plots == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_plots))
        idx = 0
        
        def get_values(data):
            if data.values.ndim > 1:
                return data.values[:, column_idx]
            return data.values
        
        # Plot clean if available
        if clean is not None:
            axes[idx].plot(clean.time, get_values(clean), 
                          color=colors[idx], linewidth=0.8)
            axes[idx].set_ylabel('Value')
            axes[idx].set_title('Clean Reference')
            idx += 1
        
        # Plot noisy
        axes[idx].plot(noisy.time, get_values(noisy), 
                      color=colors[idx], linewidth=0.8)
        axes[idx].set_ylabel('Value')
        axes[idx].set_title('Noisy Input')
        idx += 1
        
        # Plot each filtered version
        for method, data in filtered_dict.items():
            axes[idx].plot(data.time, get_values(data), 
                          color=colors[idx], linewidth=0.8)
            axes[idx].set_ylabel('Value')
            axes[idx].set_title(f'Filtered: {method}')
            idx += 1
        
        axes[-1].set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_summary_figure(self,
                               clean: Optional[AudioData],
                               noisy: AudioData,
                               filtered_dict: Dict[str, AudioData],
                               metrics_results: List[MetricsResult],
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive summary figure with all visualizations.
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            filtered_dict: Dictionary of filtered audio
            metrics_results: List of metrics results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        n_methods = len(filtered_dict)
        
        # Create grid
        gs = gridspec.GridSpec(4, n_methods + 2, figure=fig, 
                              height_ratios=[1, 1, 1, 1.2])
        
        # Row 1: Waveforms
        if clean is not None:
            ax_clean = fig.add_subplot(gs[0, 0])
            self.plot_waveform(clean, ax=ax_clean, title="Clean", color='#2ECC71')
        
        ax_noisy = fig.add_subplot(gs[0, 1])
        self.plot_waveform(noisy, ax=ax_noisy, title="Noisy", color='#E74C3C')
        
        for i, (method, audio) in enumerate(filtered_dict.items()):
            ax = fig.add_subplot(gs[0, 2 + i])
            self.plot_waveform(audio, ax=ax, title=method, color='#3498DB')
        
        # Row 2: Spectrograms
        if clean is not None:
            ax_spec_clean = fig.add_subplot(gs[1, 0])
            self.plot_spectrogram(clean, ax=ax_spec_clean, title="Clean")
        
        ax_spec_noisy = fig.add_subplot(gs[1, 1])
        self.plot_spectrogram(noisy, ax=ax_spec_noisy, title="Noisy")
        
        for i, (method, audio) in enumerate(filtered_dict.items()):
            ax = fig.add_subplot(gs[1, 2 + i])
            self.plot_spectrogram(audio, ax=ax, title=method)
        
        # Row 3: Frequency domain
        if clean is not None:
            ax_freq_clean = fig.add_subplot(gs[2, 0])
            freqs, psd = self._compute_psd(clean.signal, clean.sample_rate)
            ax_freq_clean.semilogy(freqs, psd, color='#2ECC71', linewidth=0.8)
            ax_freq_clean.set_title('Clean PSD')
            ax_freq_clean.set_xlabel('Frequency (Hz)')
        
        ax_freq_noisy = fig.add_subplot(gs[2, 1])
        freqs, psd = self._compute_psd(noisy.signal, noisy.sample_rate)
        ax_freq_noisy.semilogy(freqs, psd, color='#E74C3C', linewidth=0.8)
        ax_freq_noisy.set_title('Noisy PSD')
        ax_freq_noisy.set_xlabel('Frequency (Hz)')
        
        for i, (method, audio) in enumerate(filtered_dict.items()):
            ax = fig.add_subplot(gs[2, 2 + i])
            freqs, psd = self._compute_psd(audio.signal, audio.sample_rate)
            ax.semilogy(freqs, psd, color='#3498DB', linewidth=0.8)
            ax.set_title(f'{method} PSD')
            ax.set_xlabel('Frequency (Hz)')
        
        # Row 4: Metrics comparison
        ax_metrics = fig.add_subplot(gs[3, :])
        self._plot_metrics_table(metrics_results, ax_metrics)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _compute_psd(self, signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        from scipy import signal as sig
        freqs, psd = sig.welch(signal, sample_rate, nperseg=1024)
        return freqs, psd
    
    def _plot_metrics_table(self, results: List[MetricsResult], ax: plt.Axes) -> None:
        """Plot metrics as a table."""
        if not results:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            return
            
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Select columns for display (method must be present for groupby)
        metric_cols = ['SNR_improvement_dB', 'SDR_dB', 'PESQ', 'STOI']
        available_metric_cols = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
        
        if not available_metric_cols or 'method' not in df.columns:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            return
        
        df_display = df[['method'] + available_metric_cols].groupby('method').mean().round(3)
        
        ax.axis('off')
        table = ax.table(
            cellText=df_display.values,
            rowLabels=df_display.index,
            colLabels=available_metric_cols,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    def _save_figure(self, fig: plt.Figure, path: Path) -> None:
        """Save figure to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            path, 
            dpi=self.config.figure_dpi,
            format=self.config.figure_format,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Figure saved to: {path}")
        plt.close(fig)
    
    def save_all_plots(self,
                        clean: Optional[AudioData],
                        noisy: AudioData,
                        filtered_dict: Dict[str, AudioData],
                        metrics_results: List[MetricsResult],
                        base_name: str,
                        output_dir: Optional[Path] = None) -> List[Path]:
        """
        Save all visualization plots for an audio file.
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            filtered_dict: Dictionary of filtered audio
            metrics_results: List of metrics results
            base_name: Base name for output files
            output_dir: Output directory (uses config default if not specified)
            
        Returns:
            List of saved file paths
        """
        output_dir = output_dir or self.config.output_plots_dir
        saved_paths = []
        
        # Waveform comparison
        path = output_dir / f"waveforms.{self.config.figure_format}"
        self.plot_waveform_comparison(clean, noisy, filtered_dict, save_path=path)
        saved_paths.append(path)
        
        # Spectrogram comparison
        path = output_dir / f"spectrograms.{self.config.figure_format}"
        self.plot_spectrogram_comparison(clean, noisy, filtered_dict, save_path=path)
        saved_paths.append(path)
        
        # Individual filter analysis
        for method, audio in filtered_dict.items():
            path = output_dir / f"{method}_analysis.{self.config.figure_format}"
            self.plot_signal_and_filter_response(noisy, audio, method, save_path=path)
            saved_paths.append(path)
        
        # Metrics comparison
        if metrics_results:
            path = output_dir / f"metrics_bars.{self.config.figure_format}"
            self.plot_metrics_comparison(metrics_results, save_path=path)
            saved_paths.append(path)
            
            path = output_dir / f"metrics_boxplot.{self.config.figure_format}"
            self.plot_metrics_boxplot(metrics_results, save_path=path)
            saved_paths.append(path)
        
        # Summary figure
        path = output_dir / f"summary.{self.config.figure_format}"
        self.create_summary_figure(clean, noisy, filtered_dict, metrics_results, save_path=path)
        saved_paths.append(path)
        
        return saved_paths
