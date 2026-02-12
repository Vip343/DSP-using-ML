"""
Configuration settings for the audio denoising project.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration class for the denoising pipeline."""
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    input_audio_dir: Path = field(default=None)
    input_sensor_dir: Path = field(default=None)
    output_filtered_dir: Path = field(default=None)
    output_plots_dir: Path = field(default=None)
    output_metrics_dir: Path = field(default=None)
    
    # Audio settings
    sample_rate: int = 16000  # 16kHz for speech processing
    audio_extensions: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.flac', '.ogg'])
    
    # Sensor settings
    sensor_extensions: List[str] = field(default_factory=lambda: ['.csv', '.txt'])
    sensor_time_column: str = 'time'
    sensor_value_column: str = 'value'
    
    # DSP Filter settings
    wiener_frame_length: int = 512
    wiener_hop_length: int = 256
    spectral_sub_frame_length: int = 512
    spectral_sub_hop_length: int = 256
    spectral_sub_alpha: float = 2.0  # Oversubtraction factor
    spectral_sub_beta: float = 0.01  # Spectral floor
    lowpass_cutoff: float = 4000.0  # Hz
    highpass_cutoff: float = 100.0  # Hz
    filter_order: int = 5
    
    # AI Model settings
    use_deepfilternet: bool = True  # DeepFilterNet (original model)
    use_deepfilternet2: bool = True  # DeepFilterNet2 (improved model)
    use_deepfilternet_hf: bool = False  # DeepFilterNet2 via Hugging Face API (disabled)
    use_speechbrain: bool = True  # SpeechBrain MetricGAN+ speech enhancement
    use_rnnoise: bool = False  # RNNoise - not available on macOS
    use_demucs: bool = True
    use_noisereduce: bool = True  # Spectral gating - works without Rust
    demucs_model: str = 'htdemucs'  # Model variant
    deepfilternet_model: str = 'DeepFilterNet'  # 'DeepFilterNet' or 'DeepFilterNet2'
    device: str = 'cpu'  # 'cpu' or 'cuda'
    
    # Noise settings for synthetic noise generation
    noise_types: List[str] = field(default_factory=lambda: ['white', 'pink', 'babble'])
    snr_levels: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20])  # dB
    
    # Visualization settings
    figure_dpi: int = 150
    figure_format: str = 'png'
    spectrogram_n_fft: int = 2048
    spectrogram_hop_length: int = 512
    
    # Metrics settings
    pesq_mode: str = 'wb'  # 'nb' for narrowband, 'wb' for wideband
    
    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        if self.input_audio_dir is None:
            self.input_audio_dir = self.project_root / 'input' / 'audio'
        if self.input_sensor_dir is None:
            self.input_sensor_dir = self.project_root / 'input' / 'sensor'
        if self.output_filtered_dir is None:
            self.output_filtered_dir = self.project_root / 'output' / 'filtered'
        if self.output_plots_dir is None:
            self.output_plots_dir = self.project_root / 'output' / 'plots'
        if self.output_metrics_dir is None:
            self.output_metrics_dir = self.project_root / 'output' / 'metrics'
        
        # Ensure directories exist
        self.input_audio_dir.mkdir(parents=True, exist_ok=True)
        self.input_sensor_dir.mkdir(parents=True, exist_ok=True)
        self.output_filtered_dir.mkdir(parents=True, exist_ok=True)
        self.output_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            'project_root': str(self.project_root),
            'input_audio_dir': str(self.input_audio_dir),
            'input_sensor_dir': str(self.input_sensor_dir),
            'output_filtered_dir': str(self.output_filtered_dir),
            'output_plots_dir': str(self.output_plots_dir),
            'output_metrics_dir': str(self.output_metrics_dir),
            'sample_rate': self.sample_rate,
            'audio_extensions': self.audio_extensions,
            'sensor_extensions': self.sensor_extensions,
            'wiener_frame_length': self.wiener_frame_length,
            'wiener_hop_length': self.wiener_hop_length,
            'spectral_sub_frame_length': self.spectral_sub_frame_length,
            'spectral_sub_hop_length': self.spectral_sub_hop_length,
            'spectral_sub_alpha': self.spectral_sub_alpha,
            'spectral_sub_beta': self.spectral_sub_beta,
            'lowpass_cutoff': self.lowpass_cutoff,
            'highpass_cutoff': self.highpass_cutoff,
            'filter_order': self.filter_order,
            'use_deepfilternet': self.use_deepfilternet,
            'use_demucs': self.use_demucs,
            'demucs_model': self.demucs_model,
            'device': self.device,
            'noise_types': self.noise_types,
            'snr_levels': self.snr_levels,
            'figure_dpi': self.figure_dpi,
            'figure_format': self.figure_format,
            'spectrogram_n_fft': self.spectrogram_n_fft,
            'spectrogram_hop_length': self.spectrogram_hop_length,
            'pesq_mode': self.pesq_mode,
        }
