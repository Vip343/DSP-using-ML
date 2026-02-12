"""
Audio Denoising Thesis - Source Package

This package provides tools for comparing classical DSP filters
with pretrained AI models for audio denoising.
"""

from .data_loader import AudioLoader, SensorLoader
from .dsp_filters import DSPFilters
from .ai_denoisers import AIDenoiser
from .metrics import MetricsCalculator
from .visualization import Visualizer
from .config import Config

__all__ = [
    'AudioLoader',
    'SensorLoader', 
    'DSPFilters',
    'AIDenoiser',
    'MetricsCalculator',
    'Visualizer',
    'Config'
]

__version__ = '1.0.0'
