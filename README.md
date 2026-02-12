# Audio Denoising Thesis Project

A comprehensive Python project comparing classical DSP filters with pretrained AI models for audio denoising. This project supports both audio and sensor data processing, providing quantitative metrics and visualizations for thesis research.

## Project Overview

This project implements and compares:

### Classical DSP Methods
- **Wiener Filter**: Optimal linear filter minimizing mean square error
- **Spectral Subtraction**: Subtracts estimated noise spectrum from noisy signal
- **Lowpass/Highpass/Bandpass Filters**: Butterworth filters for frequency-based noise removal
- **Moving Average**: Simple smoothing filter for sensor data
- **Median Filter**: Effective for impulse noise removal

### AI-Based Methods
- **DeepFilterNet2**: Real-time speech enhancement neural network
- **Demucs**: Audio source separation model (extracts vocals/speech)

### Evaluation Metrics
- **SNR** (Signal-to-Noise Ratio): Measures noise reduction in dB
- **SDR** (Signal-to-Distortion Ratio): Measures signal quality preservation
- **SI-SDR** (Scale-Invariant SDR): Robust to volume differences
- **PESQ** (Perceptual Evaluation of Speech Quality): ITU standard for speech quality
- **STOI** (Short-Time Objective Intelligibility): Predicts speech intelligibility

## Project Structure

```
audio-denoising-thesis/
├── input/
│   ├── audio/           # Place .wav files here
│   └── sensor/          # Place .csv sensor data here
├── output/
│   ├── filtered/        # Denoised audio/sensor outputs
│   ├── plots/           # Spectrograms, waveforms, comparisons
│   └── metrics/         # CSV/JSON evaluation results
├── src/
│   ├── __init__.py
│   ├── config.py        # Configuration settings
│   ├── data_loader.py   # Load audio and sensor data
│   ├── dsp_filters.py   # Classical DSP filters
│   ├── ai_denoisers.py  # DeepFilterNet2 and Demucs wrappers
│   ├── metrics.py       # SNR, SDR, PESQ, STOI calculations
│   └── visualization.py # Spectrograms, comparison plots
├── main.py              # Main pipeline script
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install AI Models (Optional but Recommended)

For DeepFilterNet2:
```bash
pip install deepfilternet
```

For Demucs:
```bash
pip install demucs
```

**Note**: These models require PyTorch. For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Generate Sample Data

```bash
python main.py --generate-samples
```

This creates test audio and sensor files with various noise levels.

### 2. Run the Full Pipeline

```bash
python main.py
```

### 3. View Results

- **Filtered audio**: `output/filtered/`
- **Visualizations**: `output/plots/`
- **Metrics**: `output/metrics/`

## Usage Guide

### Adding Your Own Data

#### Audio Files

1. Place `.wav` files in `input/audio/`
2. For metrics calculation, use naming convention:
   - `clean_filename.wav` - Clean reference
   - `noisy_filename.wav` - Noisy version

Example:
```
input/audio/
├── clean_speech1.wav
├── noisy_speech1.wav
├── clean_speech2.wav
└── noisy_speech2.wav
```

#### Sensor Data

1. Place `.csv` files in `input/sensor/`
2. CSV format should include a time column and value column(s):

```csv
time,value
0.0,1.234
0.01,1.567
0.02,1.890
...
```

### Command Line Options

```bash
# Process only audio files
python main.py --audio-only

# Process only sensor files
python main.py --sensor-only

# Skip plot generation (faster)
python main.py --no-plots

# Use GPU for AI models
python main.py --device cuda

# Disable specific AI models
python main.py --no-deepfilternet
python main.py --no-demucs

# Set sample rate (default: 16000 Hz)
python main.py --sample-rate 44100
```

### Using as a Library

```python
from src.config import Config
from src.data_loader import AudioLoader
from src.dsp_filters import DSPFilters
from src.ai_denoisers import AIDenoiser
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

# Initialize
config = Config(sample_rate=16000, device='cpu')
loader = AudioLoader(config)
dsp = DSPFilters(config)
ai = AIDenoiser(config)
metrics = MetricsCalculator(config)
viz = Visualizer(config)

# Load audio
audio = loader.load('path/to/audio.wav')

# Apply DSP filter
filtered, result = dsp.filter_audio(audio, method='wiener')

# Apply AI denoiser
denoised, ai_result = ai.denoise(audio, method='deepfilternet')

# Calculate metrics (requires clean reference)
clean_audio = loader.load('path/to/clean.wav')
noisy_audio = loader.load('path/to/noisy.wav')
metrics_result = metrics.calculate_all(clean_audio, noisy_audio, denoised, 'deepfilternet')
print(f"SNR improvement: {metrics_result.snr_improvement:.2f} dB")
print(f"PESQ: {metrics_result.pesq:.2f}")
```

## Configuration Options

Edit `src/config.py` or pass parameters to `Config()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Target sample rate for audio |
| `device` | 'cpu' | Device for AI models ('cpu' or 'cuda') |
| `use_deepfilternet` | True | Enable DeepFilterNet2 |
| `use_demucs` | True | Enable Demucs |
| `wiener_frame_length` | 512 | Frame length for Wiener filter |
| `spectral_sub_alpha` | 2.0 | Oversubtraction factor |
| `spectral_sub_beta` | 0.01 | Spectral floor |
| `lowpass_cutoff` | 4000 | Lowpass filter cutoff (Hz) |
| `highpass_cutoff` | 100 | Highpass filter cutoff (Hz) |
| `figure_dpi` | 150 | DPI for saved figures |

## Output Files

### Metrics (CSV/JSON)

```csv
filename,method,SNR_input_dB,SNR_output_dB,SNR_improvement_dB,SDR_dB,SI-SDR_dB,PESQ,STOI
sample.wav,wiener,5.0,12.3,7.3,8.5,7.8,2.1,0.78
sample.wav,deepfilternet,5.0,18.5,13.5,14.2,13.9,3.2,0.91
```

### Visualizations

- `*_waveforms.png`: Time-domain comparison
- `*_spectrograms.png`: Frequency-domain comparison
- `*_analysis.png`: Detailed filter analysis
- `*_metrics_bars.png`: Bar chart of metrics
- `*_summary.png`: Comprehensive comparison

## Interpreting Results

### Metrics Guide

| Metric | Range | Interpretation |
|--------|-------|----------------|
| SNR improvement | -∞ to +∞ dB | Higher = better noise reduction |
| SDR | -∞ to +∞ dB | Higher = less distortion |
| PESQ | -0.5 to 4.5 | 4.5 = excellent quality |
| STOI | 0 to 1 | 1 = perfectly intelligible |

### Expected Results

Based on literature, for speech denoising at 10 dB input SNR:

| Method | SNR Improvement | PESQ | STOI |
|--------|----------------|------|------|
| Wiener Filter | 3-5 dB | 2.0-2.5 | 0.70-0.80 |
| Spectral Subtraction | 2-4 dB | 1.8-2.3 | 0.65-0.75 |
| DeepFilterNet2 | 8-12 dB | 3.0-3.5 | 0.85-0.95 |
| Demucs | 6-10 dB | 2.8-3.3 | 0.80-0.90 |

## Troubleshooting

### Common Issues

1. **"DeepFilterNet not available"**
   ```bash
   pip install deepfilternet
   ```

2. **"PESQ calculation failed"**
   - PESQ requires 8kHz or 16kHz audio
   - Install pesq: `pip install pesq`

3. **"Out of memory" with AI models**
   - Use CPU: `python main.py --device cpu`
   - Reduce audio file length

4. **Slow processing**
   - Skip AI models: `python main.py --no-deepfilternet --no-demucs`
   - Skip plots: `python main.py --no-plots`

## References

### Papers
- Wiener Filter: N. Wiener, "Extrapolation, Interpolation, and Smoothing of Stationary Time Series" (1949)
- Spectral Subtraction: S. Boll, "Suppression of acoustic noise in speech using spectral subtraction" (1979)
- DeepFilterNet: H. Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement Framework" (2022)
- Demucs: A. Défossez et al., "Music Source Separation in the Waveform Domain" (2019)

### Useful Links
- [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [PESQ ITU-T P.862](https://www.itu.int/rec/T-REC-P.862)
- [STOI Paper](https://ieeexplore.ieee.org/document/5713237)

## License

This project is for academic/thesis purposes. Please cite appropriately if used in research.

## Author

Audio Denoising Thesis Project - Bachelor's Thesis on Generative AI for Audio Denoising
