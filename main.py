#!/usr/bin/env python3
"""
Audio Denoising Comparison Pipeline

This script orchestrates the full comparison between classical DSP filters
and AI-based denoising models for audio and sensor data.

Usage:
    python main.py                      # Process all files in input folders
    python main.py --audio-only         # Process only audio files
    python main.py --sensor-only        # Process only sensor files
    python main.py --generate-samples   # Generate sample data for testing
    python main.py --help               # Show help

Author: Audio Denoising Thesis Project
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_loader import AudioLoader, SensorLoader, AudioData, SensorData
from src.dsp_filters import DSPFilters
from src.ai_denoisers import AIDenoiser
from src.metrics import MetricsCalculator, MetricsResult
from src.visualization import Visualizer


class DenoisingPipeline:
    """Main pipeline for audio denoising comparison."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with configuration."""
        self.config = config or Config()
        
        # Initialize components
        self.audio_loader = AudioLoader(self.config)
        self.sensor_loader = SensorLoader(self.config)
        self.dsp_filters = DSPFilters(self.config)
        self.ai_denoiser = AIDenoiser(self.config)
        self.metrics = MetricsCalculator(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Track all results
        self.all_results: List[MetricsResult] = []
    
    def generate_sample_audio(self, duration: float = 3.0, 
                               frequency: float = 440.0) -> AudioData:
        """
        Generate a sample audio signal (sine wave) for testing.
        
        Args:
            duration: Duration in seconds
            frequency: Frequency of sine wave in Hz
            
        Returns:
            AudioData with clean sine wave
        """
        sr = self.config.sample_rate
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Create a more speech-like signal with multiple harmonics
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        signal += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
        signal += 0.15 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
        
        # Add envelope for natural sound
        envelope = np.ones_like(t)
        envelope[:int(0.1 * sr)] = np.linspace(0, 1, int(0.1 * sr))  # Attack
        envelope[-int(0.2 * sr):] = np.linspace(1, 0, int(0.2 * sr))  # Release
        signal *= envelope
        
        return AudioData(
            signal=signal.astype(np.float32),
            sample_rate=sr,
            filename="sample_audio.wav",
            duration=duration,
            channels=1
        )
    
    def generate_sample_sensor(self, duration: float = 10.0,
                                frequency: float = 1.0) -> SensorData:
        """
        Generate sample sensor data for testing.
        
        Args:
            duration: Duration in seconds
            frequency: Frequency of signal
            
        Returns:
            SensorData with clean signal
        """
        sample_rate = 100  # 100 Hz sensor
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Create signal with trend and periodic component
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        signal += 0.1 * t  # Linear trend
        signal += 0.2 * np.sin(2 * np.pi * 0.1 * t)  # Low frequency component
        
        return SensorData(
            time=t,
            values=signal.reshape(-1, 1),
            filename="sample_sensor.csv",
            sample_rate=sample_rate,
            duration=duration,
            columns=['value']
        )
    
    def process_audio_file(self, 
                           clean_audio: Optional[AudioData],
                           noisy_audio: AudioData,
                           dsp_methods: List[str] = None,
                           ai_methods: List[str] = None,
                           save_outputs: bool = True) -> Dict:
        """
        Process a single audio file with all methods.
        
        Args:
            clean_audio: Clean reference (for metrics, can be None)
            noisy_audio: Noisy input audio
            dsp_methods: List of DSP methods to apply
            ai_methods: List of AI methods to apply
            save_outputs: Whether to save filtered audio files
            
        Returns:
            Dictionary with results
        """
        if dsp_methods is None:
            dsp_methods = ['wiener', 'spectral_subtraction', 'lowpass', 'bandpass']
        if ai_methods is None:
            ai_methods = ['noisereduce', 'speechbrain', 'rnnoise', 'deepfilternet', 'deepfilternet2', 'deepfilternet2_hf', 'demucs']
        
        # Filter DSP methods if specific methods were selected
        if hasattr(self.config, 'selected_methods') and self.config.selected_methods:
            dsp_methods = [m for m in dsp_methods if m in self.config.selected_methods]
        
        # Skip DSP if --no-dsp flag
        if hasattr(self.config, 'skip_dsp') and self.config.skip_dsp:
            dsp_methods = []
        
        results = {
            'filtered_audio': {},
            'metrics': [],
            'dsp_results': {},
            'ai_results': {}
        }
        
        base_name = Path(noisy_audio.filename).stem
        
        # Create subfolder for this input file
        output_subdir = self.config.output_filtered_dir / base_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Apply DSP filters
        if dsp_methods:
            print(f"\n  Applying DSP filters...")
            for method in tqdm(dsp_methods, desc="  DSP"):
                try:
                    filtered_audio, filter_result = self.dsp_filters.filter_audio(
                        noisy_audio, method=method
                    )
                    results['filtered_audio'][method] = filtered_audio
                    results['dsp_results'][method] = filter_result
                    
                    # Calculate metrics if clean reference available
                    if clean_audio is not None:
                        metrics_result = self.metrics.calculate_all(
                            clean_audio, noisy_audio, filtered_audio, method
                        )
                        results['metrics'].append(metrics_result)
                        self.all_results.append(metrics_result)
                    
                    # Save filtered audio to subfolder
                    if save_outputs:
                        output_path = output_subdir / f"{method}.wav"
                        self.audio_loader.save(filtered_audio, output_path)
                        
                except Exception as e:
                    warnings.warn(f"DSP method '{method}' failed: {e}")
        
        # Apply AI denoisers
        ai_to_run = [m for m in ai_methods if (
            (m == 'noisereduce' and self.config.use_noisereduce) or
            (m == 'speechbrain' and self.config.use_speechbrain) or
            (m == 'rnnoise' and self.config.use_rnnoise) or
            (m == 'deepfilternet' and self.config.use_deepfilternet) or
            (m == 'deepfilternet2' and self.config.use_deepfilternet2) or
            (m == 'deepfilternet2_hf' and self.config.use_deepfilternet_hf) or
            (m == 'demucs' and self.config.use_demucs)
        )]
        
        if ai_to_run:
            print(f"\n  Applying AI denoisers: {ai_to_run}")
        
        for method in ai_to_run:
            try:
                print(f"    Processing with {method}...")
                filtered_audio, ai_result = self.ai_denoiser.denoise(noisy_audio, method)
                results['filtered_audio'][method] = filtered_audio
                results['ai_results'][method] = ai_result
                
                # Calculate metrics if clean reference available
                if clean_audio is not None:
                    metrics_result = self.metrics.calculate_all(
                        clean_audio, noisy_audio, filtered_audio, method
                    )
                    results['metrics'].append(metrics_result)
                    self.all_results.append(metrics_result)
                
                # Save filtered audio to subfolder
                if save_outputs:
                    output_path = output_subdir / f"{method}.wav"
                    self.audio_loader.save(filtered_audio, output_path)
                    
            except Exception as e:
                warnings.warn(f"AI method '{method}' failed: {e}")
        
        return results
    
    def process_sensor_file(self,
                             clean_data: Optional[SensorData],
                             noisy_data: SensorData,
                             dsp_methods: List[str] = None,
                             ai_methods: List[str] = None,
                             save_outputs: bool = True) -> Dict:
        """
        Process a single sensor file with DSP filters and AI-based denoisers.
        
        AI methods treat each sensor column as a 1-D signal, wrap it as
        AudioData, run the denoiser, and convert the result back to SensorData.
        
        Args:
            clean_data: Clean reference (for metrics, can be None)
            noisy_data: Noisy sensor data
            dsp_methods: List of DSP methods to apply
            ai_methods: List of AI methods to apply
            save_outputs: Whether to save filtered data files
            
        Returns:
            Dictionary with results
        """
        if dsp_methods is None:
            dsp_methods = ['lowpass', 'highpass', 'moving_average', 'median']
        if ai_methods is None:
            ai_methods = ['noisereduce', 'speechbrain', 'deepfilternet', 'deepfilternet2',
                          'deepfilternet2_hf', 'demucs']
        
        results = {
            'filtered_data': {},
            'metrics': [],
            'dsp_results': {},
            'ai_results': {}
        }
        
        base_name = Path(noisy_data.filename).stem
        
        # --- DSP filters ---
        print(f"\n  Applying DSP filters to sensor data...")
        for method in tqdm(dsp_methods, desc="  DSP"):
            try:
                filtered_data, filter_result = self.dsp_filters.filter_sensor(
                    noisy_data, method=method
                )
                results['filtered_data'][method] = filtered_data
                results['dsp_results'][method] = filter_result
                
                if clean_data is not None:
                    metrics_result = self.metrics.calculate_sensor_metrics(
                        clean_data, noisy_data, filtered_data, method
                    )
                    results['metrics'].append(metrics_result)
                    self.all_results.append(metrics_result)
                
                if save_outputs:
                    output_path = self.config.output_filtered_dir / f"{base_name}_{method}.csv"
                    self.sensor_loader.save(filtered_data, output_path)
                    
            except Exception as e:
                warnings.warn(f"DSP method '{method}' failed for sensor data: {e}")
        
        # --- AI denoisers ---
        ai_to_run = [m for m in ai_methods if (
            (m == 'noisereduce' and self.config.use_noisereduce) or
            (m == 'speechbrain' and self.config.use_speechbrain) or
            (m == 'rnnoise' and self.config.use_rnnoise) or
            (m == 'deepfilternet' and self.config.use_deepfilternet) or
            (m == 'deepfilternet2' and self.config.use_deepfilternet2) or
            (m == 'deepfilternet2_hf' and self.config.use_deepfilternet_hf) or
            (m == 'demucs' and self.config.use_demucs)
        )]
        
        if ai_to_run:
            print(f"\n  Applying AI denoisers to sensor data: {ai_to_run}")
        
        for method in ai_to_run:
            try:
                print(f"    Processing sensor with {method}...")
                filtered_data, ai_result = self.ai_denoiser.denoise_sensor(
                    noisy_data, method=method
                )
                results['filtered_data'][method] = filtered_data
                results['ai_results'][method] = ai_result
                
                if clean_data is not None:
                    metrics_result = self.metrics.calculate_sensor_metrics(
                        clean_data, noisy_data, filtered_data, method
                    )
                    results['metrics'].append(metrics_result)
                    self.all_results.append(metrics_result)
                
                if save_outputs:
                    output_path = self.config.output_filtered_dir / f"{base_name}_{method}.csv"
                    self.sensor_loader.save(filtered_data, output_path)
                    
            except Exception as e:
                warnings.warn(f"AI method '{method}' failed for sensor data: {e}")
        
        return results
    
    def run_audio_pipeline(self, generate_plots: bool = True):
        """
        Run the complete audio denoising pipeline.
        
        Processes all audio files in the input folder.
        """
        print("\n" + "="*80)
        print("AUDIO DENOISING PIPELINE")
        print("="*80)
        
        # Load audio files
        audio_files = self.audio_loader.load_all()
        
        if not audio_files:
            print("\nNo audio files found in input folder.")
            print(f"Please add .wav files to: {self.config.input_audio_dir}")
            print("\nGenerating sample data for demonstration...")
            
            # Generate sample for demonstration
            clean_audio = self.generate_sample_audio()
            noisy_audio, _ = self.audio_loader.add_noise(clean_audio, 'white', snr_db=10)
            
            # Save samples
            self.audio_loader.save(clean_audio, 
                                  self.config.input_audio_dir / "sample_clean.wav")
            self.audio_loader.save(noisy_audio, 
                                  self.config.input_audio_dir / "sample_noisy.wav")
            
            audio_files = [noisy_audio]
            clean_references = {noisy_audio.filename: clean_audio}
        else:
            # For real files, we need clean references for metrics
            clean_references = {}
            clean_files = [f for f in audio_files if 'clean' in f.filename.lower()]
            noisy_files = [f for f in audio_files if 'noisy' in f.filename.lower()]
            
            # Files to process: noisy files, OR all non-clean files if no "noisy" files exist
            if noisy_files:
                files_to_process = noisy_files
            else:
                # Process all files EXCEPT clean files
                files_to_process = [f for f in audio_files if 'clean' not in f.filename.lower()]
            
            # Set up clean references
            if len(clean_files) == 1:
                clean_ref = clean_files[0]
                for f in files_to_process:
                    clean_references[f.filename] = clean_ref
                print(f"Using '{clean_ref.filename}' as reference for all files")
            elif len(clean_files) > 1:
                # Try to match clean/noisy pairs by name
                for f in files_to_process:
                    for clean in clean_files:
                        f_base = f.filename.lower().replace('noisy', '').replace('_', '').replace('.wav', '')
                        clean_base = clean.filename.lower().replace('clean', '').replace('_', '').replace('.wav', '')
                        if f_base == clean_base or clean_base in f_base or f_base in clean_base:
                            clean_references[f.filename] = clean
                            break
            
            # Update audio_files to only process non-clean files
            audio_files = files_to_process
            
            print(f"\nFound {len(audio_files)} audio file(s) to process")
            print(f"Found {len(clean_references)} clean reference(s)")
        
        # Process each audio file
        all_filtered = {}
        for audio in audio_files:
            print(f"\n{'='*60}")
            print(f"Processing: {audio.filename}")
            print(f"{'='*60}")
            
            clean_ref = clean_references.get(audio.filename)
            
            results = self.process_audio_file(
                clean_audio=clean_ref,
                noisy_audio=audio,
                save_outputs=True
            )
            
            all_filtered[audio.filename] = {
                'noisy': audio,
                'clean': clean_ref,
                'filtered': results['filtered_audio']
            }
            
            # Generate plots for this file (in subfolder)
            if generate_plots and results['filtered_audio']:
                print(f"\n  Generating visualizations...")
                base_name = Path(audio.filename).stem
                # Create plots subfolder
                plots_subdir = self.config.output_plots_dir / base_name
                plots_subdir.mkdir(parents=True, exist_ok=True)
                self.visualizer.save_all_plots(
                    clean=clean_ref,
                    noisy=audio,
                    filtered_dict=results['filtered_audio'],
                    metrics_results=results['metrics'],
                    base_name=base_name,
                    output_dir=plots_subdir
                )
        
        return all_filtered
    
    def run_sensor_pipeline(self, generate_plots: bool = True):
        """
        Run the sensor data filtering pipeline.
        """
        print("\n" + "="*80)
        print("SENSOR DATA FILTERING PIPELINE")
        print("="*80)
        
        # Load sensor files
        sensor_files = self.sensor_loader.load_all()
        
        if not sensor_files:
            print("\nNo sensor files found in input folder.")
            print(f"Please add .csv files to: {self.config.input_sensor_dir}")
            print("\nGenerating sample data for demonstration...")
            
            # Generate sample for demonstration
            clean_data = self.generate_sample_sensor()
            noisy_data, _ = self.sensor_loader.add_noise(clean_data, 'white', snr_db=10)
            
            # Save samples
            self.sensor_loader.save(clean_data, 
                                   self.config.input_sensor_dir / "sample_clean.csv")
            self.sensor_loader.save(noisy_data, 
                                   self.config.input_sensor_dir / "sample_noisy.csv")
            
            sensor_files = [noisy_data]
            clean_references = {noisy_data.filename: clean_data}
        else:
            clean_references = {}
            # Similar pairing logic as audio
            clean_files = [f for f in sensor_files if 'clean' in f.filename.lower()]
            noisy_files = [f for f in sensor_files if 'noisy' in f.filename.lower()]
            
            for noisy in noisy_files:
                for clean in clean_files:
                    if noisy.filename.replace('noisy', '').replace('Noisy', '') == \
                       clean.filename.replace('clean', '').replace('Clean', ''):
                        clean_references[noisy.filename] = clean
                        break
            
            if noisy_files:
                sensor_files = noisy_files
            
            print(f"\nFound {len(sensor_files)} sensor file(s) to process")
        
        # Process each sensor file
        all_filtered = {}
        for sensor_data in sensor_files:
            print(f"\n{'='*60}")
            print(f"Processing: {sensor_data.filename}")
            print(f"{'='*60}")
            
            clean_ref = clean_references.get(sensor_data.filename)
            
            results = self.process_sensor_file(
                clean_data=clean_ref,
                noisy_data=sensor_data,
                save_outputs=True
            )
            
            all_filtered[sensor_data.filename] = {
                'noisy': sensor_data,
                'clean': clean_ref,
                'filtered': results['filtered_data']
            }
            
            # Generate plots
            if generate_plots and results['filtered_data']:
                print(f"\n  Generating visualizations...")
                base_name = Path(sensor_data.filename).stem
                plot_path = self.config.output_plots_dir / f"{base_name}_sensor.png"
                self.visualizer.plot_sensor_data(
                    clean=clean_ref,
                    noisy=sensor_data,
                    filtered_dict=results['filtered_data'],
                    save_path=plot_path
                )
        
        return all_filtered
    
    def run_full_pipeline(self, 
                          process_audio: bool = True,
                          process_sensor: bool = True,
                          generate_plots: bool = True):
        """
        Run the complete pipeline for both audio and sensor data.
        """
        print("\n" + "="*80)
        print("AUDIO DENOISING THESIS - COMPARISON PIPELINE")
        print("="*80)
        print(f"\nProject directory: {self.config.project_root}")
        print(f"Input audio: {self.config.input_audio_dir}")
        print(f"Input sensor: {self.config.input_sensor_dir}")
        print(f"Output directory: {self.config.output_filtered_dir.parent}")
        
        audio_results = None
        sensor_results = None
        
        if process_audio:
            audio_results = self.run_audio_pipeline(generate_plots)
        
        if process_sensor:
            sensor_results = self.run_sensor_pipeline(generate_plots)
        
        # Save all metrics
        if self.all_results:
            print("\n" + "="*80)
            print("SAVING RESULTS")
            print("="*80)
            
            self.metrics.save_results(self.all_results, format='csv')
            self.metrics.save_results(self.all_results, format='json')
            self.metrics.print_summary(self.all_results)
            
            # Generate overall comparison plot
            if generate_plots and len(self.all_results) > 1:
                comparison_path = self.config.output_plots_dir / "overall_comparison.png"
                self.visualizer.plot_metrics_comparison(self.all_results, 
                                                        save_path=comparison_path)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to:")
        print(f"  Filtered files: {self.config.output_filtered_dir}")
        print(f"  Plots: {self.config.output_plots_dir}")
        print(f"  Metrics: {self.config.output_metrics_dir}")
        
        return {
            'audio': audio_results,
            'sensor': sensor_results,
            'metrics': self.all_results
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Denoising Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Process all files with all methods
  python main.py --methods deepfilternet2_hf  # Test only DeepFilterNet2 HF
  python main.py --methods noisereduce demucs # Use specific methods
  python main.py --audio-only                 # Process only audio files
  python main.py --generate-samples           # Generate sample data
  python main.py --no-plots                   # Skip plot generation

Available methods:
  DSP: wiener, spectral_subtraction, lowpass, bandpass
  AI:  noisereduce, speechbrain, deepfilternet, deepfilternet2, demucs
        """
    )
    
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Specific methods to run (e.g., --methods deepfilternet2_hf noisereduce)')
    parser.add_argument('--audio-only', action='store_true',
                       help='Process only audio files')
    parser.add_argument('--sensor-only', action='store_true',
                       help='Process only sensor files')
    parser.add_argument('--generate-samples', action='store_true',
                       help='Generate sample data for testing')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for AI models (default: cpu)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sample rate for audio (default: 16000)')
    parser.add_argument('--no-dsp', action='store_true',
                       help='Skip DSP filters')
    
    args = parser.parse_args()
    
    # Create configuration first (uses defaults from config.py)
    config = Config(
        device=args.device,
        sample_rate=args.sample_rate,
    )
    
    # Override with --methods argument if provided
    if args.methods:
        config.use_noisereduce = 'noisereduce' in args.methods
        config.use_speechbrain = 'speechbrain' in args.methods
        config.use_rnnoise = 'rnnoise' in args.methods
        config.use_deepfilternet = 'deepfilternet' in args.methods
        config.use_deepfilternet2 = 'deepfilternet2' in args.methods
        config.use_deepfilternet_hf = 'deepfilternet2_hf' in args.methods
        config.use_demucs = 'demucs' in args.methods
    
    # Store methods filter in config for DSP
    config.selected_methods = args.methods
    config.skip_dsp = args.no_dsp
    
    # Create and run pipeline
    pipeline = DenoisingPipeline(config)
    
    if args.generate_samples:
        print("Generating sample data...")
        
        # Generate audio samples
        clean_audio = pipeline.generate_sample_audio()
        for noise_type in ['white', 'pink']:
            for snr in [5, 10, 15]:
                noisy, _ = pipeline.audio_loader.add_noise(clean_audio, noise_type, snr)
                pipeline.audio_loader.save(
                    noisy, 
                    config.input_audio_dir / f"noisy_{noise_type}_{snr}dB.wav"
                )
        pipeline.audio_loader.save(clean_audio, config.input_audio_dir / "clean.wav")
        
        # Generate sensor samples
        clean_sensor = pipeline.generate_sample_sensor()
        for noise_type in ['white', 'pink']:
            for snr in [5, 10, 15]:
                noisy, _ = pipeline.sensor_loader.add_noise(clean_sensor, noise_type, snr)
                pipeline.sensor_loader.save(
                    noisy,
                    config.input_sensor_dir / f"noisy_{noise_type}_{snr}dB.csv"
                )
        pipeline.sensor_loader.save(clean_sensor, config.input_sensor_dir / "clean.csv")
        
        print(f"Sample data saved to:")
        print(f"  Audio: {config.input_audio_dir}")
        print(f"  Sensor: {config.input_sensor_dir}")
        return
    
    # Determine what to process
    process_audio = not args.sensor_only
    process_sensor = not args.audio_only
    generate_plots = not args.no_plots
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        process_audio=process_audio,
        process_sensor=process_sensor,
        generate_plots=generate_plots
    )
    
    return results


if __name__ == "__main__":
    main()
