"""
AI-based audio denoisers using pretrained models.
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from .config import Config
from .data_loader import AudioData, SensorData, sensor_to_audio, audio_to_sensor


@dataclass
class AIDenoiseResult:
    """Container for AI denoising output."""
    filtered_signal: np.ndarray
    method: str
    model_name: str
    processing_time: float


class AIDenoiser:
    """AI-based audio denoiser using pretrained models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Model instances (lazy loaded)
        self._deepfilternet_model = None
        self._demucs_model = None
    
    @property
    def deepfilternet_model(self):
        """Lazy load DeepFilterNet model."""
        if self._deepfilternet_model is None and self.config.use_deepfilternet:
            self._deepfilternet_model = self._load_deepfilternet()
        return self._deepfilternet_model
    
    @property
    def demucs_model(self):
        """Lazy load Demucs model."""
        if self._demucs_model is None and self.config.use_demucs:
            self._demucs_model = self._load_demucs()
        return self._demucs_model
    
    def _load_deepfilternet(self, model_name: str = 'DeepFilterNet'):
        """Load DeepFilterNet model."""
        try:
            # Try to import the df module
            from df.enhance import enhance, init_df
            
            model, df_state, _ = init_df(model_name)
            print(f"{model_name} model loaded successfully (Python API)")
            return {'model': model, 'df_state': df_state, 'use_cli': False, 'model_name': model_name}
        except ImportError as e:
            # Check if CLI is available as fallback
            import shutil
            if shutil.which('deepFilter'):
                print(f"{model_name} CLI available (using command-line interface)")
                return {'use_cli': True, 'model_name': model_name}
            warnings.warn(f"{model_name} not available: {e}. "
                         "Install with: pip install deepfilternet")
            return None
        except Exception as e:
            # Check if CLI is available as fallback
            import shutil
            if shutil.which('deepFilter'):
                print(f"{model_name} CLI available (using command-line interface)")
                return {'use_cli': True, 'model_name': model_name}
            warnings.warn(f"Error loading {model_name}: {e}")
            return None
    
    def _load_rnnoise(self):
        """Load RNNoise model."""
        try:
            import rnnoise
            print("RNNoise loaded successfully")
            return {'available': True}
        except ImportError as e:
            warnings.warn(f"RNNoise not available: {e}. "
                         "Install with: pip install rnnoise-python")
            return None
        except Exception as e:
            warnings.warn(f"Error loading RNNoise: {e}")
            return None
    
    def _load_demucs(self):
        """Load Demucs model for source separation."""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            # Load the htdemucs model (or specified variant)
            model = get_model(self.config.demucs_model)
            model.to(self.device)
            model.eval()
            print(f"Demucs model '{self.config.demucs_model}' loaded successfully")
            return model
        except ImportError as e:
            warnings.warn(f"Demucs not available: {e}. "
                         "Install with: pip install demucs")
            return None
        except Exception as e:
            warnings.warn(f"Error loading Demucs: {e}")
            return None
    
    def denoise_deepfilternet(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply DeepFilterNet for speech enhancement.
        
        DeepFilterNet is specifically designed for real-time speech denoising
        and works best with 48kHz audio.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        import tempfile
        import subprocess
        import soundfile as sf
        
        start_time = time.time()
        
        if self.deepfilternet_model is None:
            raise RuntimeError("DeepFilterNet model not available. "
                             "Check installation: pip install deepfilternet")
        
        # Check if we should use CLI
        use_cli = self.deepfilternet_model.get('use_cli', False)
        
        if use_cli:
            # Use command-line interface
            return self._denoise_deepfilternet_cli(audio_data, start_time)
        
        try:
            from df.enhance import enhance
            
            model = self.deepfilternet_model['model']
            df_state = self.deepfilternet_model['df_state']
            
            # DeepFilterNet expects 48kHz audio
            target_sr = 48000
            signal = audio_data.signal
            original_sr = audio_data.sample_rate
            
            # Resample to 48kHz if needed
            if original_sr != target_sr:
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(original_sr, target_sr)
                signal_tensor = resampler(signal_tensor)
                signal = signal_tensor.squeeze(0).numpy()
            
            # Convert to tensor and enhance
            audio_tensor = torch.from_numpy(signal).float().unsqueeze(0)
            
            # Enhance audio
            enhanced = enhance(model, df_state, audio_tensor)
            
            # Convert back to numpy
            if isinstance(enhanced, torch.Tensor):
                enhanced_signal = enhanced.squeeze().numpy()
            else:
                enhanced_signal = enhanced
            
            # Resample back to original sample rate if needed
            if original_sr != target_sr:
                enhanced_tensor = torch.from_numpy(enhanced_signal).float().unsqueeze(0)
                resampler_back = torchaudio.transforms.Resample(target_sr, original_sr)
                enhanced_tensor = resampler_back(enhanced_tensor)
                enhanced_signal = enhanced_tensor.squeeze(0).numpy()
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=enhanced_signal,
                sample_rate=original_sr,
                filename=f"deepfilternet_{audio_data.filename}",
                duration=len(enhanced_signal) / original_sr,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=enhanced_signal,
                method='deepfilternet',
                model_name='DeepFilterNet2',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except Exception as e:
            # Try CLI as fallback
            try:
                return self._denoise_deepfilternet_cli(audio_data, start_time)
            except:
                raise RuntimeError(f"DeepFilterNet processing failed: {e}")
    
    def _denoise_deepfilternet_cli(self, audio_data: AudioData, start_time: float) -> Tuple[AudioData, AIDenoiseResult]:
        """Use DeepFilterNet CLI as fallback."""
        import tempfile
        import subprocess
        import soundfile as sf
        import time
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input audio
            input_path = os.path.join(tmpdir, "input.wav")
            output_dir = tmpdir
            
            sf.write(input_path, audio_data.signal, audio_data.sample_rate)
            
            # Run deepFilter CLI
            cmd = ["deepFilter", input_path, "-o", output_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"DeepFilterNet CLI failed: {result.stderr}")
            
            # Load output
            output_path = os.path.join(output_dir, "input_DeepFilterNet2.wav")
            if not os.path.exists(output_path):
                # Try alternative output name
                for f in os.listdir(output_dir):
                    if f.endswith('.wav') and f != 'input.wav':
                        output_path = os.path.join(output_dir, f)
                        break
            
            enhanced_signal, sr = sf.read(output_path)
            
            # Resample if needed
            if sr != audio_data.sample_rate:
                import librosa
                enhanced_signal = librosa.resample(enhanced_signal, orig_sr=sr, target_sr=audio_data.sample_rate)
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=enhanced_signal,
                sample_rate=audio_data.sample_rate,
                filename=f"deepfilternet_{audio_data.filename}",
                duration=len(enhanced_signal) / audio_data.sample_rate,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=enhanced_signal,
                method='deepfilternet',
                model_name='DeepFilterNet2 (CLI)',
                processing_time=processing_time
            )
            
            return denoised_audio, result
    
    def denoise_demucs(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply Demucs for audio source separation / denoising.
        
        Demucs separates audio into sources (drums, bass, vocals, other).
        For denoising speech, we extract the 'vocals' source.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        start_time = time.time()
        
        if self.demucs_model is None:
            raise RuntimeError("Demucs model not available. "
                             "Check installation: pip install demucs")
        
        try:
            from demucs.apply import apply_model
            
            model = self.demucs_model
            signal = audio_data.signal
            original_sr = audio_data.sample_rate
            
            # Demucs expects stereo audio at its sample rate (usually 44100)
            model_sr = model.samplerate
            
            # Resample if needed
            if original_sr != model_sr:
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(original_sr, model_sr)
                signal_tensor = resampler(signal_tensor)
                signal = signal_tensor.squeeze(0).numpy()
            
            # Convert to stereo if mono
            if signal.ndim == 1:
                signal = np.stack([signal, signal], axis=0)
            
            # Convert to tensor [batch, channels, samples]
            audio_tensor = torch.from_numpy(signal).float().unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(model, audio_tensor, device=self.device)
            
            # Sources shape: [batch, sources, channels, samples]
            # Get vocals (index depends on model, usually vocals is at index 3)
            source_names = model.sources
            if 'vocals' in source_names:
                vocals_idx = source_names.index('vocals')
            else:
                # Fallback to last source or first
                vocals_idx = 0
            
            vocals = sources[0, vocals_idx].cpu().numpy()
            
            # Convert back to mono
            if vocals.ndim > 1:
                vocals = np.mean(vocals, axis=0)
            
            # Resample back to original sample rate
            if original_sr != model_sr:
                vocals_tensor = torch.from_numpy(vocals).float().unsqueeze(0)
                resampler_back = torchaudio.transforms.Resample(model_sr, original_sr)
                vocals_tensor = resampler_back(vocals_tensor)
                vocals = vocals_tensor.squeeze(0).numpy()
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=vocals,
                sample_rate=original_sr,
                filename=f"demucs_{audio_data.filename}",
                duration=len(vocals) / original_sr,
                channels=1
            )
            
            result = AIDenoiseResult(
                filtered_signal=vocals,
                method='demucs',
                model_name=self.config.demucs_model,
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except Exception as e:
            raise RuntimeError(f"Demucs processing failed: {e}")
    
    def denoise_deepfilternet_hf(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply DeepFilterNet2 via Hugging Face Spaces API.
        
        Uses the public Hugging Face Space: hshr/DeepFilterNet2
        This avoids local Rust/torchaudio installation issues.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        import tempfile
        import soundfile as sf
        import os
        
        start_time = time.time()
        
        try:
            from gradio_client import Client, handle_file
            
            # Save audio to temp file
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.wav")
                sf.write(input_path, audio_data.signal, audio_data.sample_rate)
                
                # Connect to Hugging Face Space
                print("    Connecting to DeepFilterNet2 on Hugging Face...")
                client = Client("hshr/DeepFilterNet2")
                
                # Call the API - use default endpoint (no api_name)
                # The gradio_client will automatically use the first available endpoint
                result = client.predict(handle_file(input_path))
                
                # Load enhanced audio
                enhanced_signal, sr = sf.read(result)
                
                # Resample if needed
                if sr != audio_data.sample_rate:
                    import librosa
                    enhanced_signal = librosa.resample(
                        enhanced_signal, orig_sr=sr, target_sr=audio_data.sample_rate
                    )
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=enhanced_signal,
                sample_rate=audio_data.sample_rate,
                filename=f"deepfilternet2_{audio_data.filename}",
                duration=len(enhanced_signal) / audio_data.sample_rate,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=enhanced_signal,
                method='deepfilternet2_hf',
                model_name='DeepFilterNet2 (HuggingFace)',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except ImportError:
            raise RuntimeError("gradio_client not available. Install with: pip install gradio_client")
        except Exception as e:
            # Try to show available API endpoints for debugging
            try:
                from gradio_client import Client
                client = Client("hshr/DeepFilterNet2")
                print(f"    Available API endpoints: {client.view_api(print_info=False)}")
            except:
                pass
            raise RuntimeError(f"DeepFilterNet2 HF API failed: {e}")
    
    def denoise_noisereduce(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply noisereduce spectral gating for noise reduction.
        
        This is a simpler but effective method that uses spectral gating
        to reduce stationary noise. Works well for many noise types.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        start_time = time.time()
        
        try:
            import noisereduce as nr
            
            signal = audio_data.signal
            sr = audio_data.sample_rate
            
            # Apply noisereduce with default settings
            # prop_decrease controls how much noise is reduced (0-1)
            reduced_noise = nr.reduce_noise(
                y=signal, 
                sr=sr,
                prop_decrease=0.8,  # Reduce noise by 80%
                stationary=True,    # Assume stationary noise
                n_fft=2048,
                hop_length=512
            )
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=reduced_noise,
                sample_rate=sr,
                filename=f"noisereduce_{audio_data.filename}",
                duration=len(reduced_noise) / sr,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=reduced_noise,
                method='noisereduce',
                model_name='Spectral Gating',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except ImportError:
            raise RuntimeError("noisereduce not available. Install with: pip install noisereduce")
        except Exception as e:
            raise RuntimeError(f"Noisereduce processing failed: {e}")
    
    def denoise_speechbrain(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply SpeechBrain MetricGAN+ for speech enhancement.
        
        MetricGAN+ is a GAN-based speech enhancement model that directly
        optimizes perceptual metrics like PESQ. Excellent for speech denoising.
        
        Reference: https://github.com/speechbrain/speechbrain
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        start_time = time.time()
        
        try:
            import torchaudio
            import os
            import soundfile as sf
            from huggingface_hub import hf_hub_download
            
            signal = audio_data.signal
            sr = audio_data.sample_rate
            
            # SpeechBrain MetricGAN+ expects 16kHz
            target_sr = 16000
            if sr != target_sr:
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                signal_tensor = resampler(signal_tensor)
                signal = signal_tensor.squeeze(0).numpy()
            
            # Model savedir for caching
            model_savedir = self.config.project_root / "pretrained_models" / "speechbrain_metricgan"
            os.makedirs(model_savedir, exist_ok=True)
            
            print("    Loading SpeechBrain MetricGAN+ model...")
            
            # Download model files from HuggingFace
            repo_id = "speechbrain/metricgan-plus-voicebank"
            hyperparams_path = hf_hub_download(repo_id, "hyperparams.yaml", cache_dir=str(model_savedir))
            checkpoint_path = hf_hub_download(repo_id, "enhance_model.ckpt", cache_dir=str(model_savedir))
            
            # Load hyperparams using hyperpyyaml (SpeechBrain's YAML format)
            from hyperpyyaml import load_hyperpyyaml
            from speechbrain.processing.features import STFT, ISTFT, spectral_magnitude
            from speechbrain.processing.signal_processing import resynthesize
            from speechbrain.lobes.models.MetricGAN import EnhancementGenerator
            
            # Load hyperparams (handles !new: and !ref: tags)
            with open(hyperparams_path) as f:
                hparams = load_hyperpyyaml(f)
            
            # Load checkpoint weights
            enhance_model = hparams['enhance_model']
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            enhance_model.load_state_dict(checkpoint)
            enhance_model.eval()
            
            # Process audio
            # Convert to tensor: (batch, time)
            signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
            
            # Compute STFT
            compute_stft = hparams['compute_stft']
            stft_out = compute_stft(signal_tensor)
            
            # Get magnitude (shape: batch, frames, freq_bins)
            mag = spectral_magnitude(stft_out, power=0.5)
            
            # Run enhancement model to get mask
            # The model expects relative lengths (1.0 = full length, for batched sequences)
            batch_size = mag.shape[0]
            lengths = torch.ones(batch_size)
            
            with torch.no_grad():
                mask = enhance_model(mag, lengths)
            
            # Apply mask to STFT magnitude
            enhanced_mag = mask * mag
            
            # Reconstruct using original phase
            # Get phase from original STFT
            phase = torch.atan2(stft_out[:, :, :, 1], stft_out[:, :, :, 0])
            
            # Convert enhanced magnitude back to complex STFT
            enhanced_real = enhanced_mag * torch.cos(phase)
            enhanced_imag = enhanced_mag * torch.sin(phase)
            enhanced_stft = torch.stack([enhanced_real, enhanced_imag], dim=-1)
            
            # Inverse STFT
            compute_istft = hparams['compute_istft']
            enhanced_signal = compute_istft(enhanced_stft)
            enhanced_signal = enhanced_signal.squeeze().numpy()
            
            # Truncate or pad to match original length
            original_len = len(signal)
            if len(enhanced_signal) > original_len:
                enhanced_signal = enhanced_signal[:original_len]
            elif len(enhanced_signal) < original_len:
                enhanced_signal = np.pad(enhanced_signal, (0, original_len - len(enhanced_signal)))
            
            # Resample back if needed
            if sr != target_sr:
                enhanced_tensor = torch.from_numpy(enhanced_signal).float().unsqueeze(0)
                resampler_back = torchaudio.transforms.Resample(target_sr, sr)
                enhanced_tensor = resampler_back(enhanced_tensor)
                enhanced_signal = enhanced_tensor.squeeze(0).numpy()
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=enhanced_signal,
                sample_rate=sr,
                filename=f"speechbrain_{audio_data.filename}",
                duration=len(enhanced_signal) / sr,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=enhanced_signal,
                method='speechbrain',
                model_name='SpeechBrain MetricGAN+',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except ImportError as e:
            raise RuntimeError(f"speechbrain not available: {e}. Install with: pip install speechbrain")
        except Exception as e:
            raise RuntimeError(f"SpeechBrain processing failed: {e}")
    
    def denoise_deepfilternet2(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply DeepFilterNet2 for speech enhancement.
        
        DeepFilterNet2 is an improved version with better performance
        on embedded devices while maintaining high quality.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        import tempfile
        import subprocess
        import soundfile as sf
        import os
        
        start_time = time.time()
        
        try:
            from df.enhance import enhance, init_df
            
            # Load DeepFilterNet2 model specifically
            model, df_state, _ = init_df('DeepFilterNet2')
            
            signal = audio_data.signal
            original_sr = audio_data.sample_rate
            
            # DeepFilterNet expects 48kHz audio
            target_sr = 48000
            if original_sr != target_sr:
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(original_sr, target_sr)
                signal_tensor = resampler(signal_tensor)
                signal = signal_tensor.squeeze(0).numpy()
            
            # Convert to tensor and enhance
            audio_tensor = torch.from_numpy(signal).float().unsqueeze(0)
            enhanced = enhance(model, df_state, audio_tensor)
            
            if isinstance(enhanced, torch.Tensor):
                enhanced_signal = enhanced.squeeze().numpy()
            else:
                enhanced_signal = enhanced
            
            # Resample back if needed
            if original_sr != target_sr:
                enhanced_tensor = torch.from_numpy(enhanced_signal).float().unsqueeze(0)
                resampler_back = torchaudio.transforms.Resample(target_sr, original_sr)
                enhanced_tensor = resampler_back(enhanced_tensor)
                enhanced_signal = enhanced_tensor.squeeze(0).numpy()
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=enhanced_signal,
                sample_rate=original_sr,
                filename=f"deepfilternet2_{audio_data.filename}",
                duration=len(enhanced_signal) / original_sr,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=enhanced_signal,
                method='deepfilternet2',
                model_name='DeepFilterNet2',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except Exception as e:
            raise RuntimeError(f"DeepFilterNet2 processing failed: {e}")
    
    def denoise_rnnoise(self, audio_data: AudioData) -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply RNNoise for noise suppression.
        
        RNNoise is a recurrent neural network based noise suppression
        library by Xiph.org. It's lightweight and effective for real-time use.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        import time
        start_time = time.time()
        
        try:
            import rnnoise
            
            signal = audio_data.signal
            sr = audio_data.sample_rate
            
            # RNNoise expects 48kHz mono audio
            target_sr = 48000
            if sr != target_sr:
                import librosa
                signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
            
            # Ensure float32 and proper range
            signal = signal.astype(np.float32)
            
            # Apply RNNoise
            denoiser = rnnoise.RNNoise()
            
            # Process in frames (RNNoise uses 10ms frames at 48kHz = 480 samples)
            frame_size = 480
            output = np.zeros_like(signal)
            
            for i in range(0, len(signal) - frame_size, frame_size):
                frame = signal[i:i + frame_size]
                filtered_frame = denoiser.process_frame(frame)
                output[i:i + frame_size] = filtered_frame
            
            # Handle remaining samples
            remaining = len(signal) % frame_size
            if remaining > 0:
                # Pad last frame
                last_frame = np.zeros(frame_size, dtype=np.float32)
                last_frame[:remaining] = signal[-remaining:]
                filtered_last = denoiser.process_frame(last_frame)
                output[-remaining:] = filtered_last[:remaining]
            
            # Resample back if needed
            if sr != target_sr:
                import librosa
                output = librosa.resample(output, orig_sr=target_sr, target_sr=sr)
            
            processing_time = time.time() - start_time
            
            denoised_audio = AudioData(
                signal=output,
                sample_rate=sr,
                filename=f"rnnoise_{audio_data.filename}",
                duration=len(output) / sr,
                channels=audio_data.channels
            )
            
            result = AIDenoiseResult(
                filtered_signal=output,
                method='rnnoise',
                model_name='RNNoise',
                processing_time=processing_time
            )
            
            return denoised_audio, result
            
        except ImportError:
            raise RuntimeError("rnnoise not available. Install with: pip install rnnoise-python")
        except Exception as e:
            raise RuntimeError(f"RNNoise processing failed: {e}")
    
    def denoise(self, audio_data: AudioData, method: str = 'noisereduce') -> Tuple[AudioData, AIDenoiseResult]:
        """
        Apply specified AI denoiser to audio.
        
        Args:
            audio_data: Input audio data
            method: AI method ('deepfilternet', 'demucs', 'noisereduce')
            
        Returns:
            Tuple of (denoised AudioData, AIDenoiseResult)
        """
        methods = {
            'deepfilternet': self.denoise_deepfilternet,
            'deepfilternet2': self.denoise_deepfilternet2,
            'deepfilternet2_hf': self.denoise_deepfilternet_hf,
            'speechbrain': self.denoise_speechbrain,
            'rnnoise': self.denoise_rnnoise,
            'demucs': self.denoise_demucs,
            'noisereduce': self.denoise_noisereduce,
        }
        
        if method not in methods:
            raise ValueError(f"Unknown AI method: {method}. "
                           f"Available: {list(methods.keys())}")
        
        return methods[method](audio_data)
    
    def denoise_all_methods(self, audio_data: AudioData) -> dict:
        """
        Apply all available AI denoisers to audio.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Dictionary mapping method name to (AudioData, AIDenoiseResult)
        """
        results = {}
        
        if self.config.use_noisereduce:
            try:
                results['noisereduce'] = self.denoise_noisereduce(audio_data)
            except Exception as e:
                warnings.warn(f"Noisereduce failed: {e}")
        
        if self.config.use_rnnoise:
            try:
                results['rnnoise'] = self.denoise_rnnoise(audio_data)
            except Exception as e:
                warnings.warn(f"RNNoise failed: {e}")
        
        if self.config.use_deepfilternet:
            try:
                results['deepfilternet'] = self.denoise_deepfilternet(audio_data)
            except Exception as e:
                warnings.warn(f"DeepFilterNet failed: {e}")
        
        if self.config.use_deepfilternet2:
            try:
                results['deepfilternet2'] = self.denoise_deepfilternet2(audio_data)
            except Exception as e:
                warnings.warn(f"DeepFilterNet2 failed: {e}")
        
        if self.config.use_speechbrain:
            try:
                results['speechbrain'] = self.denoise_speechbrain(audio_data)
            except Exception as e:
                warnings.warn(f"SpeechBrain failed: {e}")
        
        if self.config.use_deepfilternet_hf:
            try:
                results['deepfilternet2_hf'] = self.denoise_deepfilternet_hf(audio_data)
            except Exception as e:
                warnings.warn(f"DeepFilterNet2 HF failed: {e}")
        
        if self.config.use_demucs:
            try:
                results['demucs'] = self.denoise_demucs(audio_data)
            except Exception as e:
                warnings.warn(f"Demucs failed: {e}")
        
        return results
    
    def denoise_sensor(self, sensor_data: SensorData, method: str = 'noisereduce',
                       column_idx: int = 0) -> Tuple[SensorData, AIDenoiseResult]:
        """
        Apply an AI denoiser to a single column of sensor data.

        The sensor signal is wrapped as AudioData, processed by the chosen
        AI method, and the result is mapped back to SensorData.

        Args:
            sensor_data: Input sensor data
            method: AI denoising method name
            column_idx: Which value column to denoise

        Returns:
            Tuple of (denoised SensorData, AIDenoiseResult)
        """
        audio_wrapper = sensor_to_audio(sensor_data, column_idx)
        denoised_audio, ai_result = self.denoise(audio_wrapper, method)
        denoised_sensor = audio_to_sensor(denoised_audio, sensor_data, column_idx)
        return denoised_sensor, ai_result

    def denoise_all_methods_sensor(self, sensor_data: SensorData,
                                   column_idx: int = 0) -> dict:
        """
        Apply all enabled AI denoisers to sensor data.

        Args:
            sensor_data: Input sensor data
            column_idx: Which value column to denoise

        Returns:
            Dictionary mapping method name to (SensorData, AIDenoiseResult)
        """
        results = {}
        method_flags = [
            ('noisereduce', self.config.use_noisereduce),
            ('speechbrain', self.config.use_speechbrain),
            ('deepfilternet', self.config.use_deepfilternet),
            ('deepfilternet2', self.config.use_deepfilternet2),
            ('deepfilternet2_hf', self.config.use_deepfilternet_hf),
            ('demucs', self.config.use_demucs),
            ('rnnoise', self.config.use_rnnoise),
        ]
        for method, enabled in method_flags:
            if not enabled:
                continue
            try:
                results[method] = self.denoise_sensor(sensor_data, method, column_idx)
            except Exception as e:
                warnings.warn(f"AI method '{method}' failed on sensor data: {e}")
        return results

    def is_available(self, method: str) -> bool:
        """Check if a specific AI method is available."""
        if method == 'deepfilternet':
            return self.deepfilternet_model is not None
        elif method == 'demucs':
            return self.demucs_model is not None
        elif method == 'noisereduce':
            try:
                import noisereduce
                return True
            except ImportError:
                return False
        return False
    
    def available_methods(self) -> list:
        """Return list of available AI methods."""
        methods = []
        if self.config.use_deepfilternet:
            try:
                if self.deepfilternet_model is not None:
                    methods.append('deepfilternet')
            except:
                pass
        if self.config.use_demucs:
            try:
                if self.demucs_model is not None:
                    methods.append('demucs')
            except:
                pass
        return methods
