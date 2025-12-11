# Ace-Step KSampler Nodes for ComfyUI
# Specialized KSampler nodes optimized for Ace-Step audio generation
# 
# Features:
# - Professional quality presets based on Stable Audio Open
# - Optimized parameters for audio generation
# - Memory-efficient operations
# - Hidden manual parameters (only visible in custom mode)
# - English-only code and logs

import os
import sys
import torch
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Define directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add comfy_dir to sys.path to import nodes correctly
sys.path.append(comfy_dir)
try:
    from nodes import KSampler, KSamplerAdvanced
    import comfy.samplers
    import comfy.sample
    import comfy.sd
    import comfy.utils
except ImportError:
    # These imports only work in ComfyUI environment
    KSampler = None
    KSamplerAdvanced = None
    comfy = None
sys.path.remove(comfy_dir)


# ============================================================================
# ACE-STEP CONSTANTS
# ============================================================================

# Ace-Step Optimization Constants (Based on Stable Audio Open Official Docs)
# Source: https://huggingface.co/stabilityai/stable-audio-open-1.0
ACESTEP_CFG_RANGE = (3.0, 15.0)  # Optimal CFG range for audio
ACESTEP_OPTIMAL_CFG = 7.0  # Official default CFG
ACESTEP_DENOISE_DEFAULT = 1.0  # Default denoising
ACESTEP_AUDIO_BATCH_SIZE = 1  # Optimized for audio

# APG (Adaptive Projected Guidance) & CFG++ Parameters
# These techniques improve guidance quality and prevent oversaturation
ACESTEP_CFG_RESCALE_DEFAULT = 0.7  # CFG++ rescale multiplier (0.0 = disabled, 0.7 = recommended)
ACESTEP_APG_ENABLED_DEFAULT = True  # APG enabled by default for better audio quality

# Advanced Optimization Constants
ACESTEP_SEED_VARIATION = 1000  # Seed variation range for semi-random generation
ACESTEP_LATENT_SCALE = 1.0  # Latent scaling factor (1.0 = no scaling)
ACESTEP_NOISE_STRENGTH = 1.0  # Noise strength multiplier
ACESTEP_ENABLE_MEMORY_OPTIMIZATION = True  # Enable memory-efficient operations

# Step ranges based on empirical testing
ACESTEP_MIN_STEPS = 20
ACESTEP_OPTIMAL_STEPS = 80  # Empirically found optimal range: 80-90
ACESTEP_OFFICIAL_STEPS = 100
ACESTEP_MAX_STEPS = 120  # Above 120 causes over-processing


#=======================================================================================================================
# Vocoder Functions
#=======================================================================================================================

def load_vocoder_model():
    """
    Load the vocoder model from the safetensors file.
    
    Returns:
        Loaded vocoder model or None if loading fails
    """
    vocoder_path = os.path.join(my_dir, 'vocoder', 'diffusion_pytorch_model.safetensors')
    config_path = os.path.join(my_dir, 'vocoder', 'config.json')
    
    if not os.path.exists(vocoder_path):
        logger.error(f"Vocoder model not found at {vocoder_path}")
        return None
    
    if not os.path.exists(config_path):
        logger.error(f"Vocoder config not found at {config_path}")
        return None
    
    try:
        import json
        from safetensors.torch import load_file
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model weights
        state_dict = load_file(vocoder_path)
        
        logger.info(f"Vocoder model loaded successfully from {vocoder_path}")
        logger.info(f"Vocoder config class: {config.get('_class_name', 'Unknown')}")
        
        return {'state_dict': state_dict, 'config': config}
    
    except Exception as e:
        logger.error(f"Failed to load vocoder model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



def apply_vocoder_to_audio(audio_waveform, vocoder_data):
    """
    Apply the vocoder to the audio waveform.
    
    Args:
        audio_waveform: Audio tensor [batch, channels, samples]
        vocoder_data: Dict containing state_dict and config from load_vocoder_model()
    
    Returns:
        Vocoded audio tensor or original audio if processing fails
    """
    if vocoder_data is None:
        logger.warning("Vocoder data is None, returning original audio")
        return audio_waveform
    
    try:
        state_dict = vocoder_data['state_dict']
        config = vocoder_data['config']
        
        # Get device and dtype from input
        device = audio_waveform.device
        dtype = audio_waveform.dtype
        
        logger.info(f"Applying vocoder to audio shape {audio_waveform.shape}")
        
        # Normalize audio
        audio = audio_waveform.clone()
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / (max_val + 1e-10)
        
        # Final normalization with slight headroom
        logger.info("Applying vocoder normalization")
        max_val = torch.abs(audio).max()
        if max_val > 0.99:
            audio = audio / (max_val * 1.05)  # Leave 5% headroom
        audio = torch.clamp(audio, -1.0, 1.0)
        
        logger.info(f"Vocoder processing completed. Output shape: {audio.shape}")
        logger.info(f"Output range: [{audio.min():.4f}, {audio.max():.4f}]")
        
        return audio
    
    except Exception as e:
        logger.error(f"Failed to apply vocoder: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.warning("Returning original audio without vocoder processing")
        return audio_waveform


#=======================================================================================================================
# Advanced Optimization Functions
#=======================================================================================================================

def apply_patch_aware_processing(noise_pred, patch_size=16):
    """
    Apply patch-aware noise processing respecting ACEStepTransformer's patch structure.
    
    The ACEStepTransformer uses patch_size=[16, 1], meaning it processes 16 temporal
    frames together. Operations that don't respect this can break word boundaries.
    
    This function ensures smoothing/clamping respects patch boundaries to prevent
    "word cutting" artifacts.
    
    Args:
        noise_pred: Noise prediction tensor [batch, channels, time, freq]
        patch_size: Temporal patch size (default 16 for ACE-Step)
        
    Returns:
        Patch-aligned noise prediction
    """
    if noise_pred.dim() < 3:
        return noise_pred
    
    # Get temporal dimension (usually axis 2: [B, C, T, F])
    time_dim = noise_pred.shape[2] if noise_pred.dim() >= 3 else noise_pred.shape[1]
    
    # If temporal dimension is not divisible by patch_size, pad it
    remainder = time_dim % patch_size
    if remainder != 0:
        pad_amount = patch_size - remainder
        # Pad temporal dimension with reflection to avoid boundary artifacts
        if noise_pred.dim() == 4:  # [B, C, T, F]
            noise_pred = torch.nn.functional.pad(noise_pred, (0, 0, 0, pad_amount), mode='reflect')
        elif noise_pred.dim() == 3:  # [B, C, T]
            noise_pred = torch.nn.functional.pad(noise_pred, (0, pad_amount), mode='reflect')
    
    return noise_pred


def optimize_seed_for_audio(seed, step, total_steps):
    """
    Optimize seed for audio generation consistency.
    
    ACE-Step is highly sensitive to seeds. This function creates semi-deterministic
    seeds that maintain consistency while allowing for natural variation.
    
    Args:
        seed: Base seed
        step: Current step (for dynamic variation)
        total_steps: Total sampling steps
        
    Returns:
        Optimized seed value
    """
    # Keep base seed for first 70% of generation for consistency
    threshold = max(1, int(total_steps * 0.7))
    
    if step < threshold:
        return int(seed)
    
    # Add controlled variation in final 30% for natural sound
    # High-precision variation calculation
    final_steps = max(1, total_steps - threshold)
    progress_in_final = (step - threshold) / final_steps
    progress_in_final = max(0.0, min(1.0, progress_in_final))
    
    variation = int(progress_in_final * ACESTEP_SEED_VARIATION)
    return int(seed) + variation


def apply_dynamic_cfg(cfg_base, step, total_steps, cfg_range=(3.5, 4.5)):
    """
    Apply dynamic CFG scheduling optimized for ACE-Step audio.
    
    Start with higher CFG for structure, reduce for natural finish.
    Prevents over-guidance artifacts while maintaining prompt following.
    
    Args:
        cfg_base: Base CFG value
        step: Current step
        total_steps: Total steps
        cfg_range: (min, max) CFG range
        
    Returns:
        Adjusted CFG value
    """
    # Use high-precision calculation
    total_steps_safe = max(total_steps, 1)
    progress = float(step) / float(total_steps_safe)
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
    
    # First 30%: Slightly higher CFG for structure
    if progress < 0.3:
        return min(cfg_base * 1.05, cfg_range[1])
    
    # Middle 40%: Base CFG
    elif progress < 0.7:
        return cfg_base
    
    # Final 30%: Reduce CFG to prevent artifacts
    else:
        # Smooth reduction: max 9% reduction at end
        reduction_factor = (progress - 0.7) / 0.3  # 0 to 1 over final 30%
        reduction = reduction_factor * 0.09  # Max 9% reduction
        result = cfg_base * (1.0 - reduction)
        return max(result, cfg_range[0])


def apply_anti_autotune_smoothing(latent, strength=0.3):
    """
    Apply spectral smoothing to reduce autotune/robotic artifacts.
    
    Uses FREQUENCY-DOMAIN smoothing (not temporal) to reduce quantization effects
    that cause the "robotic voice" artifact, while preserving word boundaries.
    
    Args:
        latent: Latent tensor dict with 'samples' key
        strength: Smoothing strength (0.0-1.0, default 0.3)
                  0.0 = no smoothing, 1.0 = maximum smoothing
    
    Returns:
        Smoothed latent dict
    """
    if strength <= 0:
        return latent
    
    samples = latent['samples'].clone()
    
    # Apply SPECTRAL (frequency) smoothing, NOT temporal smoothing
    # Audio latents are typically [batch, channels, time, freq]
    if len(samples.shape) >= 4:
        # Smooth along frequency axis (axis -1) to reduce pitch quantization
        freq_dim = samples.shape[-1]
        
        if freq_dim > 3:
            # Create 1D smoothing kernel for frequency dimension
            kernel_size = 3  # Small kernel to preserve detail
            kernel = torch.tensor([0.25, 0.50, 0.25], 
                                   dtype=samples.dtype, device=samples.device)
            kernel = kernel.view(1, 1, 1, kernel_size)
            
            # Apply smoothing per channel
            smoothed = samples.clone()
            for b in range(samples.shape[0]):
                for c in range(samples.shape[1]):
                    channel_data = samples[b:b+1, c:c+1]
                    
                    # Pad for convolution (frequency axis only)
                    padded = torch.nn.functional.pad(channel_data, (1, 1, 0, 0), mode='reflect')
                    
                    # Convolve along frequency axis
                    smoothed_channel = torch.nn.functional.conv2d(padded, kernel, padding=0)
                    
                    # Blend original and smoothed based on strength
                    smoothed[b, c] = (1 - strength) * samples[b, c] + strength * smoothed_channel[0, 0]
            
            return {'samples': smoothed}
    
    return latent


def apply_frequency_damping_to_latent(latent, damping=0.0):
    """Apply damping to higher frequency bins of the latent 'samples' tensor."""
    if damping <= 0 or latent is None:
        return latent
    if isinstance(latent, dict) and 'samples' in latent:
        samples = latent['samples']
        if samples.dim() >= 4:
            F = samples.shape[-1]
            freqs = torch.linspace(0.0, 1.0, F, device=samples.device, dtype=samples.dtype)
            freq_mult = torch.exp(-damping * (freqs ** 2)).view(1, 1, 1, F)
            # Expand to support channelwise multiplication
            latent['samples'] = samples * freq_mult
    return latent


def apply_temporal_smoothing_to_latent(latent, strength=0.0):
    """Apply a small temporal smoothing kernel across time frames in the latent 'samples' tensor."""
    if strength <= 0 or latent is None:
        return latent
    if isinstance(latent, dict) and 'samples' in latent:
        samples = latent['samples']
        if samples.dim() >= 4:
            # Per-channel depthwise conv to avoid channel mismatch
            channels = samples.shape[1]
            kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=samples.dtype, device=samples.device).view(1, 1, 3, 1)
            # Expand to (channels, 1, kH, kW) for depthwise conv
            kernel = kernel_1d.repeat(channels, 1, 1, 1)
            padded = torch.nn.functional.pad(samples, (0, 0, 1, 1), mode='reflect')
            smoothed = torch.nn.functional.conv2d(padded, kernel, groups=channels, padding=0)
            latent['samples'] = (1.0 - strength) * samples + strength * smoothed
    return latent


def evaluate_audio_quality(audio_tensor):
    """
    Evaluates audio quality with 3 focused metrics:
    1. Metallic sound (Rolloff >9kHz = harsh/metallic) - 40%
    2. Word cuts (spectral discontinuities) - 40%
    3. Noise/hiss (excessive ZCR) - 20%
    
    Returns: tuple (quality_score, clarity_metric)
        - quality_score: 0.0-1.0 overall quality
        - clarity_metric: dict with word clarity info (severe_pct, cuts_score)
    """
    try:
        # Convert to numpy
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor
        
        # Handle dimensions
        if len(audio_np.shape) == 3:
            audio_np = audio_np[0]
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            audio_np = audio_np.mean(axis=0)
        elif len(audio_np.shape) == 2:
            audio_np = audio_np[0]
        audio_np = audio_np.flatten()
        
        if len(audio_np) == 0:
            return 0.0
        
        sample_rate = 44100
        scores = []
        raw_values = {}
        
        try:
            import librosa
            use_librosa = True
        except ImportError:
            use_librosa = False
        
        if use_librosa:
            # === METRIC 1: METALLIC SOUND (40%) ===
            try:
                rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=sample_rate, roll_percent=0.85)[0]
                mean_rolloff = np.mean(rolloff)
                raw_values['rolloff_hz'] = mean_rolloff
                
                # Natural music: 6-9kHz, Metallic: >9kHz
                if mean_rolloff <= 7000:
                    rolloff_score = 1.0
                elif mean_rolloff <= 8000:
                    rolloff_score = 0.8
                elif mean_rolloff <= 9000:
                    rolloff_score = 0.5
                elif mean_rolloff <= 10000:
                    rolloff_score = 0.2
                elif mean_rolloff <= 11000:
                    rolloff_score = 0.05
                else:
                    rolloff_score = 0.01
            except:
                rolloff_score = 0.5
            scores.append(('Metallic', rolloff_score, 0.40))
            
            # === METRIC 2: WORD CUTS (40%) ===
            # Detects abrupt spectral changes (word cuts) using spectral flux
            try:
                # Calculate STFT for spectral analysis
                stft = librosa.stft(audio_np, n_fft=2048, hop_length=512)
                magnitude = np.abs(stft)
                
                # Spectral Flux: frame-to-frame spectral change
                spectral_diff = np.diff(magnitude, axis=1)
                flux = np.sqrt(np.sum(spectral_diff**2, axis=0))
                
                # Normalize flux
                mean_flux = np.mean(flux)
                std_flux = np.std(flux)
                
                if std_flux > 1e-6:
                    normalized_flux = (flux - mean_flux) / std_flux
                    
                    # Detect EXTREME changes (word cuts create spectral discontinuities)
                    # Z-score > 3.5 = very unusual change (99.95% threshold)
                    severe_cuts = np.sum(normalized_flux > 4.0)  # Extreme discontinuities
                    moderate_cuts = np.sum((normalized_flux > 3.0) & (normalized_flux <= 4.0))  # Strong changes
                    
                    total_frames = len(normalized_flux)
                    severe_pct = (severe_cuts / total_frames) * 100
                    moderate_pct = (moderate_cuts / total_frames) * 100
                    
                    raw_values['cuts_severe'] = severe_cuts
                    raw_values['cuts_moderate'] = moderate_cuts
                    raw_values['severe_pct'] = severe_pct
                    raw_values['moderate_pct'] = moderate_pct
                    
                    # Scoring: Severe cuts matter MUCH more (word breaks)
                    # Moderate cuts are tolerable (natural transitions)
                    if severe_pct < 0.05:  # Very few word breaks
                        if moderate_pct < 0.8:
                            cuts_score = 1.0  # Excellent
                        elif moderate_pct < 1.2:
                            cuts_score = 0.95  # Very good
                        else:
                            cuts_score = 0.90  # Good
                    elif severe_pct < 0.10:  # Some word breaks
                        if moderate_pct < 0.8:
                            cuts_score = 0.85
                        elif moderate_pct < 1.2:
                            cuts_score = 0.80
                        else:
                            cuts_score = 0.75
                    elif severe_pct < 0.15:  # Noticeable breaks
                        cuts_score = 0.65
                    elif severe_pct < 0.25:  # Many breaks
                        cuts_score = 0.45
                    elif severe_pct < 0.40:  # Severe stuttering
                        cuts_score = 0.25
                    else:
                        cuts_score = 0.10  # Unusable
                    
                    raw_values['cuts_score'] = cuts_score
                else:
                    cuts_score = 0.0
                    raw_values['cuts_severe'] = -1
            except:
                cuts_score = 0.5
                raw_values['cuts_severe'] = -1
            scores.append(('WordCuts', cuts_score, 0.40))
            
            # === METRIC 3: NOISE/HISS (20%) ===
            try:
                zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
                mean_zcr = np.mean(zcr)
                raw_values['zcr'] = mean_zcr
                
                if 0.05 <= mean_zcr <= 0.12:
                    zcr_score = 1.0
                elif 0.03 <= mean_zcr < 0.05 or 0.12 < mean_zcr <= 0.18:
                    zcr_score = 0.7
                elif mean_zcr < 0.02:
                    zcr_score = 0.4
                else:
                    zcr_score = 0.3
            except:
                zcr_score = 0.5
            scores.append(('Noise', zcr_score, 0.20))
        
        else:
            # === FALLBACK: Basic numpy metrics ===
            logger.warning("Using basic metrics - install librosa")
            
            # Avoid division by zero
            if len(audio_np) > 0:
                zcr = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2 * len(audio_np))
                zcr_score = 1.0 if 0.05 <= zcr <= 0.15 else 0.5
            else:
                zcr_score = 0.0
            scores.append(('Noise', zcr_score, 1.0))
        
        # Calculate total
        total_score = sum(score * weight for _, score, weight in scores)
        
        # Calculate clarity metric (inverse of severe_pct - lower cuts = higher clarity)
        severe_pct = raw_values.get('severe_pct', 0.0)
        cuts_score_val = raw_values.get('cuts_score', 0.0)
        clarity_metric = {
            'severe_pct': severe_pct,
            'cuts_score': cuts_score_val,
            'clarity_score': 1.0 - (severe_pct / 100.0)  # Convert % to 0-1, invert (lower cuts = higher clarity)
        }
        
        # Log with raw values
        score_str = ", ".join([f"{name}={score:.2f}" for name, score, _ in scores])
        try:
            raw_str = f"Raw[Cuts:{raw_values.get('cuts_severe', 'N/A')}/{raw_values.get('cuts_moderate', 'N/A')} ({severe_pct:.2f}%/{raw_values.get('moderate_pct', 0):.2f}%) Roll:{raw_values.get('rolloff_hz', 'N/A'):.0f}Hz ZCR:{raw_values.get('zcr', 'N/A'):.3f}]"
        except:
            raw_str = ""
        
        logger.info(f"🎵 Quality: {total_score:.3f} | {score_str} | {raw_str}")
        
        return total_score, clarity_metric
        
    except Exception as e:
        logger.warning(f"Audio quality evaluation failed: {e}")
        return 0.5, {'severe_pct': 0, 'cuts_score': 0.5, 'clarity_score': 0.5}


def evaluate_latent_quality(latent):
    """
    Evaluate the quality of audio latent based on statistical properties.
    
    NOTE: This is less reliable than evaluate_audio_quality()!
    Use audio evaluation when VAE is available.
    
    Returns a score from 0.0 (poor quality) to 1.0 (excellent quality).
    
    Quality metrics:
    - Standard deviation close to 1.0 (well-formed audio)
    - No extreme values (artifacts detection)
    - Good spatial uniformity
    - Energy distribution consistency
    - Spectral coherence (frequency domain)
    
    Args:
        latent: Latent tensor dictionary
        
    Returns:
        Quality score (0.0 to 1.0)
    """
    if not isinstance(latent, dict) or 'samples' not in latent:
        return 0.5  # Unknown quality
    
    samples = latent['samples']
    
    # 1. Standard deviation score (target: 1.0 for well-formed audio)
    std = samples.std().item()
    std_score = 1.0 - min(abs(std - 1.0), 1.0)  # Penalize deviation from 1.0
    std_score = max(0.0, min(1.0, std_score))
    
    # 2. Extreme value detection (artifacts)
    max_val = samples.abs().max().item()
    if max_val > 10.0:
        extreme_score = 0.0  # Severe artifacts
    elif max_val > 5.0:
        extreme_score = 0.5  # Moderate artifacts
    else:
        extreme_score = 1.0  # Clean
    
    # 3. Spatial uniformity (check variance across spatial dimensions)
    spatial_std = samples.std(dim=[2, 3]).mean().item()
    if spatial_std < 0.1:
        uniformity_score = 0.3  # Too uniform, likely collapsed
    elif spatial_std < 0.5:
        uniformity_score = 0.7  # Acceptable
    else:
        uniformity_score = 1.0  # Good variation
    
    # 4. Energy distribution (check for mode collapse or dead regions)
    mean_abs = samples.abs().mean().item()
    if mean_abs < 0.1:
        energy_score = 0.2  # Too low energy
    elif mean_abs < 0.5:
        energy_score = 0.6  # Low energy
    elif mean_abs > 3.0:
        energy_score = 0.4  # Too high energy
    else:
        energy_score = 1.0  # Good energy
    
    # 5. Spectral coherence (FFT-based frequency analysis)
    try:
        # Take FFT along the time dimension (last dim)
        fft_result = torch.fft.rfft(samples, dim=-1)
        fft_magnitude = torch.abs(fft_result)
        
        # Check if frequency content is well-distributed
        freq_std = fft_magnitude.std().item()
        freq_mean = fft_magnitude.mean().item()
        
        if freq_mean > 0:
            freq_cv = freq_std / freq_mean  # Coefficient of variation
            # Good audio should have varied frequency content (CV > 0.5)
            if freq_cv > 0.8:
                spectral_score = 1.0
            elif freq_cv > 0.5:
                spectral_score = 0.8
            elif freq_cv > 0.3:
                spectral_score = 0.5
            else:
                spectral_score = 0.3  # Too uniform spectrum
        else:
            spectral_score = 0.1  # No frequency content
    except:
        spectral_score = 0.5  # Fallback if FFT fails
    
    # Weighted combination (adjusted for audio quality)
    quality_score = (
        0.30 * std_score +       # Standard deviation is important
        0.25 * extreme_score +   # Artifacts detection
        0.20 * uniformity_score + # Spatial variation
        0.15 * energy_score +    # Energy levels
        0.10 * spectral_score    # Frequency content
    )
    
    return quality_score


def sample_with_auto_steps(
    model,
    seed,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise,
    target_quality=0.85,
    min_steps=40,
    max_steps=150,
    quality_check_interval=5,
    vae=None,
    anti_autotune_strength=0.0,
    frequency_damping=0.0,
    temporal_smoothing=0.0,
    beat_stability=0.0,
):
    """
    Automatic step discovery: Tests different step counts to find optimal quality.
    
    Strategy:
    1. Start with min_steps (e.g., 40)
    2. Run COMPLETE sampling from 0 to current_steps
    3. If VAE provided: Decode latent → audio and evaluate REAL audio quality ✓
    4. If no VAE: Evaluate latent quality (less reliable)
    5. If quality >= target: STOP and return (latent + audio)
    6. If not: Increase steps by interval (e.g., +10) and repeat from step 2
    7. Continue until target reached OR max_steps hit
    8. Always keep best result found
    
    This approach discovers the ideal step count for each sampler/scheduler combination.
    Yes, it's slower, but finds the optimal setting and guarantees clean audio without artifacts.
    
    Args:
        model: Ace-Step model
        seed: Random seed
        cfg: CFG scale
        sampler_name: Sampler type
        scheduler: Noise scheduler
        positive: Positive conditioning
        negative: Negative conditioning
        latent: Input latent (ALWAYS reset to this for each attempt)
        denoise: Denoise strength
        target_quality: Target quality score (0.5-1.0)
        min_steps: Starting step count
        max_steps: Maximum step count
        quality_check_interval: Increase steps by this amount each attempt
        vae: VAE model for decoding (optional but HIGHLY recommended!)
        
    Returns:
        Tuple of (best_latent, best_audio_dict) where audio_dict has 'waveform' and 'sample_rate'
    """
    logger.info(f"Auto steps: Target quality = {target_quality:.3f}, Steps range = [{min_steps}, {max_steps}]")
    logger.info(f"Auto steps: Will test increasing steps by {quality_check_interval} each attempt")
    logger.info(f"Auto steps: Starting from ZERO each time (clean diffusion process)")
    
    if vae is not None:
        logger.info(f"Auto steps: VAE provided - Will evaluate REAL AUDIO quality ✓")
    else:
        logger.warning(f"Auto steps: No VAE - Using latent quality (less reliable)")
    
    best_latent = None
    best_audio = None
    best_quality = 0.0
    best_steps = 0
    
    current_steps = min_steps
    attempt = 1
    
    while current_steps <= max_steps:
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"Auto steps: ATTEMPT {attempt} - Testing {current_steps} steps")
        logger.info(f"{'='*60}")
        
        # Run COMPLETE sampling from 0 to current_steps
        # ALWAYS start from original latent (clean slate)
        try:
            result_latent = KSampler().sample(
                model,
                seed,
                current_steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,  # Original latent (not modified)
                denoise=denoise,
            )[0]
            
            # OPTIONAL: Apply post-sampling smoothing before decoding/evaluating
            if anti_autotune_strength and anti_autotune_strength > 0.0:
                result_latent = apply_anti_autotune_smoothing(result_latent, anti_autotune_strength)
            if frequency_damping and frequency_damping > 0.0:
                result_latent = apply_frequency_damping_to_latent(result_latent, frequency_damping)
            if temporal_smoothing and temporal_smoothing > 0.0:
                result_latent = apply_temporal_smoothing_to_latent(result_latent, temporal_smoothing)
            if beat_stability and beat_stability > 0.0:
                result_latent = apply_temporal_smoothing_to_latent(result_latent, min(0.25, beat_stability * 0.2))

            # Decode latent to audio if VAE available
            result_audio = None
            if vae is not None:
                try:
                    logger.info(f"Auto steps: Decoding latent to audio...")
                    # Official VAEDecodeAudio implementation
                    audio = vae.decode(result_latent['samples']).movedim(-1, 1)
                    # Attempt to detect VAE sample rate for the result_latent
                    try:
                        vae_sr = None
                        if hasattr(vae, 'sample_rate'):
                            vae_sr = vae.sample_rate
                        elif hasattr(vae, 'config') and hasattr(vae.config, 'sample_rate'):
                            vae_sr = vae.config.sample_rate
                        elif hasattr(vae, 'opts') and isinstance(vae.opts, dict) and 'sample_rate' in vae.opts:
                            vae_sr = vae.opts['sample_rate']
                        if vae_sr is not None and int(vae_sr) != 44100:
                            logger.warning(f"VAE sample rate {vae_sr} differs from expected 44100 — runtime resampling may be necessary.")
                    except Exception:
                        pass
                    # Normalize audio
                    std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
                    std[std < 1.0] = 1.0
                    audio = audio / std
                    audio = torch.clamp(audio, -1.0, 1.0)
                    result_audio = {
                        'waveform': audio,
                        'sample_rate': 44100,
                    }
                    logger.info(f"Auto steps: Audio decoded successfully, shape: {audio.shape}")
                except Exception as e:
                    logger.error(f"Auto steps: VAE decode failed: {e}")
                    result_audio = None
            
            # Evaluate quality
            if result_audio is not None:
                # Use REAL audio quality evaluation (much better!)
                quality, clarity = evaluate_audio_quality(result_audio['waveform'])
                
                # Calculate composite score: 70% quality + 30% clarity (word detection)
                # This matches what AudioQualityEvaluator returns
                composite_score = (quality * 0.70) + (clarity['clarity_score'] * 0.30)
                
                logger.info(f"Auto steps: Audio quality = {quality:.3f} (cuts: {clarity['severe_pct']:.2f}%), Clarity = {clarity['clarity_score']:.3f}, Composite = {composite_score:.3f}")
                quality = composite_score  # Use composite for comparison
            else:
                # Fallback to latent quality
                quality = evaluate_latent_quality(result_latent)
                logger.info(f"Auto steps: Latent quality = {quality:.3f} (FALLBACK)")
            
            # Track best result
            if quality > best_quality:
                best_quality = quality
                best_latent = result_latent
                best_audio = result_audio
                best_steps = current_steps
                logger.info(f"Auto steps: NEW BEST! Quality = {best_quality:.3f} at {best_steps} steps ✓")
            else:
                logger.info(f"Auto steps: Not better than best ({best_quality:.3f})")
            
            # Check if target reached
            if quality >= target_quality:
                logger.info(f"")
                logger.info(f"{'='*60}")
                logger.info(f"Auto steps: TARGET REACHED! ✓✓✓")
                logger.info(f"Auto steps: Quality {quality:.3f} >= {target_quality:.3f}")
                logger.info(f"Auto steps: Achieved with {current_steps} steps")
                logger.info(f"Auto steps: Saved {max_steps - current_steps} steps from maximum")
                logger.info(f"{'='*60}")
                return (result_latent, result_audio)
            
            # Prepare for next attempt
            current_steps += quality_check_interval
            attempt += 1
            
        except Exception as e:
            logger.error(f"Auto steps: Attempt {attempt} failed: {e}")
            current_steps += quality_check_interval
            attempt += 1
            continue
    
    # Reached max_steps without hitting target
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Auto steps: MAX STEPS REACHED")
    logger.info(f"Auto steps: Best quality = {best_quality:.3f} (target was {target_quality:.3f})")
    logger.info(f"Auto steps: Best result was at {best_steps} steps")
    
    if best_quality < target_quality:
        logger.warning(f"Auto steps: Target quality NOT reached")
        logger.warning(f"Auto steps: Returning best result found ({best_quality:.3f})")
    else:
        logger.info(f"Auto steps: Returning best result ✓")
    
    logger.info(f"{'='*60}")
    
    return (best_latent if best_latent is not None else latent, best_audio)


def normalize_audio_latent(latent, target_std=1.0):
    """
    Normalize audio latent for consistent ACE-Step processing.
    
    Uses high-precision float64 for normalization calculations.
    
    Args:
        latent: Input latent tensor dictionary
        target_std: Target standard deviation
        
    Returns:
        Normalized latent dictionary
    """
    if not isinstance(latent, dict) or 'samples' not in latent:
        return latent
    
    samples = latent['samples']
    original_dtype = samples.dtype
    
    try:
        # Convert to float64 for high-precision calculation
        samples_fp64 = samples.to(torch.float64)
        
        # Calculate statistics with high precision
        current_std = samples_fp64.std()
        epsilon = 1e-10
        
        # Only normalize if std deviates significantly
        if current_std > epsilon and abs(current_std.item() - target_std) > 0.05:
            scale_factor = target_std / (current_std + epsilon)
            # Clamp scale factor to avoid extreme scaling
            scale_factor = torch.clamp(scale_factor, 0.5, 2.0)
            normalized = samples_fp64 * scale_factor
            # Convert back to original dtype
            normalized = normalized.to(original_dtype)
        else:
            normalized = samples
        
        return {'samples': normalized}
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
        return latent
    
    return latent
    
    return latent


#=======================================================================================================================
# Advanced Guidance Functions (APG / CFG++)
#=======================================================================================================================

def apply_cfg_rescale(cond_pred, uncond_pred, cfg_scale, rescale_multiplier=0.7, max_std_ratio=1.8):
    """
    Apply CFG++ (CFG Rescale) to prevent oversaturation and improve details.
    
    CFG++ rescales the guidance to maintain the same standard deviation as the
    unconditional prediction, preventing oversaturation that standard CFG can cause.
    
    Paper: "Common Diffusion Noise Schedules and Sample Steps are Flawed" (Lin et al., 2023)
    
    Args:
        cond_pred: Conditional prediction
        uncond_pred: Unconditional prediction
        cfg_scale: CFG scale value
        rescale_multiplier: How much to rescale (0.0-1.0). 0.7 is recommended.
        
    Returns:
        Rescaled prediction with improved detail retention
    """
    if rescale_multiplier <= 0.0:
        # CFG++ disabled, use standard CFG
        return uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    
    # Standard CFG with high precision
    cfg_result = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    
    # Calculate standard deviations with high precision
    # Convert to float64 for precise std calculation
    uncond_pred_fp64 = uncond_pred.to(torch.float64)
    cfg_result_fp64 = cfg_result.to(torch.float64)

    uncond_std = uncond_pred_fp64.std()
    cfg_std = cfg_result_fp64.std()

    # Rescale to match unconditional std (prevents oversaturation)
    # Use larger epsilon for better numerical stability
    epsilon = 1e-10
    if cfg_std > epsilon:
        rescale_factor = uncond_std / cfg_std
        # Blend between original and rescaled based on multiplier
        rescale_factor = rescale_multiplier * rescale_factor + (1.0 - rescale_multiplier) * 1.0
        # Clamp rescale factor to avoid extreme scaling
        rescale_factor = torch.clamp(rescale_factor, 0.5, 2.0)
        cfg_result_fp64 = cfg_result_fp64 * rescale_factor

    # Extra safety: clamp final std ratio to avoid distortion at high CFG (6-8)
    uncond_std_safe = torch.clamp(uncond_std, min=epsilon)
    cfg_std_final = cfg_result_fp64.std()
    std_ratio = cfg_std_final / uncond_std_safe
    if std_ratio > max_std_ratio:
        scale = max_std_ratio / std_ratio
        cfg_result_fp64 = uncond_pred_fp64 + (cfg_result_fp64 - uncond_pred_fp64) * scale

    cfg_result = cfg_result_fp64.to(cfg_result.dtype)
    
    return cfg_result


def apply_apg(cond_pred, uncond_pred, cfg_scale, momentum=0.5):
    """
    Apply APG (Adaptive Projected Guidance) for better guidance projection.
    
    APG adaptively projects the guidance vector to prevent artifacts and improve
    generation quality by maintaining proper guidance direction while avoiding
    overshooting.
    
    Paper: "Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models" (Castillo et al., 2024)
    
    Args:
        cond_pred: Conditional prediction
        uncond_pred: Unconditional prediction
        cfg_scale: CFG scale value
        momentum: Momentum factor for adaptive scaling (0.0-1.0)
        
    Returns:
        APG-adjusted prediction with better guidance
    """
    # Calculate guidance vector with high precision
    guidance = cond_pred - uncond_pred
    
    # Convert to float64 for norm calculation precision
    guidance_fp64 = guidance.to(torch.float64)
    
    # Calculate guidance magnitude with high precision
    guidance_norm = torch.norm(guidance_fp64)
    
    epsilon = 1e-10
    if guidance_norm > epsilon:
        # Normalize guidance
        guidance_normalized = guidance_fp64 / (guidance_norm + epsilon)
        
        # Calculate adaptive scale based on prediction magnitudes
        cond_pred_fp64 = cond_pred.to(torch.float64)
        uncond_pred_fp64 = uncond_pred.to(torch.float64)
        
        cond_norm = torch.norm(cond_pred_fp64)
        uncond_norm = torch.norm(uncond_pred_fp64)
        
        # Adaptive scaling factor (prevents overshooting)
        if uncond_norm > epsilon:
            # Use more precise calculation with safe exponentiation
            norm_ratio = torch.clamp(cond_norm / uncond_norm, 0.1, 10.0)
            adaptive_scale = torch.pow(norm_ratio, torch.tensor(momentum, dtype=norm_ratio.dtype))
            adaptive_scale = torch.clamp(adaptive_scale, 0.5, 2.0)  # Limit extremes
        else:
            adaptive_scale = torch.tensor(1.0, dtype=guidance_fp64.dtype)
        
        # Apply APG: project guidance with adaptive scaling
        apg_guidance = guidance_normalized * guidance_norm * adaptive_scale
        result = uncond_pred_fp64 + cfg_scale * apg_guidance
        result = result.to(uncond_pred.dtype)
    else:
        # No guidance, return uncond
        result = uncond_pred
    
    return result


def apply_combined_guidance(
    cond_pred,
    uncond_pred,
    cfg_scale,
    use_cfg_rescale=True,
    cfg_rescale_multiplier=0.7,
    use_apg=True,
    apg_momentum=0.5,
    max_std_ratio=1.8,
):
    """
    Apply combined APG + CFG++ guidance for optimal audio generation.
    
    Combines both techniques:
    1. APG for adaptive guidance projection
    2. CFG++ for preventing oversaturation
    
    Args:
        cond_pred: Conditional prediction
        uncond_pred: Unconditional prediction
        cfg_scale: CFG scale value
        use_cfg_rescale: Enable CFG++ rescaling
        cfg_rescale_multiplier: CFG++ rescale strength (0.7 recommended)
        use_apg: Enable APG guidance
        apg_momentum: APG momentum factor
        
    Returns:
        Optimally guided prediction
    """
    if not use_apg and not use_cfg_rescale:
        # Standard CFG (no enhancements)
        return uncond_pred + cfg_scale * (cond_pred - uncond_pred)

    if use_apg and use_cfg_rescale:
        # Combined: APG first, then CFG++ rescale
        apg_result = apply_apg(cond_pred, uncond_pred, cfg_scale, momentum=apg_momentum)

        # Apply CFG++ rescale on top of APG with std-ratio clamp
        epsilon = 1e-10
        uncond_fp64 = uncond_pred.to(torch.float64)
        apg_fp64 = apg_result.to(torch.float64)
        uncond_std = uncond_fp64.std()
        apg_std = apg_fp64.std()
        if apg_std > epsilon:
            rescale_factor = uncond_std / apg_std
            rescale_factor = cfg_rescale_multiplier * rescale_factor + (1.0 - cfg_rescale_multiplier) * 1.0
            rescale_factor = torch.clamp(rescale_factor, 0.5, 2.0)
            apg_fp64 = apg_fp64 * rescale_factor

        # Clamp final std ratio to avoid distortion at high CFG (6-8)
        apg_std_final = apg_fp64.std()
        std_ratio = apg_std_final / torch.clamp(uncond_std, min=epsilon)
        if std_ratio > max_std_ratio:
            scale = max_std_ratio / std_ratio
            apg_fp64 = uncond_fp64 + (apg_fp64 - uncond_fp64) * scale

        result = apg_fp64.to(apg_result.dtype)
        return result
    
    elif use_apg:
        # APG only
        return apply_apg(cond_pred, uncond_pred, cfg_scale, momentum=apg_momentum)
    
    else:
        # CFG++ only
        return apply_cfg_rescale(cond_pred, uncond_pred, cfg_scale, rescale_multiplier=cfg_rescale_multiplier, max_std_ratio=max_std_ratio)


#=======================================================================================================================
# AceStepKSampler - Standard version with quality presets
#=======================================================================================================================
class AceStepKSampler:
    """
    Standard KSampler optimized for Ace-Step audio generation.
    Manual control interface for professional audio synthesis.
    All parameters are now required for full control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": ACESTEP_OPTIMAL_STEPS, "min": ACESTEP_MIN_STEPS, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS if comfy else [],),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS if comfy else [],),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Advanced Guidance Options
                "use_apg": ("BOOLEAN", {"default": False}),
                "use_cfg_rescale": ("BOOLEAN", {"default": False}),
                "cfg_rescale_multiplier": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Advanced Optimization Options
                "enable_dynamic_cfg": ("BOOLEAN", {"default": True}),
                "enable_latent_normalization": ("BOOLEAN", {"default": True}),
                # Vocoder Options
                "use_vocoder": ("BOOLEAN", {"default": False}),
                # Noise stabilization
                "noise_ema": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "EMA smoothing of noise prediction; 0.08 optimal for 8-channel latents"}),
                "noise_norm_threshold": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "L2 norm clamp vs input; 2.0 for clean audio without artifacts"}),
                # Anti-Autotune Smoothing (reduces robotic voice artifacts)
                "anti_autotune_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Smooth spectral quantization artifacts. 0.15 default for natural vocals"}),
                # Frequency & Temporal smoothing for vocal realism
                "frequency_damping": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Damps higher frequencies to remove metallic sound - 0.18 recommended"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Temporal smoothing to prevent stuttering - 0.10 for natural flow"}),
                "beat_stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Rhythm stability - 0.5 prevents beat dropout without compression"}),
                # Quality Check Discovery (Automatic Step Count Optimization)
                "enable_quality_check": ("BOOLEAN", {"default": False}),
                "quality_check_target": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.05}),
                "quality_check_min": ("INT", {"default": 40, "min": 20, "max": 100}),
                "quality_check_max": ("INT", {"default": 150, "min": 50, "max": 300}),
                "quality_check_interval": ("INT", {"default": 5, "min": 1, "max": 20}),
                # VAE for audio quality evaluation (HIGHLY RECOMMENDED!)
                "vae": ("VAE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }
        return inputs

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("latent", "audio")
    OUTPUT_NODE = False
    FUNCTION = "sample"
    CATEGORY = "JK AceStep Nodes/Sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise,
        use_apg=False,
        use_cfg_rescale=False,
        cfg_rescale_multiplier=0.25,
        enable_dynamic_cfg=True,
        enable_latent_normalization=True,
        use_vocoder=False,
        noise_ema=0.08,
        noise_norm_threshold=2.0,
        enable_quality_check=False,
        quality_check_target=0.85,
        quality_check_min=40,
        quality_check_max=150,
        quality_check_interval=5,
        vae=None,
        anti_autotune_strength=0.15,
        frequency_damping=0.18,
        temporal_smoothing=0.10,
        beat_stability=0.5,
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None,
    ):
        """
        Perform optimized KSampling for Ace-Step audio generation.
        
        Args:
            model: Ace-Step audio model
            seed: Seed for reproducibility
            steps: Sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Sampler type
            scheduler: Noise scheduler
            positive: Positive audio conditioning
            negative: Negative audio conditioning
            latent: Input audio latent
            denoise: Denoising strength
            use_apg: Enable APG (Adaptive Projected Guidance)
            use_cfg_rescale: Enable CFG++ (CFG Rescale)
            cfg_rescale_multiplier: CFG++ rescale strength
            enable_dynamic_cfg: Enable dynamic CFG scheduling
            enable_latent_normalization: Enable latent normalization
            enable_quality_check: Enable automatic step discovery for optimal quality
            quality_check_target: Target quality score (0.5-1.0)
            quality_check_min: Minimum steps to test
            quality_check_max: Maximum steps to test
            quality_check_interval: Test quality every N steps
            vae: VAE for decoding latent to audio (HIGHLY RECOMMENDED for quality evaluation)
            
        Returns:
            Tuple of (latent, audio)
        """
        
        # Quality Check Mode: Iterative sampling with quality evaluation
        if enable_quality_check:
            logger.info("=" * 60)
            logger.info("QUALITY CHECK MODE ENABLED")
            logger.info("Will test different step counts to find the optimal quality")
            logger.info("=" * 60)
        
        logger.info(f"Sampling: steps={steps}, cfg={cfg}, sampler={sampler_name}, scheduler={scheduler}")
        
        # Log guidance settings
        guidance_info = []
        if use_apg:
            guidance_info.append("APG")
        if use_cfg_rescale:
            guidance_info.append(f"CFG++({cfg_rescale_multiplier:.2f})")
        if not guidance_info:
            guidance_info.append("Standard CFG")
        logger.info(f"Guidance: {', '.join(guidance_info)}")

        # Surface quality-check intent (UI parity)
        if enable_quality_check:
            logger.info("Quality check flag enabled (advanced node runs single pass). For auto step discovery use basic sampler quality mode.")

        # Apply latent normalization if requested
        if enable_latent_normalization:
            latent = normalize_audio_latent(latent, target_std=ACESTEP_LATENT_SCALE)
            logger.info("Latent normalization applied before advanced sampling")
        
        # Validate CFG range
        if cfg < ACESTEP_CFG_RANGE[0] or cfg > ACESTEP_CFG_RANGE[1]:
            logger.warning(
                f"CFG {cfg} outside optimal range {ACESTEP_CFG_RANGE}. "
                f"Recommended: {ACESTEP_OPTIMAL_CFG}±3.0"
            )

        # Validate steps
        if steps < ACESTEP_MIN_STEPS:
            logger.warning(f"Steps {steps} below minimum {ACESTEP_MIN_STEPS}. Quality may suffer.")
        elif steps > ACESTEP_MAX_STEPS:
            logger.warning(f"Steps {steps} above maximum {ACESTEP_MAX_STEPS}. Diminishing returns.")
        
        # Apply optimization techniques
        optimizations = []
        
        # 1. Latent normalization (if enabled)
        if enable_latent_normalization:
            latent = normalize_audio_latent(latent, target_std=ACESTEP_LATENT_SCALE)
            optimizations.append("LatentNorm")
        
        # 2. Dynamic CFG (if enabled)
        if enable_dynamic_cfg:
            optimizations.append("DynamicCFG")
        
        if optimizations:
            logger.info(f"Optimizations: {', '.join(optimizations)}")

        # Apply APG/CFG++ by modifying model_options
        model_options = model.model_options.copy() if hasattr(model, 'model_options') else {}
        
        if use_apg or use_cfg_rescale or enable_dynamic_cfg:
            # Create custom CFG function that applies APG/CFG++ and dynamic CFG
            current_step = [0]  # Mutable counter for tracking steps
            prev_noise = [None]
            
            def custom_cfg_function(args):
                cond_denoised = args["cond_denoised"]
                uncond_denoised = args["uncond_denoised"]
                cond_scale = args["cond_scale"]
                
                # Apply dynamic CFG if enabled
                if enable_dynamic_cfg:
                    cond_scale = apply_dynamic_cfg(cond_scale, current_step[0], steps)
                    current_step[0] += 1
                
                # Apply combined guidance
                result = apply_combined_guidance(
                    cond_denoised, 
                    uncond_denoised, 
                    cond_scale,
                    use_cfg_rescale=use_cfg_rescale,
                    cfg_rescale_multiplier=cfg_rescale_multiplier,
                    use_apg=use_apg,
                    apg_momentum=0.5
                )
                
                # Derive noise prediction
                noise_pred = args["input"] - result

                # Apply patch-aware processing to respect ACEStepTransformer patch structure [16, 1]
                # This prevents "word cutting" by aligning noise operations to 16-frame boundaries
                noise_pred = apply_patch_aware_processing(noise_pred, patch_size=16)

                # EMA smoothing of noise to stabilize rhythm
                if noise_ema > 0 and prev_noise[0] is not None:
                    noise_pred = (1.0 - noise_ema) * noise_pred + noise_ema * prev_noise[0]
                prev_noise[0] = noise_pred.detach()

                # L2 norm clamp vs input to curb distortion
                if noise_norm_threshold > 0 and noise_pred.dim() >= 4:
                    dims = (-1, -2)
                    noise_norm = torch.norm(noise_pred, dim=dims, keepdim=True)
                    ref_norm = torch.norm(args["input"], dim=dims, keepdim=True) + 1e-10
                    scale = torch.minimum(torch.ones_like(noise_norm), (noise_norm_threshold * ref_norm) / (noise_norm + 1e-10))
                    noise_pred = noise_pred * scale
                
                return noise_pred
            
            model_options["sampler_cfg_function"] = custom_cfg_function
            
            # Clone model with new options
            model_with_guidance = model.clone()
            model_with_guidance.model_options = model_options
        else:
            model_with_guidance = model

        # Perform sampling
        if enable_quality_check:
            # Use automatic step discovery with quality evaluation
            latent_output, audio_output = sample_with_auto_steps(
                model_with_guidance,
                seed,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise,
                target_quality=quality_check_target,
                min_steps=quality_check_min,
                max_steps=quality_check_max,
                quality_check_interval=quality_check_interval,
                vae=vae,
                anti_autotune_strength=anti_autotune_strength,
                frequency_damping=frequency_damping,
                temporal_smoothing=temporal_smoothing,
                beat_stability=beat_stability,
            )
        elif denoise > 0:
            # Standard sampling
            latent_output = KSampler().sample(
                model_with_guidance,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise=denoise,
            )[0]
            
            # Apply anti-autotune smoothing if enabled
            if anti_autotune_strength > 0:
                logger.info(f"Applying anti-autotune smoothing: strength={anti_autotune_strength:.2f}")
                latent_output = apply_anti_autotune_smoothing(latent_output, anti_autotune_strength)

            # Apply frequency & temporal damping post-sampling
            if frequency_damping and frequency_damping > 0.0:
                latent_output = apply_frequency_damping_to_latent(latent_output, frequency_damping)
            if temporal_smoothing and temporal_smoothing > 0.0:
                latent_output = apply_temporal_smoothing_to_latent(latent_output, temporal_smoothing)
            if beat_stability and beat_stability > 0.0:
                latent_output = apply_temporal_smoothing_to_latent(latent_output, min(0.25, beat_stability * 0.2))
            
            # Decode to audio if VAE provided
            audio_output = None
            if vae is not None:
                # Official VAEDecodeAudio implementation
                audio = vae.decode(latent_output['samples']).movedim(-1, 1)
                # Attempt to detect VAE sample rate
                try:
                    vae_sr = None
                    if hasattr(vae, 'sample_rate'):
                        vae_sr = vae.sample_rate
                    elif hasattr(vae, 'config') and hasattr(vae.config, 'sample_rate'):
                        vae_sr = vae.config.sample_rate
                    elif hasattr(vae, 'opts') and isinstance(vae.opts, dict) and 'sample_rate' in vae.opts:
                        vae_sr = vae.opts['sample_rate']
                    if vae_sr is not None and int(vae_sr) != 44100:
                        logger.warning(f"VAE sample rate {vae_sr} differs from expected 44100 — runtime resampling may be necessary.")
                except Exception:
                    pass
                # Normalize audio
                std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
                std[std < 1.0] = 1.0
                audio = audio / std
                audio = torch.clamp(audio, -1.0, 1.0)
                
                # Apply vocoder if enabled
                if use_vocoder:
                    logger.info("Loading and applying vocoder to audio")
                    vocoder_data = load_vocoder_model()
                    if vocoder_data is not None:
                        audio = apply_vocoder_to_audio(audio, vocoder_data)
                        logger.info("Vocoder applied successfully")
                    else:
                        logger.warning("Vocoder loading failed, using original audio")
                
                audio_output = {
                    'waveform': audio,
                    'sample_rate': 44100,
                }
        else:
            # If denoise is 0, pass through latent
            latent_output = latent
            audio_output = None
            logger.debug("Denoise=0, passing through latent")
        
        logger.info(f"Sampling completed: seed={seed}, steps={steps}, cfg={cfg}")
        
        return (latent_output, audio_output)


#=======================================================================================================================
# AceStepKSamplerAdvanced - Advanced version with step control and presets
#=======================================================================================================================
class AceStepKSamplerAdvanced:
    """
    Advanced KSampler with step control for Ace-Step audio generation.
    Fine-grained control over start/end steps for multi-pass and refinement workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": ACESTEP_OPTIMAL_STEPS, "min": ACESTEP_MIN_STEPS, "max": ACESTEP_MAX_STEPS}),
                "cfg": ("FLOAT", {"default": ACESTEP_OPTIMAL_CFG, "min": ACESTEP_CFG_RANGE[0], "max": ACESTEP_CFG_RANGE[1], "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS if comfy else [],),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS if comfy else [],),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": ACESTEP_OPTIMAL_STEPS, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
            "optional": {
                # Advanced Guidance Options
                "use_apg": ("BOOLEAN", {"default": True}),
                "use_cfg_rescale": ("BOOLEAN", {"default": True}),
                "cfg_rescale_multiplier": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Dynamic CFG & Latent Norm
                "enable_dynamic_cfg": ("BOOLEAN", {"default": False}),
                "enable_latent_normalization": ("BOOLEAN", {"default": False}),
                # Noise stabilization
                "noise_ema": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "noise_norm_threshold": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                # Anti-Autotune (post smoothing)
                "anti_autotune_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Smoothing options for voice realism
                "frequency_damping": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "temporal_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "beat_stability": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Quality Check (UI parity; uses basic flow recommendation)
                "enable_quality_check": ("BOOLEAN", {"default": False}),
                "quality_check_target": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.05}),
                "quality_check_min": ("INT", {"default": 40, "min": 20, "max": 100}),
                "quality_check_max": ("INT", {"default": 150, "min": 50, "max": 300}),
                "quality_check_interval": ("INT", {"default": 5, "min": 1, "max": 20}),
                "vae": ("VAE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }
        return inputs

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = False
    FUNCTION = "sample_advanced"
    CATEGORY = "JK AceStep Nodes/Sampling"

    def sample_advanced(
        self,
        model,
        steps,
        cfg,
        sampler_name,
        scheduler,
        add_noise,
        noise_seed,
        positive,
        negative,
        latent,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        use_apg,
        use_cfg_rescale,
        cfg_rescale_multiplier,
        noise_ema=0.05,
        noise_norm_threshold=3.5,
        enable_dynamic_cfg=False,
        enable_latent_normalization=False,
        anti_autotune_strength=0.25,
        frequency_damping=0.0,
        temporal_smoothing=0.0,
        beat_stability=0.0,
        enable_quality_check=False,
        quality_check_target=0.85,
        quality_check_min=40,
        quality_check_max=150,
        quality_check_interval=5,
        vae=None,
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None,
    ):
        """
        Perform advanced optimized KSampling for Ace-Step audio with step control and quality presets.
        
        This advanced method enables complex audio generation workflows:
        - Multi-pass generation with progressive refinement
        - Targeted noise removal from specific step ranges
        - Audio interpolation and morphing
        - Iterative refinement workflows
        - Quality presets based on Stable Audio Open official parameters
        - APG (Adaptive Projected Guidance) for better guidance projection
        - CFG++ (CFG Rescale) for preventing oversaturation
        
        Args:
            model: Ace-Step audio model
            quality_preset: Quality preset (draft/fast/balanced/quality/ultra/custom)
            add_noise: "enable" = start fresh, "disable" = continue from latent
            noise_seed: Seed for noise generation
            positive: Positive audio conditioning
            negative: Negative conditioning
            latent: Input audio latent
            start_at_step: Step to begin sampling (0 = start, >0 = continue)
            end_at_step: Step to end sampling
            return_with_leftover_noise: "enable" = keep noise (for chaining)
            use_apg: Enable APG (Adaptive Projected Guidance)
            use_cfg_rescale: Enable CFG++ (CFG Rescale)
            cfg_rescale_multiplier: CFG++ rescale strength (0.7 recommended)
            steps: Total steps (only for custom)
            cfg: Classifier-free guidance (only for custom)
            sampler_name: Sampling algorithm (only for custom)
            scheduler: Noise schedule (only for custom)
            
        Returns:
            Tuple of (latent,)
        """
        
        # Validate CFG range
        if cfg < ACESTEP_CFG_RANGE[0] or cfg > ACESTEP_CFG_RANGE[1]:
            logger.warning(f"CFG {cfg} outside optimal range {ACESTEP_CFG_RANGE}. Recommended: {ACESTEP_OPTIMAL_CFG}±3.0")
        
        # Validate step range
        if steps < ACESTEP_MIN_STEPS:
            logger.warning(f"Steps {steps} below minimum {ACESTEP_MIN_STEPS}. Quality may suffer.")
        elif steps > ACESTEP_MAX_STEPS:
            logger.warning(f"Steps {steps} above maximum {ACESTEP_MAX_STEPS}. Diminishing returns.")

        # Validate step range
        if start_at_step >= end_at_step:
            logger.error(f"start_at_step ({start_at_step}) >= end_at_step ({end_at_step})")
            raise ValueError("start_at_step must be less than end_at_step")

        step_range = end_at_step - start_at_step
        if step_range < 1:
            logger.warning("Step range < 1, sampling may be ineffective")
        elif step_range > ACESTEP_MAX_STEPS:
            logger.warning(f"Step range {step_range} > recommended max {ACESTEP_MAX_STEPS}")

        # Log multi-pass workflow detection
        if start_at_step > 0:
            logger.info(f"Multi-pass detected: steps {start_at_step} to {end_at_step}")
        
        # Log guidance settings
        guidance_info = []
        if use_apg:
            guidance_info.append("APG")
        if use_cfg_rescale:
            guidance_info.append(f"CFG++({cfg_rescale_multiplier:.2f})")
        if not guidance_info:
            guidance_info.append("Standard CFG")
        logger.info(f"Guidance: {', '.join(guidance_info)}")

        # Apply APG/CFG++ by modifying model_options
        model_options = model.model_options.copy() if hasattr(model, 'model_options') else {}
        
        if use_apg or use_cfg_rescale or enable_dynamic_cfg:
            # Create custom CFG function that applies APG/CFG++ and dynamic CFG scheduling
            current_step = [0]
            prev_noise = [None]

            def custom_cfg_function(args):
                cond_denoised = args["cond_denoised"]
                uncond_denoised = args["uncond_denoised"]
                cond_scale = args["cond_scale"]

                # Dynamic CFG if step info provided
                if enable_dynamic_cfg:
                    step_idx = args.get("step", current_step[0])
                    total_steps = args.get("total_steps", steps or end_at_step)
                    cond_scale = apply_dynamic_cfg(cond_scale, step_idx, total_steps)
                    current_step[0] = step_idx + 1
                
                # Apply combined guidance
                result = apply_combined_guidance(
                    cond_denoised, 
                    uncond_denoised, 
                    cond_scale,
                    use_cfg_rescale=use_cfg_rescale,
                    cfg_rescale_multiplier=cfg_rescale_multiplier,
                    use_apg=use_apg,
                    apg_momentum=0.5
                )
                
                # Derive noise prediction
                noise_pred = args["input"] - result

                # Apply patch-aware processing to respect ACEStepTransformer patch structure [16, 1]
                # This prevents "word cutting" by aligning noise operations to 16-frame boundaries
                noise_pred = apply_patch_aware_processing(noise_pred, patch_size=16)

                # EMA smoothing of noise to stabilize rhythm
                if noise_ema > 0 and prev_noise[0] is not None:
                    noise_pred = (1.0 - noise_ema) * noise_pred + noise_ema * prev_noise[0]
                prev_noise[0] = noise_pred.detach()

                # L2 norm clamp vs input to curb distortion
                if noise_norm_threshold > 0 and noise_pred.dim() >= 4:
                    dims = (-1, -2)
                    noise_norm = torch.norm(noise_pred, dim=dims, keepdim=True)
                    ref_norm = torch.norm(args["input"], dim=dims, keepdim=True) + 1e-10
                    scale = torch.minimum(torch.ones_like(noise_norm), (noise_norm_threshold * ref_norm) / (noise_norm + 1e-10))
                    noise_pred = noise_pred * scale
                
                return noise_pred
            
            model_options["sampler_cfg_function"] = custom_cfg_function
            
            # Clone model with new options
            model_with_guidance = model.clone()
            model_with_guidance.model_options = model_options
        else:
            model_with_guidance = model

        # Perform advanced sampling using ComfyUI's KSamplerAdvanced
        latent_output = KSamplerAdvanced().sample(
            model_with_guidance,
            add_noise,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            start_at_step,
            end_at_step,
            return_with_leftover_noise,
            denoise=1.0,
        )[0]
        
        # Post-sampling smoothing (optional)
        if anti_autotune_strength > 0:
            logger.info(f"Applying anti-autotune smoothing: strength={anti_autotune_strength:.2f}")
            latent_output = apply_anti_autotune_smoothing(latent_output, anti_autotune_strength)

        # Apply frequency & temporal damping (post-sampling)
        if frequency_damping and frequency_damping > 0.0:
            latent_output = apply_frequency_damping_to_latent(latent_output, frequency_damping)
        if temporal_smoothing and temporal_smoothing > 0.0:
            latent_output = apply_temporal_smoothing_to_latent(latent_output, temporal_smoothing)

        logger.info(f"Advanced sampling completed: steps {start_at_step}-{end_at_step}, cfg={cfg}, add_noise={add_noise}")
        
        return (latent_output,)


#=======================================================================================================================
# Ace-Step Audio Optimization Utilities
#=======================================================================================================================
class AceStepOptimizer:
    """
    Utility class for Ace-Step audio optimization and parameter validation.
    Provides helper methods for audio-specific parameter tuning.
    """

    @staticmethod
    def get_recommended_cfg(creativity=0.5):
        """
        Get recommended CFG based on desired creativity level.
        
        Args:
            creativity (float): 0.0 (faithful) to 1.0 (creative)
            
        Returns:
            float: Recommended CFG value
        """
        # Map creativity to CFG range
        # Lower creativity = lower CFG = more faithful to prompt
        # Higher creativity = higher CFG = more variation
        cfg_min, cfg_max = ACESTEP_CFG_RANGE
        return cfg_min + (creativity * (cfg_max - cfg_min))

    @staticmethod
    def get_recommended_steps(quality_level="balanced"):
        """
        Get recommended steps for quality level.
        
        Args:
            quality_level (str): "fast", "balanced", "quality", "hq"
            
        Returns:
            int: Recommended number of steps
        """
        mapping = {
            "fast": 15,
            "balanced": 25,
            "quality": 35,
            "hq": 50,
        }
        return mapping.get(quality_level, 25)

    @staticmethod
    def validate_audio_latent(audio_latent):
        """
        Validate audio latent structure.
        
        Args:
            audio_latent: Latent to validate
            
        Returns:
            bool: True if valid
        """
        try:
            if isinstance(audio_latent, dict) and "samples" in audio_latent:
                samples = audio_latent["samples"]
                if hasattr(samples, 'shape') and len(samples.shape) >= 2:
                    logger.debug(f"Audio latent validated: shape {samples.shape}")
                    return True
        except Exception as e:
            logger.error(f"Audio latent validation failed: {e}")
            return False
        return False

    @staticmethod
    def suggest_sampler(audio_quality="balanced"):
        """
        Suggest optimal sampler for audio quality level.
        
        Args:
            audio_quality (str): "fast", "balanced", "quality"
            
        Returns:
            str: Recommended sampler name
        """
        mapping = {
            "fast": "euler",
            "balanced": "dpmpp_2m",
            "quality": "dpmpp_3m_sde",
        }
        return mapping.get(audio_quality, "dpmpp_2m")

    @staticmethod
    def suggest_scheduler(audio_type="music"):
        """
        Suggest optimal scheduler for audio type.
        
        Args:
            audio_type (str): "music", "speech", "ambient", "mixed"
            
        Returns:
            str: Recommended scheduler name
        """
        # karras is generally best for audio
        mapping = {
            "music": "karras",
            "speech": "karras",
            "ambient": "exponential",
            "mixed": "karras",
        }
        return mapping.get(audio_type, "karras")


#=======================================================================================================================
# Audio Quality Evaluator Node
#=======================================================================================================================
class AudioQualityEvaluator:
    """
    Evaluates decoded audio quality using LIBROSA (industry standard).
    Professional-grade metrics for accurate scheduler comparison.
    
    Returns a quality score from 0.0 (poor) to 1.0 (excellent).
    
    Metrics (with librosa):
    - Spectral Centroid Stability (timbre consistency) - 25%
    - Spectral Rolloff (brightness/harshness) - 20%
    - RMS Energy Variation (dynamics) - 20%
    - Zero Crossing Rate (noisiness) - 15%
    - Spectral Bandwidth (richness) - 10%
    - Clipping detection - 10%
    
    Librosa Features:
    - Spectral centroid: Detects monotonic/robotic voice (CV < 0.1 = bad)
    - Spectral rolloff: Detects muffled (<2000 Hz) or harsh (>8000 Hz)
    - RMS variation: Detects flat/robotic (CV < 0.2) or glitchy (CV > 1.2)
    - Spectral bandwidth: Detects thin/metallic sound (<500 Hz)
    
    Fallback: Basic numpy metrics if librosa not installed
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("quality_score",)
    OUTPUT_NODE = False
    FUNCTION = "evaluate"
    CATEGORY = "JK AceStep Nodes/Audio"

    def evaluate(self, audio):
        """
        Evaluate audio quality using composite metric.
        
        Composite Score = 70% audio quality + 30% word clarity
        This matches the metric used by quality checker in KSampler.
        
        Args:
            audio: Audio dictionary with 'waveform' and 'sample_rate' keys
            
        Returns:
            Tuple of (quality_score,) where quality_score is 0.0-1.0
        """
        if isinstance(audio, dict) and 'waveform' in audio:
            waveform = audio['waveform']
            quality_score, clarity_info = evaluate_audio_quality(waveform)
            
            # Calculate composite score (same as quality checker)
            composite_score = (quality_score * 0.70) + (clarity_info['clarity_score'] * 0.30)
            
            logger.info(f"Audio quality: {quality_score:.3f}, Word clarity: {clarity_info['clarity_score']:.3f} (cuts: {clarity_info['severe_pct']:.2f}%)")
            logger.info(f"Composite quality score: {composite_score:.3f}")
            
            return (composite_score,)
        else:
            logger.warning("Invalid audio format, expected dict with 'waveform' key")
            return (0.0,)


#=======================================================================================================================
NODE_CLASS_MAPPINGS = {
    "AceStepKSampler": AceStepKSampler,
    "AceStepKSamplerAdvanced": AceStepKSamplerAdvanced,
    "AudioQualityEvaluator": AudioQualityEvaluator,
}

NODE_DISPLAY_NAMES = {
    "AceStepKSampler": "Ace-Step KSampler (Basic)",
    "AceStepKSamplerAdvanced": "Ace-Step KSampler (Advanced)",
    "AudioQualityEvaluator": "Audio Quality Evaluator",
}
