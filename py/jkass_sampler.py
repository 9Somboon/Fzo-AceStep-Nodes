# JKASS Sampler for ComfyUI
# Clean, stable implementation optimized for ACE-Step audio

import torch
from tqdm import trange


@torch.no_grad()
def sample_jkass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    ancestral_eta=1.0,
    ancestral_seed=42,
    blend_mode="lerp",
    beat_stability=1.0,
    frequency_damping=0.0,
    temporal_smoothing=0.0,
    **kwargs
):
    """
    JKASS sampler for ACE-Step audio generation.
    
    Clean implementation that preserves audio fidelity.
    Uses proven PingPong algorithm with proper tensor handling.
    
    Parameters beat_stability, frequency_damping, temporal_smoothing are kept
    for API compatibility but NOT actively used (set to 0.0 defaults).
    They were causing audio artifacts, so we keep the clean original algorithm.
    
    Args:
        model: wrapped model that accepts (x, sigma, **extra_args) 
        x: initial latent
        sigmas: full sigma schedule tensor
        extra_args: dict with conditioning, seed, model_options, etc.
        callback: optional step callback
        disable: disable progress bar
        noise_sampler: optional noise sampler function
    
    Returns: denoised latent tensor
    """
    if extra_args is None:
        extra_args = {}
    
    # Check minimum length
    if len(sigmas) <= 1:
        return x
    
    # Create batch size tensor for model calls (CRITICAL FIX)
    # This is the KEY difference from naive implementation
    s_in = x.new_ones([x.shape[0]])
    
    # Initialize outputs
    x_current = x.clone()
    
    # Main sampling loop - iterate through sigma schedule
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # CRITICAL: Call model with sigma * s_in (broadcasted to batch_size)
        # This is how all ComfyUI samplers do it - NOT a scalar sigma
        denoised = model(x_current, sigma * s_in, **extra_args)
        
        # Call callback if provided
        if callback is not None:
            callback({
                'x': x_current,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })
        
        # JKASS step - optimized interpolation
        if sigma_next == 0:
            # Final denoising step - just use denoised
            x_current = denoised
        else:
            # Optimized interpolation with sigma scaling
            sigma_value = sigma.item()
            sigma_next_value = sigma_next.item()
            
            if sigma_value > 1e-6:
                # Calculate noise prediction
                noise_pred = (x_current - denoised) / sigma
                
                # Calculate sigma ratio for adaptive blending
                sigma_ratio = sigma_next_value / sigma_value
                
                # Base interpolation
                x_next = denoised + noise_pred * sigma_next
                
                # Adaptive refinement based on sigma transition
                # Smoother transitions when sigma drops significantly
                if sigma_ratio < 0.75 and sigma_next_value > 0.12:
                    # Blend more towards denoised for very large sigma drops
                    blend_factor = 0.12 * (1.0 - sigma_ratio)
                    x_current = x_next * (1.0 - blend_factor) + denoised * blend_factor
                else:
                    x_current = x_next
            else:
                # Final step - use denoised
                x_current = denoised
    
    return x_current

