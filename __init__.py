# Ace-Step KSampler Nodes for ComfyUI
# Custom nodes specialized for Ace-Step audio generation

from .ace_step_ksampler import NODE_CLASS_MAPPINGS as KSAMPLER_MAPPINGS, NODE_DISPLAY_NAMES as KSAMPLER_NAMES
from .ace_step_prompt_gen import NODE_CLASS_MAPPINGS as PROMPT_MAPPINGS, NODE_DISPLAY_NAMES as PROMPT_NAMES
from .gemini_nodes import NODE_CLASS_MAPPINGS as GEMINI_MAPPINGS, NODE_DISPLAY_NAMES as GEMINI_NAMES

# Combine all node mappings
NODE_CLASS_MAPPINGS = {**KSAMPLER_MAPPINGS, **PROMPT_MAPPINGS, **GEMINI_MAPPINGS}
NODE_DISPLAY_NAMES = {**KSAMPLER_NAMES, **PROMPT_NAMES, **GEMINI_NAMES}

# Register custom samplers with ComfyUI
def add_samplers():
    """Register custom samplers with ComfyUI KSampler."""
    try:
        from comfy.samplers import KSampler, k_diffusion_sampling
        from .py.jkass_sampler import sample_jkass
        
        # Register Samplers
        for sampler_name, sampler_func in [
            ("jkass", sample_jkass),
        ]:
            if sampler_name not in KSampler.SAMPLERS:
                try:
                    # Find uni_pc_bh2 and insert after it
                    idx = KSampler.SAMPLERS.index("uni_pc_bh2")
                    KSampler.SAMPLERS.insert(idx + 1, sampler_name)
                except (ValueError, IndexError):
                    # Fallback: just append
                    KSampler.SAMPLERS.append(sampler_name)
                
                # Register the sampling function
                setattr(k_diffusion_sampling, f"sample_{sampler_name}", sampler_func)
            
    except ImportError as e:
        print(f"[ACE-STEP] Warning: Could not import ComfyUI modules: {e}")
    except Exception as e:
        print(f"[ACE-STEP] Error registering samplers: {e}")

# Call this at module load time
add_samplers()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAMES"]
