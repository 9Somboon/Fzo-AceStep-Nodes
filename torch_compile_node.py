import torch
import comfy.model_management as mm
import os

# Disable torch.compile globally - ACE-Step incompatible
os.environ["TORCH_COMPILE_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

# Disable JIT compilation for ACE-Step compatibility
os.environ["TORCH_JIT_PROFILING"] = "0"

class AceStepTorchCompile:
    """
    DEPRECATED: Torch Compile is not compatible with ACE-Step models.
    
    This node is kept for backward compatibility but does NOT apply compilation.
    It returns the original model unchanged.
    
    torch.compile has known incompatibilities with:
    - ACE-Step's dynamic tensor operations
    - CUDA graphs and memory management
    - Dynamic shape handling
    
    For better performance, consider:
    - Using fewer sampling steps (50-100 instead of 150+)
    - Switching to faster samplers (DPM++, Euler A)
    - Using precision FP16 or BF16
    - Enabling memory-efficient attention
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "passthrough"
    CATEGORY = "JK AceStep Nodes/Optimization"
    
    def passthrough(self, model):
        """
        Returns model unchanged. torch.compile is disabled for ACE-Step.
        
        This is a passthrough node kept for workflow compatibility.
        """
        print("[AceStep Torch Compile] DEPRECATED: torch.compile disabled for ACE-Step compatibility")
        print("[AceStep Torch Compile] Returning original model unchanged")
        print("[AceStep Torch Compile] For better performance:")
        print("[AceStep Torch Compile]   - Reduce sampling steps to 50-100")
        print("[AceStep Torch Compile]   - Use faster samplers (DPM++, Euler A)")
        print("[AceStep Torch Compile]   - Enable FP16/BF16 precision")
        return (model,)
    """
    Torch Compile node for ACE-Step models.
    
    Applies torch.compile() to the model for JIT optimization and faster inference.
    Requires PyTorch 2.0+ with CUDA support for best results.
    
    Compilation modes:
    - default: Balanced speed/compile time
    - reduce-overhead: Minimize Python overhead, good for small models
    - max-autotune: Maximum performance, longer compile time (recommended for audio)
    - max-autotune-no-cudagraphs: Like max-autotune but without CUDA graphs (most stable)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (
                    ["max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead", "default"],
                    {
                        "default": "max-autotune-no-cudagraphs",
                        "tooltip": "max-autotune-no-cudagraphs: Best performance without CUDA graph issues"
                    }
                ),
                "dynamic": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dynamic shapes (may reduce performance but more flexible)"
                    }
                ),
                "fullgraph": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force full graph compilation (may fail on complex models)"
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("compiled_model",)
    FUNCTION = "compile_model"
    CATEGORY = "JK AceStep Nodes/Optimization"
    
    def compile_model(self, model, mode: str, dynamic: bool, fullgraph: bool):
        """
        Apply torch.compile to the model for JIT optimization.
        
        Args:
            model: ComfyUI MODEL object
            mode: Compilation mode
            dynamic: Enable dynamic shapes
            fullgraph: Force full graph compilation
        
        Returns:
            Compiled model wrapped in ComfyUI MODEL format
        """
        # Check PyTorch version
        torch_version = torch.__version__.split('+')[0]
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major < 2:
            print(f"[AceStep Torch Compile] Warning: PyTorch {torch_version} detected. "
                  f"torch.compile requires PyTorch 2.0+. Returning original model.")
            return (model,)
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("[AceStep Torch Compile] Warning: CUDA not available. "
                  "torch.compile works best with CUDA. Compilation may be slower.")
        
        # Clone model to avoid modifying original
        compiled_model = model.clone()
        
        try:
            # Get the actual diffusion model from ComfyUI wrapper
            if hasattr(compiled_model, 'model') and hasattr(compiled_model.model, 'diffusion_model'):
                original_diff_model = compiled_model.model.diffusion_model
                
                print(f"[AceStep Torch Compile] Compiling model with mode='{mode}', "
                      f"dynamic={dynamic}, fullgraph={fullgraph}")
                print("[AceStep Torch Compile] First inference will be slow (compilation phase)...")
                
                # Force disable CUDA graphs globally
                import os
                os.environ["TORCH_COMPILE_DISABLE_CUDAGRAPHS"] = "1"
                os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
                print("[AceStep Torch Compile] CUDA graphs forcefully disabled")
                
                # Configure compilation with explicit options to avoid CUDA graphs
                import torch._inductor.config as inductor_config
                inductor_config.triton.cudagraphs = False
                
                # Apply torch.compile with specified settings
                compiled_diff_model = torch.compile(
                    original_diff_model,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=fullgraph,
                    options={
                        "triton.cudagraphs": False,
                    }
                )
                
                # Replace the diffusion model with compiled version
                compiled_model.model.diffusion_model = compiled_diff_model
                
                print("[AceStep Torch Compile] Model compilation successful!")
                print("[AceStep Torch Compile] Note: First run will trigger compilation and be slower.")
                print("[AceStep Torch Compile] Expect 2-3 minutes for first generation, then much faster.")
                
            else:
                print("[AceStep Torch Compile] Warning: Could not find diffusion_model in model structure. "
                      "Returning original model.")
                return (model,)
                
        except Exception as e:
            print(f"[AceStep Torch Compile] Error during compilation: {e}")
            print("[AceStep Torch Compile] Returning original model.")
            return (model,)
        
        return (compiled_model,)

class AceStepCompileSettings:
    """
    DEPRECATED: Removed due to ACE-Step incompatibility with torch.compile.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "passthrough"
    CATEGORY = "JK AceStep Nodes/Optimization"
    
    def passthrough(self, model):
        print("[AceStep Compile Settings] DEPRECATED: torch.compile disabled")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "AceStepTorchCompile": AceStepTorchCompile,
    "AceStepCompileSettings": AceStepCompileSettings,
}

NODE_DISPLAY_NAMES = {
    "AceStepTorchCompile": "Ace-Step Torch Compile",
    "AceStepCompileSettings": "Ace-Step Compile Settings (Advanced)",
}
