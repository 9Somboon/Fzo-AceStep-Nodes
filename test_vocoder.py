#!/usr/bin/env python3
"""
Test script to verify vocoder loading and basic functionality
"""

import os
import sys
import torch

# Add parent directories to path
my_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_dir)

# Import vocoder functions from ace_step_ksampler
from ace_step_ksampler import load_vocoder_model, apply_vocoder_to_audio

def test_vocoder_loading():
    """Test if vocoder model loads correctly"""
    print("=" * 60)
    print("Testing Vocoder Loading")
    print("=" * 60)
    
    vocoder_data = load_vocoder_model()
    
    if vocoder_data is None:
        print("❌ FAILED: Vocoder model did not load")
        return False
    
    print("✓ Vocoder model loaded successfully")
    
    # Check contents
    if 'state_dict' in vocoder_data:
        state_dict = vocoder_data['state_dict']
        print(f"✓ State dict loaded with {len(state_dict)} parameters")
        
        # Print first few parameter names
        param_names = list(state_dict.keys())[:5]
        print(f"  Sample parameters: {param_names}")
    
    if 'config' in vocoder_data:
        config = vocoder_data['config']
        print(f"✓ Config loaded")
        if '_class_name' in config:
            print(f"  Model class: {config['_class_name']}")
    
    return True

def test_vocoder_application():
    """Test if vocoder can be applied to audio"""
    print("\n" + "=" * 60)
    print("Testing Vocoder Application")
    print("=" * 60)
    
    # Load vocoder
    vocoder_data = load_vocoder_model()
    if vocoder_data is None:
        print("❌ FAILED: Could not load vocoder model")
        return False
    
    # Create dummy audio tensor
    # Shape: [batch=1, channels=1, samples=44100]
    dummy_audio = torch.randn(1, 1, 44100)
    print(f"✓ Created dummy audio tensor: {dummy_audio.shape}")
    
    # Apply vocoder
    try:
        output_audio = apply_vocoder_to_audio(dummy_audio, vocoder_data)
        print(f"✓ Vocoder applied successfully")
        print(f"  Output shape: {output_audio.shape}")
        print(f"  Output dtype: {output_audio.dtype}")
        print(f"  Output device: {output_audio.device}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Vocoder application error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "VOCODER TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Test 1: Loading
    try:
        result = test_vocoder_loading()
        results.append(("Vocoder Loading", result))
    except Exception as e:
        print(f"❌ EXCEPTION in vocoder loading test: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Vocoder Loading", False))
    
    # Test 2: Application
    try:
        result = test_vocoder_application()
        results.append(("Vocoder Application", result))
    except Exception as e:
        print(f"❌ EXCEPTION in vocoder application test: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Vocoder Application", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<45} {status}")
    
    all_passed = all(result for _, result in results)
    print()
    if all_passed:
        print("✓ All tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
