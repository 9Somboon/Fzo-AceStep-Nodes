"""Simple inspector for safetensors checkpoints.

Use this script to list keys and shape info for safetensors files.
It helps to detect whether a checkpoint is a) VAE/decoder b) main UNet c) vocoder, etc.

Run:
  python tools/inspect_checkpoint.py path/to/checkpoint.safetensors

If `safetensors` isn't installed, you can run with a small fallback to attempt to open the file bytes.
"""
import sys
import os
import argparse

def inspect_with_safetensors(path):
    try:
        from safetensors.torch import load_file
        import torch
        d = load_file(path)
        print(f"Found {len(d)} tensors in {path}")
        # Print first 30 keys + shape
        for k, v in list(d.items())[:30]:
            if hasattr(v, 'shape'):
                shape = tuple(v.shape)
            else:
                shape = 'unknown'
            print(f"  {k}: {shape}")
    except Exception as e:
        print('safetensors load failed:', e)
        print('Try installing via pip: pip install safetensors')

def inspect_bytes(path):
    try:
        with open(path, 'rb') as f:
            header = f.read(512)
        print('Read 512 bytes header (hex):', header[:128].hex())
        # Look for ascii hints
        try:
            ascii_text = header.decode('latin1')
            print('Header as text (latin1 snippet):', ''.join([c if 32<=ord(c)<127 else '.' for c in ascii_text[:200]]))
        except Exception:
            pass
    except Exception as e:
        print('Read failed:', e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='Path to safetensors file to inspect')
    args = ap.parse_args()
    path = args.path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        print('File not found:', path)
        sys.exit(2)

    print('Inspecting:', path)
    inspect_with_safetensors(path)
    print('\nFallback raw header:')
    inspect_bytes(path)

if __name__ == '__main__':
    main()
