"""Converte checkpoints PyTorch de vocoders para formato safetensors + cria um config JSON compatível.

Usage:
  python tools/convert_vocoder_checkpoint.py --input path/to/vocoder.pth --outdir ACE-Step-v1-3.5B/music_vocoder --name my_vocoder

Suporta checkpoints PyTorch (.pt/.pth) e safetensors (pass-through).
Gera 'diffusion_pytorch_model.safetensors' e 'config_default.json' se não houver.
"""
import argparse
import os
import json
import torch
from safetensors.torch import save_file

def load_state_dict(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    else:
        sd = torch.load(path, map_location='cpu')
        # If file is a model object, try to get .state_dict()
        if not isinstance(sd, dict):
            if hasattr(sd, 'state_dict'):
                sd = sd.state_dict()
            elif hasattr(sd, 'model') and hasattr(sd['model'], 'state_dict'):
                sd = sd['model'].state_dict()
        return sd

def save_safetensors(sd, out_path):
    # sd: dict[str -> torch.Tensor]
    tensors = {}
    for k, v in sd.items():
        if hasattr(v, 'numpy') or hasattr(v, 'dtype'):
            # ensure it's a tensor
            tensors[k] = v.cpu()
        else:
            try:
                tensors[k] = torch.as_tensor(v)
            except Exception:
                print(f"Warning: skipping non-tensor key {k}")
    save_file(tensors, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', default='diffusion_pytorch_model.safetensors')
    ap.add_argument('--sample_rate', type=int, default=44100)
    ap.add_argument('--n_mels', type=int, default=128)
    ap.add_argument('--n_fft', type=int, default=2048)
    ap.add_argument('--hop_length', type=int, default=512)
    args = ap.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    print('Loading checkpoint:', args.input)
    sd = load_state_dict(args.input)
    out_path = os.path.join(args.outdir, args.name)
    if args.input.endswith('.safetensors'):
        print('Input already safetensors, copying...')
        import shutil
        shutil.copy2(args.input, out_path)
    else:
        print('Saving safetensors to:', out_path)
        save_safetensors(sd, out_path)

    # Write config_default.json if not present
    cfg_path = os.path.join(args.outdir, 'config_default.json')
    if not os.path.exists(cfg_path):
        cfg = {
            "_class_name": "ADaMoSHiFiGANV1",
            "_diffusers_version":"0.32.2",
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "sampling_rate": args.sample_rate,
            "input_channels": args.n_mels,
            "f_min": 40,
            "f_max": 16000
        }
        print('Writing config_default.json with configuration for n_mels:', args.n_mels)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

    print('Done')

if __name__ == '__main__':
    main()
