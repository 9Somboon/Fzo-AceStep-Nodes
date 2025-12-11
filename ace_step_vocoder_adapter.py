"""Vocoder Adapter Node

This node is a minimal adapter to help feed Ace-Step latents or decoded audio into a vocoder.
It tries to detect the expected input type of the provided vocoder object and calls the right API.
Supported flows:
- Latent -> VAE -> waveform -> mel -> vocoder
- Latent -> mel (if latent appears to be mel) -> vocoder
- Clean waveform -> vocoder (if vocoder expects waveform for final polish)

Notes:
- Vocoder objects must be Python objects exposed to ComfyUI nodes (i.e., selected via a model node).
- The node supports `mel_transform` using `librosa` if present, otherwise uses Torch-based mel filter.
"""
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class AceStepVocoderAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocoder": ("MODEL",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100}),
                "n_mels": ("INT", {"default": 128}),
                "n_fft": ("INT", {"default": 2048}),
                "hop_length": ("INT", {"default": 512}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "adapt"
    CATEGORY = "JK AceStep Nodes/Vocoder"

    def _to_mel_torch(self, waveform, sr=44100, n_fft=2048, hop=512, n_mels=128):
        # waveform: [B, C, T] or [T]
        import torch.nn.functional as F
        if waveform.dim() == 3:
            wav = waveform[:, 0]
        elif waveform.dim() == 2:
            wav = waveform[:, 0]
        else:
            wav = waveform.unsqueeze(0)
        try:
            import torchaudio
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)(wav)
            log_mel = torch.log(torch.clamp(mel_spec, 1e-9))
            return log_mel
        except Exception:
            # Lowest-effort fallback: compute STFT magnitude and map bins
            stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, return_complex=True)
            mag = torch.abs(stft)
            # naive spectral-to-mel via linear downsampling
            mel = F.interpolate(mag.unsqueeze(1), size=n_mels, mode='linear').squeeze(1)
            return torch.log(torch.clamp(mel, 1e-9))

    def adapt(self, vocoder, vae, latent, sample_rate=44100, n_mels=None, n_fft=2048, hop_length=512):
        # optional parameters set by node UI can be passed through **kwargs later
        # Try to introspect vocoder for n_mels/hop_length/etc
        if n_mels is None:
            n_mels = getattr(vocoder, 'n_mels', None)
        if n_mels is None and hasattr(vocoder, 'config') and getattr(vocoder.config, 'n_mels', None) is not None:
            n_mels = vocoder.config.n_mels
        if n_mels is None:
            n_mels = 128
        n_fft = getattr(vocoder, 'n_fft', n_fft)
        hop = getattr(vocoder, 'hop_length', hop_length)
        # Step 1: If 'latent' is a dict and contains 'samples', try decode
        audio_wave = None
        if isinstance(latent, dict) and 'samples' in latent:
            try:
                audio_wave = vae.decode(latent['samples']).movedim(-1, 1)
            except Exception as e:
                logger.warning(f"VAE decode failed: {e}")
                audio_wave = None

        # Step 1b: If the latent seems to be mel already (many vocoders expect mel)
        latent_is_mel = False
        if isinstance(latent, dict) and 'samples' in latent:
            samples = latent['samples']
            # Heuristic: if last frequency dim is <= n_mels and reasonably small, it's probably a mel
            if samples.dim() == 4 and samples.shape[-1] <= max(128, n_mels):
                latent_is_mel = True
            elif samples.dim() == 3 and samples.shape[1] == n_mels:
                latent_is_mel = True

        # Step 2: Form mel if needed
        mel = None
        if latent_is_mel:
            samples = latent['samples']
            if samples.dim() == 4:
                # [B, C, T, F] -> collapse C by mean -> [B, T, F], then permute to [B, F, T]
                mel = samples.mean(dim=1).permute(0, 2, 1)
                # interpolate frequencies to n_mels if different
                if mel.shape[1] != n_mels:
                    mel = torch.nn.functional.interpolate(mel.unsqueeze(1), size=(n_mels, mel.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
            elif samples.dim() == 3:
                # [B, C, T] - assume channel dim is n_mels
                if samples.shape[1] != n_mels:
                    mel = torch.nn.functional.interpolate(samples.unsqueeze(1), size=(n_mels, samples.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
                else:
                    mel = samples
            else:
                mel = samples
        elif audio_wave is not None:
            # if needed, resample audio to vocoder sampling rate
            vocoder_sr = getattr(vocoder, 'sampling_rate', getattr(vocoder, 'sample_rate', sample_rate))
            if audio_wave is not None and hasattr(audio_wave, 'shape') and int(vocoder_sr) != int(sample_rate):
                try:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=int(vocoder_sr))
                    audio_wave = resampler(audio_wave)
                    sample_rate = int(vocoder_sr)
                except Exception:
                    # fallback: log warning and continue
                    logger.warning('Resample failed: torchaudio not available or error during resample')
            mel = self._to_mel_torch(audio_wave, sr=sample_rate, n_fft=n_fft, hop=hop, n_mels=n_mels)
        else:
            # last resort: try using the latent values collapsed
            try:
                s = latent['samples']
                mel = s.mean(dim=1) if s.dim() >= 4 else s
            except Exception as e:
                logger.error(f"Failed to derive mel: {e}")
                raise RuntimeError("Unable to derive mel from latent")

        # Step 3: Try call the vocoder
        if mel is None:
            raise RuntimeError('Failed to produce mel for vocoder input')

        # Many vocoders accept (batch, n_mels, T) or (n_mels, T)
        if mel.dim() == 2:
            batched = mel.unsqueeze(0)
        else:
            batched = mel

        # Try common method names
        # Several vocoders expect inputs with log mel spec shape [B, n_mels, T]
        # If mel is not yet log-scaled, apply log
        try:
            if batched.min() >= 0:
                batched = torch.log(torch.clamp(batched, min=1e-9))
        except Exception:
            pass

        if hasattr(vocoder, 'infer'):
            try:
                out = vocoder.infer(batched)
                return ({'waveform': out, 'sample_rate': sample_rate},)
            except Exception as e:
                logger.warning(f"vocoder.infer() failed: {e}")
        if hasattr(vocoder, 'synthesize'):
            try:
                out = vocoder.synthesize(batched)
                return ({'waveform': out, 'sample_rate': sample_rate},)
            except Exception as e:
                logger.warning(f"vocoder.synthesize() failed: {e}")
        if hasattr(vocoder, 'decode'):
            try:
                out = vocoder.decode(batched)
                return ({'waveform': out, 'sample_rate': sample_rate},)
            except Exception as e:
                logger.warning(f"vocoder.decode() failed: {e}")

        # If vocoder is a function, call it directly
        if callable(vocoder):
            try:
                out = vocoder(batched)
                return ({'waveform': out, 'sample_rate': sample_rate},)
            except Exception as e:
                logger.warning(f"vocoder callable failed: {e}")

        logger.error('No known vocoder API found. Please use a vocoder object exposing infer/synthesize/decode or pass a callable.')
        raise RuntimeError('Unsupported vocoder object')


NODE_CLASS_MAPPINGS = {
    'AceStepVocoderAdapter': AceStepVocoderAdapter,
}

NODE_DISPLAY_NAMES = {
    'AceStepVocoderAdapter': 'Ace-Step Vocoder Adapter',
}
