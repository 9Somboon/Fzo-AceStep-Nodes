import json
import re
import urllib.error
import urllib.request
import random


class AceStepGroqLyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Music style or style prompt (e.g., Synthwave with female vocals)",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "password": True,
                        "placeholder": "Groq API Key",
                    },
                ),
                "model": (
                    [
                        "llama-3.3-70b-versatile",      # Meta Llama 3.3 70B - Melhor qualidade
                        "llama-3.1-8b-instant",         # Meta Llama 3.1 8B - Mais rápido e leve
                        "openai/gpt-oss-120b",          # OpenAI GPT-OSS 120B - Muito poderoso
                        "openai/gpt-oss-20b",           # OpenAI GPT-OSS 20B - Bom balanceamento
                        "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout (preview)
                        "meta-llama/llama-4-maverick-17b-128e-instruct",  # Llama 4 Maverick (preview)
                    ],
                    {
                        "default": "llama-3.3-70b-versatile",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 8192,
                        "step": 128,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Groq"

    def _build_prompt(self, style: str, seed: int) -> str:
        base_style = style.strip() or "Generic song"
        allowed_tags = (
            "Use ONLY these section tags in square brackets (no numbers): [Intro], [Verse], [Pre-Chorus], [Chorus], "
            "[Post-Chorus], [Bridge], [Breakdown], [Drop], [Hook], [Refrain], [Instrumental], [Solo], [Rap], [Outro]. "
            "Do NOT add numbers to tags (e.g., use [Verse], not [Verse 1])."
        )
        instructions = (
            "You are a music lyricist. Generate song lyrics in the requested style. "
            "Return ONLY the lyrics as plain text. Do not add titles, prefaces, markdown, code fences, or quotes. "
            f"{allowed_tags} Never use parentheses for section labels. "
            "Keep it concise and coherent."
        )
        # Include seed in prompt to make each generation unique for ComfyUI
        return f"Style: {base_style}. {instructions} [Generation seed: {seed}]"

    def _call_groq(self, api_key: str, model: str, prompt: str, max_tokens: int = 1024):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.9,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return "", f"[Groq] HTTPError: {e.code} {error_detail}"
        except urllib.error.URLError as e:  # pragma: no cover
            return "", f"[Groq] URLError: {e.reason}"
        except Exception as e:  # pragma: no cover
            return "", f"[Groq] Error: {e}"

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return "", "[Groq] Failed to parse response JSON."

        text = ""
        if isinstance(parsed, dict):
            choices = parsed.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "").strip()
        
        text = self._clean_markdown(text)
        if not text:
            text = "[Groq] Empty response."
        return text, ""

    def _clean_markdown(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        # Normalize section labels: convert leading (Verse) style to [Verse] and strip numbers
        normalized_lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 48:
                inner = stripped[1:-1].strip()
                if inner:
                    # drop trailing numbers inside parentheses, e.g., (Verse 1) -> [Verse]
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            # If the label is in brackets but uses lowercase or extra spaces, clean it up
            if stripped.startswith("[") and stripped.endswith("]") and len(stripped) <= 64:
                inner = stripped[1:-1].strip()
                if inner:
                    # drop trailing numbers from tags like "Verse 1" -> "Verse"
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int, control_before_generate=None):
        api_key = api_key.strip()
        if not api_key:
            return {"ui": {"text": ["[Groq] API key is missing."]}, "result": ("[Groq] API key is missing.",)}

        prompt = self._build_prompt(style, seed)
        lyrics, error = self._call_groq(api_key=api_key, model=model, prompt=prompt, max_tokens=max_tokens)
        output_text = error or lyrics
        
        return {
            "ui": {"text": [output_text]}, 
            "result": (output_text,)
        }


NODE_CLASS_MAPPINGS = {
    "AceStepGroqLyrics": AceStepGroqLyrics,
}

NODE_DISPLAY_NAMES = {
    "AceStepGroqLyrics": "Ace-Step Groq Lyrics",
}
