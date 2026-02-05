import json
import re
import urllib.error
import urllib.request
import random


class AceStepLMStudioLyrics:
    """Node for generating lyrics using local LLM models via LM Studio (OpenAI-compatible API)."""
    
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
                "api_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234",
                        "multiline": False,
                        "placeholder": "LM Studio API URL (e.g., http://localhost:1234 or http://192.168.1.100:1234)",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Model name (leave empty to use server default, or paste model name from LM Studio)",
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
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
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
    CATEGORY = "JK AceStep Nodes/LMStudio"

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

    def _call_lmstudio(self, api_url: str, model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.9):
        # Normalize URL - remove trailing slash
        api_url = api_url.rstrip("/")
        url = f"{api_url}/v1/chat/completions"
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False,
        }
        
        # Only include model if specified (LM Studio will use the loaded model otherwise)
        if model.strip():
            payload["model"] = model.strip()
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={
                "Content-Type": "application/json",
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # Longer timeout for local models
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return "", f"[LMStudio] HTTPError: {e.code} {error_detail}"
        except urllib.error.URLError as e:  # pragma: no cover
            return "", f"[LMStudio] URLError: {e.reason} - Make sure LM Studio is running and the server is enabled"
        except Exception as e:  # pragma: no cover
            return "", f"[LMStudio] Error: {e}"

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return "", "[LMStudio] Failed to parse response JSON."

        text = ""
        if isinstance(parsed, dict):
            choices = parsed.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "").strip()
        
        text = self._clean_markdown(text)
        if not text:
            text = "[LMStudio] Empty response."
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

    def generate(self, style: str, api_url: str, model: str, max_tokens: int, temperature: float, seed: int, control_before_generate=None):
        api_url = api_url.strip()
        if not api_url:
            return {"ui": {"text": ["[LMStudio] API URL is missing."]}, "result": ("[LMStudio] API URL is missing.",)}

        prompt = self._build_prompt(style, seed)
        lyrics, error = self._call_lmstudio(api_url=api_url, model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        output_text = error or lyrics
        
        return {
            "ui": {"text": [output_text]}, 
            "result": (output_text,)
        }


class AceStepLMStudioFetchModels:
    """Utility node to fetch and display available models from LM Studio server."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234",
                        "multiline": False,
                        "placeholder": "LM Studio API URL",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "fetch_models"
    CATEGORY = "JK AceStep Nodes/LMStudio"

    def fetch_models(self, api_url: str):
        api_url = api_url.strip().rstrip("/")
        url = f"{api_url}/v1/models"
        
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return {"ui": {"text": [f"[LMStudio] HTTPError: {e.code} {error_detail}"]}, "result": (f"Error: {e.code}",)}
        except urllib.error.URLError as e:
            return {"ui": {"text": [f"[LMStudio] URLError: {e.reason}"]}, "result": (f"Connection failed: {e.reason}",)}
        except Exception as e:
            return {"ui": {"text": [f"[LMStudio] Error: {e}"]}, "result": (f"Error: {e}",)}

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return {"ui": {"text": ["[LMStudio] Failed to parse response."]}, "result": ("Parse error",)}

        models = []
        if isinstance(parsed, dict) and "data" in parsed:
            for m in parsed["data"]:
                model_id = m.get("id", "unknown")
                models.append(model_id)
        
        if not models:
            result = "No models found. Make sure a model is loaded in LM Studio."
        else:
            result = "Available models:\n" + "\n".join(f"  • {m}" for m in models)
        
        return {"ui": {"text": [result]}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "AceStepLMStudioLyrics": AceStepLMStudioLyrics,
    "AceStepLMStudioFetchModels": AceStepLMStudioFetchModels,
}

NODE_DISPLAY_NAMES = {
    "AceStepLMStudioLyrics": "Ace-Step LM Studio Lyrics",
    "AceStepLMStudioFetchModels": "Ace-Step LM Studio Fetch Models",
}
