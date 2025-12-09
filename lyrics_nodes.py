# AceStep Lyrics Generator Nodes for ComfyUI
# Multi-API support for text/lyrics generation
# Supports: OpenAI, Anthropic Claude, Perplexity, Cohere, Replicate, HuggingFace, Together AI, Fireworks AI, Google Gemini, Groq

import urllib.request
import urllib.error
import json
import re
import os

try:
    from comfy.model_management import get_torch_device
    from folder_paths import get_output_directory
    folder_paths = type('obj', (object,), {'get_output_directory': get_output_directory})()
except ImportError:
    folder_paths = None

try:
    import folder_paths as folder_paths_module
except ImportError:  # pragma: no cover
    folder_paths_module = None


# ==================== GEMINI LYRICS ====================
class AceStepGeminiLyrics:
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
                        "placeholder": "Gemini API Key",
                    },
                ),
                "model": (
                    [
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-latest",
                        "gemini-2.5-flash-lite",
                        "gemini-2.5-flash-lite-latest",
                        "gemini-2.5-pro",
                        "gemini-2.5-pro-latest",
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-lite",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
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
    CATEGORY = "JK AceStep Nodes/Lyrics"

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
        return f"Style: {base_style}. {instructions} [Generation seed: {seed}]"

    def _call_gemini(self, api_key: str, model: str, prompt: str, max_tokens: int = 1024):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return "", f"[Gemini] HTTPError: {e.code} {error_detail}"
        except urllib.error.URLError as e:
            return "", f"[Gemini] URLError: {e.reason}"
        except Exception as e:
            return "", f"[Gemini] Error: {e}"

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return "", "[Gemini] Failed to parse response JSON."

        text = ""
        if isinstance(parsed, dict):
            candidates = parsed.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
            if not text:
                text = parsed.get("text", "").strip()
        text = self._clean_markdown(text)
        if not text:
            text = "[Gemini] Empty response."
        return text, ""

    def _clean_markdown(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        normalized_lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 48:
                inner = stripped[1:-1].strip()
                if inner:
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            if stripped.startswith("[") and stripped.endswith("]") and len(stripped) <= 64:
                inner = stripped[1:-1].strip()
                if inner:
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int, control_before_generate=None):
        api_key = api_key.strip()
        if not api_key:
            return ("Error: API key is missing.",)

        prompt = self._build_prompt(style, seed)
        lyrics, error = self._call_gemini(api_key=api_key, model=model, prompt=prompt, max_tokens=max_tokens)
        output_text = error or lyrics
        
        return (output_text,)


# ==================== GROQ LYRICS ====================
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
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",
                        "llama-3.2-1b-preview",
                        "llama-3.2-3b-preview",
                        "openai/gpt-oss-120b",
                        "openai/gpt-oss-20b",
                        "meta-llama/llama-guard-3-8b",
                        "meta-llama/llama-4-scout-17b-16e-instruct",
                        "meta-llama/llama-4-maverick-17b-128e-instruct",
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
    CATEGORY = "JK AceStep Nodes/Lyrics"

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
        except urllib.error.URLError as e:
            return "", f"[Groq] URLError: {e.reason}"
        except Exception as e:
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
        normalized_lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 48:
                inner = stripped[1:-1].strip()
                if inner:
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            if stripped.startswith("[") and stripped.endswith("]") and len(stripped) <= 64:
                inner = stripped[1:-1].strip()
                if inner:
                    parts = inner.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        inner = " ".join(parts[:-1])
                    line = f"[{inner}]"
            normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int, control_before_generate=None):
        api_key = api_key.strip()
        if not api_key:
            return ("Error: API key is missing.",)

        prompt = self._build_prompt(style, seed)
        lyrics, error = self._call_groq(api_key=api_key, model=model, prompt=prompt, max_tokens=max_tokens)
        output_text = error or lyrics
        
        return (output_text,)


# ==================== OTHER LYRICS GENERATORS ====================
class AceStepOpenAI_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "OpenAI API Key"}),
                "model": ([
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4-turbo-preview",
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k"
                ], {"default": "gpt-4o"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }).encode()

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["choices"][0]["message"]["content"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepClaude_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Anthropic API Key"}),
                "model": ([
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20250219",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-2.1"
                ], {"default": "claude-3-5-sonnet-20241022"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["content"][0]["text"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepPerplexity_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Perplexity API Key"}),
                "model": ([
                    "sonar",
                    "sonar-pro",
                    "sonar-reasoning",
                    "sonar-reasoning-pro"
                ], {"default": "sonar"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }).encode()

            req = urllib.request.Request(
                "https://api.perplexity.ai/chat/completions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["choices"][0]["message"]["content"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepCohere_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Cohere API Key"}),
                "model": ([
                    "command-r-plus-08-2024",
                    "command-r-08-2024",
                    "command-r7b-12-2024",
                    "command-r-plus",
                    "command-r",
                    "command"
                ], {"default": "command-r-plus-08-2024"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }).encode()

            req = urllib.request.Request(
                "https://api.cohere.ai/v1/generate",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["generations"][0]["text"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepReplicate_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Replicate API Key"}),
                "model": ([
                    "meta/llama-3.1-405b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "meta/llama-3.1-8b-instruct",
                    "meta/llama-3-70b-instruct",
                    "meta/llama-2-70b-chat",
                    "mistralai/mistral-7b-instruct-v0.3",
                    "mistralai/mistral-small-24b-instruct-2501",
                    "mistralai/mixtral-8x7b-instruct-v0.1"
                ], {"default": "meta/llama-3.1-70b-instruct"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "version": model,
                "input": {"prompt": prompt, "max_length": max_tokens},
            }).encode()

            req = urllib.request.Request(
                "https://api.replicate.com/v1/predictions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                # Replicate returns status URL, need to poll
                return (result.get("status", "Processing..."),)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepHuggingFace_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "HuggingFace API Key"}),
                "model": ([
                    "meta-llama/Llama-3.1-405B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "deepseek-ai/deepseek-v3",
                    "qwen/Qwen2.5-72B-Instruct",
                    "HuggingFaceH4/zephyr-7b-beta",
                    "tiiuae/falcon-7b-instruct"
                ], {"default": "meta-llama/Llama-3.1-70B-Instruct"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens},
            }).encode()

            req = urllib.request.Request(
                f"https://api-inference.huggingface.co/models/{model}",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result[0].get("generated_text", "")
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepTogetherAI_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Together AI API Key"}),
                "model": ([
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "meta-llama/Llama-3.1-405B-Instruct-Turbo",
                    "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    "mistralai/Mistral-Small-24B-Instruct-2501",
                    "mistralai/Ministral-3-8B-Instruct-2512",
                    "deepseek-ai/DeepSeek-V3.1",
                    "deepseek-ai/DeepSeek-R1",
                    "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "moonshotai/Kimi-K2-Instruct-0905",
                    "google/gemma-3-27b-it"
                ], {"default": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }).encode()

            req = urllib.request.Request(
                "https://api.together.xyz/v1/chat/completions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["choices"][0]["message"]["content"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepFireworks_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Fireworks API Key"}),
                "model": ([
                    "deepseek-ai/deepseek-v3p2",
                    "deepseek-ai/deepseek-r1",
                    "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "Qwen/Qwen3-Next-80B-A3B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct-Turbo",
                    "meta-llama/Llama-3.3-70B-Instruct",
                    "meta-llama/Llama-3.1-405B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "mistralai/Mistral-Large-3-675B-Instruct-2512",
                    "mistralai/Mistral-Small-24B-Instruct-2501",
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "mistralai/Mixtral-8x22B-Instruct",
                    "zai-org/GLM-4.6"
                ], {"default": "meta-llama/Llama-3.3-70B-Instruct"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = f"{text}\n\nGenerate song lyrics for this prompt:"
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }).encode()

            req = urllib.request.Request(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["choices"][0]["message"]["content"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepGemini_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Google Gemini API Key"}),
                "model": (["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"], {"default": "gemini-2.5-flash"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def _build_prompt(self, text, seed):
        return f"{text}\n\nGenerate song lyrics for this prompt.\n[Seed: {seed}]"

    def generate(self, text, api_key, model, max_tokens, seed):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = self._build_prompt(text, seed)
        
        try:
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
            }).encode()

            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["candidates"][0]["content"]["parts"][0]["text"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class AceStepGroq_Lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Prompt for lyrics generation"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Groq API Key"}),
                "model": (["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-4-9b"], {"default": "llama-3.3-70b-versatile"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def _build_prompt(self, text, seed):
        return f"{text}\n\nGenerate song lyrics for this prompt.\n[Seed: {seed}]"

    def generate(self, text, api_key, model, max_tokens, seed):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = self._build_prompt(text, seed)
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }).encode()

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                lyrics = result["choices"][0]["message"]["content"]
                return (lyrics,)
        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "AceStepGemini_Lyrics": AceStepGeminiLyrics,
    "AceStepGroq_Lyrics": AceStepGroqLyrics,
    "AceStepOpenAI_Lyrics": AceStepOpenAI_Lyrics,
    "AceStepClaude_Lyrics": AceStepClaude_Lyrics,
    "AceStepPerplexity_Lyrics": AceStepPerplexity_Lyrics,
    "AceStepCohere_Lyrics": AceStepCohere_Lyrics,
    "AceStepReplicate_Lyrics": AceStepReplicate_Lyrics,
    "AceStepHuggingFace_Lyrics": AceStepHuggingFace_Lyrics,
    "AceStepTogetherAI_Lyrics": AceStepTogetherAI_Lyrics,
    "AceStepFireworks_Lyrics": AceStepFireworks_Lyrics,
}

NODE_DISPLAY_NAMES = {
    "AceStepGemini_Lyrics": "Ace-Step Gemini Lyrics",
    "AceStepGroq_Lyrics": "Ace-Step Groq Lyrics",
    "AceStepOpenAI_Lyrics": "Ace-Step OpenAI Lyrics",
    "AceStepClaude_Lyrics": "Ace-Step Claude Lyrics",
    "AceStepPerplexity_Lyrics": "Ace-Step Perplexity Lyrics",
    "AceStepCohere_Lyrics": "Ace-Step Cohere Lyrics",
    "AceStepReplicate_Lyrics": "Ace-Step Replicate Lyrics",
    "AceStepHuggingFace_Lyrics": "Ace-Step HuggingFace Lyrics",
    "AceStepTogetherAI_Lyrics": "Ace-Step Together AI Lyrics",
    "AceStepFireworks_Lyrics": "Ace-Step Fireworks Lyrics",
}
