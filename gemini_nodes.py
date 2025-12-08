import json
import os
import re
import urllib.error
import urllib.request

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None


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
                        "gemini-2.5-flash",           # current main flash (low cost/free tier eligible)
                        "gemini-2.5-flash-latest",
                        "gemini-2.5-flash-lite",      # fastest/cheapest
                        "gemini-2.5-flash-lite-latest",
                        "gemini-2.5-pro",             # higher quality, paid
                        "gemini-3-pro-preview",       # newest preview, may require billing
                        "gemini-2.0-flash",           # older stable
                        "gemini-2.0-flash-lite",      # older lite
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Gemini"

    def _build_prompt(self, style: str) -> str:
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
        return f"Style: {base_style}. {instructions}"

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
        except urllib.error.URLError as e:  # pragma: no cover
            return "", f"[Gemini] URLError: {e.reason}"
        except Exception as e:  # pragma: no cover
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

    def generate(self, style: str, api_key: str, model: str, max_tokens: int):
        api_key = api_key.strip()
        if not api_key:
            return {"ui": {"text": ["[Gemini] API key is missing."]}, "result": ("[Gemini] API key is missing.",)}

        prompt = self._build_prompt(style)
        lyrics, error = self._call_gemini(api_key=api_key, model=model, prompt=prompt, max_tokens=max_tokens)
        output_text = error or lyrics
        return {"ui": {"text": [output_text]}, "result": (output_text,)}


class AceStepSaveText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Text content to save",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "text/lyrics",
                        "placeholder": "folder/path/filename (e.g., text/lyrics or lyrics/15/music)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "JK AceStep Nodes/IO"

    def _sanitize_prefix(self, prefix: str) -> str:
        # Sanitize folder path and filename: clean up invalid chars but preserve slashes
        # Split into parts, sanitize each, then rejoin
        parts = prefix.split("/")
        sanitized_parts = []
        for part in parts:
            cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", part).strip("._-")
            if cleaned:
                sanitized_parts.append(cleaned)
        return "/".join(sanitized_parts) if sanitized_parts else "text/default"

    def _next_available_path(self, base_output: str, prefix: str):
        # Parse prefix into directory path and filename
        # e.g., "text/lyrics" -> (text, lyrics), "lyrics/15/music" -> (lyrics/15, music)
        prefix_parts = prefix.split("/")
        if len(prefix_parts) < 2:
            # If only one part, assume it's filename in default text folder
            folder_path = os.path.join(base_output, "text")
            filename_base = prefix_parts[0] or "default"
        else:
            # Last part is filename, everything else is the folder path
            filename_base = prefix_parts[-1]
            folder_rel = "/".join(prefix_parts[:-1])
            folder_path = os.path.join(base_output, folder_rel)
        
        os.makedirs(folder_path, exist_ok=True)
        
        index = 1
        while True:
            suffix = "" if index == 1 else str(index)
            filename = f"{filename_base}{suffix}.txt"
            candidate = os.path.join(folder_path, filename)
            if not os.path.exists(candidate):
                return candidate
            index += 1


    def save(self, text: str, filename_prefix: str = "text/lyrics"):
        base_output = folder_paths.get_output_directory() if folder_paths else os.path.join(os.getcwd(), "output")
        
        # Sanitize the path
        prefix = self._sanitize_prefix(filename_prefix)
        
        # Get path and increment suffix if file exists
        target_path = self._next_available_path(base_output, prefix)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)

        return {"ui": {"text": [target_path]}, "result": (target_path,) }


NODE_CLASS_MAPPINGS = {
    "AceStepGeminiLyrics": AceStepGeminiLyrics,
    "AceStepSaveText": AceStepSaveText,
}

NODE_DISPLAY_NAMES = {
    "AceStepGeminiLyrics": "Ace-Step Gemini Lyrics",
    "AceStepSaveText": "Ace-Step Save Text",
}
