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


# ==================== SHARED PROMPT BUILDER ====================
def get_instrumentality_guidance(instrumentality_balance: str) -> str:
    """
    Get specific guidance for instrumentality balance.
    Instrumentality Balance options:
    - "Pure Instrumental": All instrumental, no vocals
    - "Very Instrumental": Heavy instrumental sections with minimal vocals
    - "Balanced": Mix of instrumental and vocal sections equally
    - "Vocal Focused": More vocal content with supporting instrumental elements
    - "Vocals Only": Pure vocals with no instrumental production tags
    """
    guidance_map = {
        "Pure Instrumental": (
            "INSTRUMENTALITY INSTRUCTIONS:\n"
            "This song should be PURE INSTRUMENTAL with NO VOCALS whatsoever.\n"
            "- Use ONLY instrumental production tags like [synth solo], [drum fill], [bass line], [ambient pad], etc.\n"
            "- DO NOT include any sung lyrics or vocal content\n"
            "- Fill the entire song with creative instrumental elements and effects\n"
            "- Think: Electronic, orchestral, or ambient music compositions\n"
            "- Example structure: [intro] [bass] [synth solo] [drop] [bass] [swell] [drum fill] [outro]\n"
        ),
        "Very Instrumental": (
            "INSTRUMENTALITY INSTRUCTIONS:\n"
            "This song should be VERY INSTRUMENTAL - approximately 75% instrumental, 25% vocal.\n"
            "- Use MANY instrumental production tags (synths, bass, drums, effects)\n"
            "- IMPORTANT: Vocals should appear ONLY in SHORT BURSTS, not continuous sections\n"
            "- SHORT vocal moments: 1-2 lines maximum at a time\n"
            "- LONG instrumental sections: multiple tags with NO vocals between them\n"
            "- Vocals MUST be separated by instrumental sections - never put vocals back-to-back\n"
            "- Example CORRECT structure:\n"
            "  [synth solo]\n"
            "  [bass]\n"
            "  [drop]\n"
            "  One short vocal line here\n"
            "  (blank line)\n"
            "  [drum fill]\n"
            "  [ambient pad]\n"
            "  [swell]\n"
            "  Second vocal line\n"
            "  (blank line)\n"
            "  [filtered saw bass]\n"
            "  [evolving synth leads]\n"
            "- Most of the song should be tags and instrumental moments\n"
            "- Vocals should feel like ACCENTS/HIGHLIGHTS, not the main focus\n"
        ),
        "Balanced": (
            "INSTRUMENTALITY INSTRUCTIONS:\n"
            "This song should BALANCE instrumental and vocal content - approximately 50/50.\n"
            "- Mix vocal sections with instrumental sections equally\n"
            "- Use production tags to separate and define both vocal and instrumental moments\n"
            "- Alternate between singing and instrumental-only sections\n"
            "- Example: [intro] [vocal line] [synth solo] [vocal line] [bass line] [vocal line] [outro]\n"
            "- Keep it dynamic by switching between vocals and pure instrumental parts\n"
        ),
        "Vocal Focused": (
            "INSTRUMENTALITY INSTRUCTIONS:\n"
            "This song should be VOCAL FOCUSED - approximately 75% vocal, 25% instrumental.\n"
            "- Prioritize singing and melodic content\n"
            "- Use instrumental production tags sparingly for emphasis and transitions\n"
            "- Most lines should be sung lyrics, not instrumental tags\n"
            "- Use instrumental tags only at key moments: [intro], [drop], [bridge moment], [outro]\n"
            "- Example: (vocals) (vocals) [swell] (vocals) (vocals) [drum fill] (vocals) (vocals)\n"
        ),
        "Vocals Only": (
            "INSTRUMENTALITY INSTRUCTIONS:\n"
            "This song should be VOCALS ONLY - pure acapella or spoken word style.\n"
            "- DO NOT use any instrumental production tags\n"
            "- Focus entirely on vocal delivery, lyrics, and singing\n"
            "- You may use [acapella] tag if needed, but no other instrumental elements\n"
            "- Think: Vocal arrangements, harmonies, rhythmic spoken parts\n"
            "- This is like acapella or a cappella music\n"
        ),
    }
    
    return guidance_map.get(instrumentality_balance, guidance_map["Balanced"])


def build_acestore_lyrics_prompt(style_or_description: str, seed: int, instrumentality_balance: str = "Balanced") -> str:
    """
    Build creative prompt for Ace-Step music generation with production tags.
    Educates AI to use tags creatively for musical elements with proper formatting.
    Includes instrumentality balance guidance.
    """
    description = (style_or_description or "Generic song").strip()
    
    instrumentality_guidance = get_instrumentality_guidance(instrumentality_balance)
    
    ace_step_guide = (
        "IMPORTANT: You will write song lyrics for the Ace-Step music generation AI. "
        "Ace-Step interprets tags in square brackets [like this] as PRODUCTION INSTRUCTIONS or MUSICAL EVENTS. "
        "Here's how it works:\n\n"
        "1. TEXT WITHOUT TAGS = Sung vocal content (verses, lyrics, melodies)\n"
        "2. TEXT IN [BRACKETS] = Special musical instructions/events that Ace-Step will interpret\n\n"
        "EXAMPLES of creative tags Ace-Step understands:\n"
        "- [instrumental]: No vocals, pure instrumental music\n"
        "- [break down]: Simplify and reduce elements\n"
        "- [drum fill]: Drum-focused percussion moment\n"
        "- [chopped samples]: Sampled/cut-up audio moment\n"
        "- [drop]: Build-up release, powerful moment\n"
        "- [synth solo]: Synthesizer melody moment\n"
        "- [vocal harmony]: Multiple vocal layers\n"
        "- [reverb wash]: Echo/spacious effect\n"
        "- [bass line]: Prominent bass moment\n"
        "- [acapella]: Vocals only, no instruments\n"
        "- [delay effect]: Repeating echoes\n"
        "- [beat drop]: Rhythm emphasis point\n"
        "- [ambient pad]: Atmospheric, sustained sounds\n"
        "- [call and response]: Interactive vocal section\n"
        "- [glitch]: Digital/stuttering effect\n"
        "- [swell]: Gradual volume increase\n"
        "- [breakdown section]: Musical simplification\n"
        "- [bridge moment]: Transitional section\n"
        "- [crisp hi-hats]: Bright, sharp hi-hat percussion\n"
        "- [filtered saw bass]: Synthesized bass with filtering\n"
        "- [evolving synth leads]: Dynamic synthesizer melodies\n"
        "- [intro]: Opening moment\n"
        "- [outro]: Closing moment\n\n"
        "BE CREATIVE: Think about what musical elements, effects, and moments would make sense. "
        "Create tags that describe actual production events that Ace-Step should generate. "
        "Mix sung content with creative production tags to build dynamic arrangements."
    )
    
    formatting_rules = (
        "FORMATTING RULES (VERY IMPORTANT):\n"
        "1. EACH LYRIC LINE = ONE LINE ONLY (no multiple lyrics on same line)\n"
        "2. EACH PRODUCTION TAG = ONE LINE ONLY (tag alone on its line, like: [tag name])\n"
        "3. SPACING BETWEEN RHYMING GROUPS = ONE BLANK LINE after each rhyming phrase group\n"
        "4. LYRIC LINE LENGTH - CRITICAL FOR SINGING:\n"
        "   - MAXIMUM 10-12 WORDS per line (easier to sing without mumbling)\n"
        "   - MAXIMUM 60 characters per line (fits natural breath phrasing)\n"
        "   - SHORT lines are SINGABLE lines\n"
        "   - LONG lines force singers to rush and mumble\n"
        "   - Each line should be ONE MUSICAL PHRASE/THOUGHT\n"
        "   - Examples of GOOD line length:\n"
        "     * 'In the silence, I hear' (5 words)\n"
        "     * 'A rhythm that's calling me' (5 words)\n"
        "     * 'The night is young and alive' (6 words)\n"
        "     * 'Oh, the music fills my soul' (6 words)\n"
        "   - Examples of BAD line length (TOO LONG - AVOID):\n"
        "     * 'In the velvet haze of twilight, stars align' (8 words, OK length but too poetic)\n"
        "     * 'A celestial map, guiding me to the rhythm's shrine' (10 words, too complex)\n"
        "     * 'Softly glowing embers of a fire that's yet to rise' (11 words, WAY TOO LONG)\n"
        "     * 'Illuminating the path where music meets the skies' (9 words, still too long)\n"
        "5. DO NOT put tags and lyrics on the same line\n"
        "6. DO NOT put multiple lyrics on the same line\n"
        "7. USE BLANK LINES to separate rhyming groups for clarity\n"
        "8. KEEP IT SIMPLE - Short, punchy, singable lines are BETTER than long poetic lines"
    )
    
    creativity_guide = (
        "VOCABULARY & CREATIVITY RULES (CRITICAL FOR AVOIDING REPETITION):\n"
        "AVOID OVERUSED PHRASES - You must use diverse vocabulary throughout the song:\n"
        "- NEVER repeat phrases like 'feel the beat', 'heart beating', 'music alive', 'energy thrive' in the same song\n"
        "- NEVER use the same adjectives repeatedly (don't say 'young' and 'young', 'alive' and 'alive')\n"
        "- NEVER reuse the same verbs across multiple lines (avoid repeating 'feel', 'hear', 'dance', 'move')\n"
        "- Each verse/section should have COMPLETELY DIFFERENT vocabulary from other verses\n\n"
        "VOCABULARY DIVERSITY TECHNIQUES:\n"
        "1. USE SYNONYMS AND VARIED EXPRESSIONS:\n"
        "   Instead of repeating 'feel the beat':\n"
        "   - 'sense the rhythm', 'catch the pulse', 'surrender to the groove', 'ride the tempo'\n"
        "   - 'chase the bass', 'follow the drums', 'let the rhythm guide me'\n\n"
        "2. DESCRIBE DIFFERENT SENSATIONS:\n"
        "   Use varied sensory words:\n"
        "   - Visual: 'colors flash', 'lights spin', 'shadows dance', 'neon glow'\n"
        "   - Touch: 'skin alive', 'warmth rising', 'hands reaching', 'fingers moving'\n"
        "   - Emotional: 'soul soaring', 'mind spinning', 'spirit lifted', 'essence flowing'\n\n"
        "3. CREATE UNIQUE METAPHORS:\n"
        "   - Avoid clichés like 'heart beating to the music'\n"
        "   - Use fresh imagery: 'breathing with the synths', 'thoughts dissolving in sound'\n"
        "   - Mix unexpected comparisons: 'like rain on summer nights', 'like fire and ice colliding'\n\n"
        "4. VARY SENTENCE STRUCTURE:\n"
        "   - Don't use 'I feel...' for every line\n"
        "   - Alternate: 'Lights explode', 'My vision blurs', 'The music lifts me', 'Energy surrounds'\n"
        "   - Use questions, exclamations, statements, fragments\n\n"
        "5. BUILD VOCABULARY BANKS FOR DIFFERENT THEMES:\n"
        "   For energy/excitement: surge, ignite, explode, rush, ignition, burst, blaze, surge\n"
        "   For movement: glide, spiral, float, swirl, cascade, wave, undulate, propel\n"
        "   For sound: echo, resonate, vibrate, hum, throb, pulse, reverberate, sing\n"
        "   For emotion: soar, transcend, liberate, dissolve, awaken, illuminate, unlock\n\n"
        "SELF-CHECK BEFORE GENERATING:\n"
        "- If you use a word, DON'T use it again in the next 5 lines minimum\n"
        "- If you write 'feel', check you don't have it 2 lines away\n"
        "- Every section should feel like a NEW exploration of the theme\n"
        "- Think: Would a human lyricist write the same phrase 3 times in one song? NO!\n\n"
        "EXAMPLE OF POOR REPETITION (AVOID):\n"
        "  I feel the beat tonight\n"
        "  Feel the music in my heart\n"
        "  When I feel the rhythm strong\n"
        "  Dancing to this beat of mine\n\n"
        "EXAMPLE OF GOOD DIVERSITY (DO THIS):\n"
        "  I feel the beat tonight\n"
        "  Let the music fill my chest\n"
        "  When the rhythm takes control\n"
        "  Dancing through this sonic dream"
    )
    
    instructions = (
        "You are a creative music lyricist working with Ace-Step AI music generator. "
        "Your job is to write song lyrics WITH embedded production tags that guide the music generation.\n\n"
        f"{ace_step_guide}\n\n"
        f"{instrumentality_guidance}\n\n"
        f"{formatting_rules}\n\n"
        f"{creativity_guide}\n\n"
        "GUIDELINES:\n"
        "- Write naturally flowing song content without tags\n"
        "- INSERT CREATIVE PRODUCTION TAGS [in brackets] to describe musical events\n"
        "- Tags describe what HAPPENS MUSICALLY at that moment\n"
        "- Be creative! Think about the song's progression and add tags that make sense\n"
        "- Mix vocal sections with instrumental/effect sections using tags\n"
        "- Return ONLY the lyrics with tags - no titles, explanations, or markdown\n"
        "- Keep the song cohesive but dynamic and interesting\n"
        "- REMEMBER: Each line is ONE item (either a tag, or one lyric line)\n"
        "- REMEMBER: Add blank lines between rhyming groups for visual separation\n"
        "- CRITICAL: KEEP LYRIC LINES SHORT (max 10-12 words, max 60 characters)\n"
        "  * Short lines are singable and don't force mumbling\n"
        "  * Break long thoughts into multiple short lines\n"
        "  * Each line = ONE musical breath/phrase\n"
        "- RESPECT INSTRUMENTALITY BALANCE:\n"
        "  * If 'Very Instrumental': Use MANY tags, vocals appear as SHORT BURSTS only\n"
        "  * If 'Balanced': Equal mix of tags and vocals\n"
        "  * If 'Vocal Focused': Mostly vocals with tags for emphasis\n"
        "  * Don't put multiple vocal lines in a row - separate with instrumental tags\n"
        "- MOST IMPORTANT: Use DIVERSE vocabulary - never repeat phrases or words excessively\n\n"
        f"Generate a {description} song with creative production elements, SHORT singable lines, and UNIQUE vocabulary throughout. [Seed: {seed}]"
    )
    return instructions


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
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def _build_prompt(self, style: str, seed: int, instrumentality_balance: str = "Balanced") -> str:
        return build_acestore_lyrics_prompt(style, seed, instrumentality_balance)

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

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int, instrumentality_balance: str = "Balanced", control_before_generate=None):
        api_key = api_key.strip()
        if not api_key:
            return ("Error: API key is missing.",)

        prompt = self._build_prompt(style, seed, instrumentality_balance)
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
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def _build_prompt(self, style: str, seed: int, instrumentality_balance: str = "Balanced") -> str:
        return build_acestore_lyrics_prompt(style, seed, instrumentality_balance)

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

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int, instrumentality_balance: str = "Balanced", control_before_generate=None):
        api_key = api_key.strip()
        if not api_key:
            return ("Error: API key is missing.",)

        prompt = self._build_prompt(style, seed, instrumentality_balance)
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
                    "gpt-5.1",
                    "gpt-5.1-codex",
                    "gpt-5",
                    "gpt-5-pro",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4",
                    "o3",
                    "o3-mini",
                    "o1",
                    "o1-mini"
                ], {"default": "gpt-4o"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        # Build Ace-Step aware prompt
        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                    "claude-opus-4.5",
                    "claude-opus-4.1",
                    "claude-sonnet-4.5",
                    "claude-sonnet-4",
                    "claude-haiku-4.5",
                    "claude-haiku-3.5",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20250219"
                ], {"default": "claude-opus-4.5"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        # Build Ace-Step aware prompt
        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                    "sonar-reasoning-pro",
                    "sonar-deep-research"
                ], {"default": "sonar"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                    "command-a-03-2025",
                    "command-r7b-12-2024",
                    "command-r-plus-08-2024",
                    "command-r-08-2024",
                    "command-a-translate",
                    "command-a-reasoning",
                    "command-a-vision",
                    "aya-expanse-32b",
                    "aya-expanse-8b",
                    "aya-vision",
                    "aya-translate"
                ], {"default": "command-a-03-2025"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "HuggingFace API Token (read access)"}),
                "model": ([
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "HuggingFaceH4/zephyr-7b-beta",
                    "tiiuae/falcon-7b-instruct",
                    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                    "teknium/OpenHermes-2.5-Mistral-7B",
                    "TheBloke/neural-chat-7B-v3-3-GGUF",
                    "Qwen/Qwen1.5-7B-Chat",
                    "openlm-research/open_llama_7b_v2"
                ], {"default": "mistralai/Mistral-7B-Instruct-v0.2"}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API token not provided. Get it from https://huggingface.co/settings/tokens",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
        try:
            # Use text generation task with minimal parameters
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            # Try text-generation endpoint first (more reliable)
            payload = json.dumps({
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_tokens, 1024),
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            }).encode()

            req = urllib.request.Request(
                f"https://api-inference.huggingface.co/models/{model}",
                data=payload,
                headers=headers,
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode())
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        lyrics = result[0].get("generated_text", "")
                        # Remove prompt from output if present
                        if lyrics.startswith(prompt):
                            lyrics = lyrics[len(prompt):].strip()
                        return (lyrics if lyrics else "Generated empty response",)
                    elif isinstance(result, dict):
                        if "generated_text" in result:
                            lyrics = result.get("generated_text", "")
                            if lyrics.startswith(prompt):
                                lyrics = lyrics[len(prompt):].strip()
                            return (lyrics if lyrics else "Generated empty response",)
                        elif "error" in result:
                            return (f"Model error: {result['error']}",)
                        else:
                            return (f"Unexpected response: {str(result)[:200]}",)
                    else:
                        return (f"Unexpected response format: {str(result)[:200]}",)
                        
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="ignore")
                if e.code == 410:
                    return (f"Error 410: Model unavailable. Try another model from the list.",)
                elif e.code == 401:
                    return (f"Error 401: Invalid API token. Check https://huggingface.co/settings/tokens",)
                elif e.code == 503:
                    return (f"Error 503: Model is loading. Please try again in 30 seconds.",)
                elif e.code == 429:
                    return (f"Error 429: Rate limited. Please wait a few minutes.",)
                else:
                    return (f"HTTP Error {e.code}: {error_body[:150]}",)
                    
        except urllib.error.URLError as e:
            return (f"Connection error: {str(e.reason)[:150]}",)
        except Exception as e:
            return (f"Error: {str(e)[:200]}",)


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
                    "mistralai/Mistral-Small-24B-Instruct-2501",
                    "Qwen/Qwen2.5-72B-Instruct",
                    "deepseek-ai/DeepSeek-V3",
                    "moonshotai/Kimi-K2-Instruct",
                    "GLM-4-Plus",
                    "Nous-Hermes-3-70B"
                ], {"default": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
                    "deepseek-ai/deepseek-v3",
                    "deepseek-ai/deepseek-r1",
                    "Qwen/Qwen3-235B-A22B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct-Turbo",
                    "meta-llama/Llama-4-Maverick-17B",
                    "meta-llama/Llama-4-Scout-17B",
                    "meta-llama/Llama-3.3-70B-Instruct",
                    "meta-llama/Llama-3.1-405B-Instruct",
                    "mistralai/Mistral-Large-3-675B-Instruct",
                    "mistralai/Mistral-Small-24B-Instruct-2501",
                    "google/GLM-4.6",
                    "moonshotai/Kimi-K2",
                    "google/Gemma-3-27b"
                ], {"default": "meta-llama/Llama-3.3-70B-Instruct"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instrumentality_balance": (
                    [
                        "Pure Instrumental",
                        "Very Instrumental",
                        "Balanced",
                        "Vocal Focused",
                        "Vocals Only",
                    ],
                    {
                        "default": "Balanced",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "JK AceStep Nodes/Lyrics"

    def generate(self, text, api_key, model, max_tokens, seed, instrumentality_balance: str = "Balanced"):
        if not api_key:
            return ("Error: API key not provided",)

        prompt = build_acestore_lyrics_prompt(text, seed, instrumentality_balance)
        
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
