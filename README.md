# JK AceStep Nodes for ComfyUI

Advanced nodes optimized for [ACE-Step](https://huggingface.co/Accusoft/ACE-Step) audio generation in ComfyUI.

## 📦 Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jeankassio/JK-AceStep-Nodes.git
```

Restart ComfyUI. Nodes will appear under `JK AceStep Nodes/` categories.

---

## 🎵 Nodes

### Ace-Step KSampler (Basic)
Full-featured sampler with quality check, advanced guidance (APG, CFG++, Dynamic CFG), anti-autotune smoothing, and automatic step optimization.

**Category:** `JK AceStep Nodes/Sampling`

---

### Ace-Step KSampler (Advanced)
Extended sampler with start/end step control for multi-pass workflows and refinement.

**Category:** `JK AceStep Nodes/Sampling`

---

### Ace-Step Prompt Gen
Prompt generator with **150+ professional music styles** (Electronic, Rock, Jazz, Classical, Brazilian, World Music, and more).

**Category:** `JK AceStep Nodes/Prompt`

---

## 🎤 Lyrics Generators

Ten AI-powered lyrics generation nodes supporting various LLM providers:

### Ace-Step OpenAI Lyrics
Lyrics generation using OpenAI GPT models.

**Supported Models:**
- `gpt-4o` - Latest multimodal model (recommended)
- `gpt-4o-mini` - Smaller, faster variant
- `gpt-4-turbo` - Previous gen high performance
- `gpt-4` - Stable production model
- `gpt-3.5-turbo` - Fast, cost-effective

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Claude Lyrics
Lyrics generation using Anthropic Claude models.

**Supported Models:**
- `claude-3-5-sonnet-20241022` - Latest, best quality
- `claude-3-opus-20250219` - Most capable variant
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast, compact model

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Gemini Lyrics
Lyrics generation using Google Gemini API.

**Supported Models:**
- `gemini-2.5-flash` - Latest flash model
- `gemini-2.5-flash-latest` - Latest updates
- `gemini-2.5-flash-lite` - Fastest/cheapest
- `gemini-2.5-pro` - Highest quality (paid)
- `gemini-2.5-pro-latest` - Pro with latest features
- `gemini-2.0-flash` - Previous generation
- `gemini-2.0-flash-lite` - Previous lite variant
- `gemini-1.5-pro` - Older pro model
- `gemini-1.5-flash` - Older flash model

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Groq Lyrics
High-speed lyrics generation using Groq API.

**Supported Models (Production):**
- `llama-3.3-70b-versatile` - Meta Llama 3.3 (best quality)
- `llama-3.1-8b-instant` - Meta Llama 3.1 (fast, compact)
- `llama-3.2-1b-preview` - Meta Llama 3.2 (ultra-compact)
- `llama-3.2-3b-preview` - Meta Llama 3.2 (small)
- `llama-3.2-11b-vision-preview` - Meta Llama with vision
- `llama-3.2-90b-vision-preview` - Large vision model
- `meta-llama/llama-guard-3-8b` - Meta Llama Guard (safety)
- `openai/gpt-oss-120b` - OpenAI OSS 120B
- `openai/gpt-oss-20b` - OpenAI OSS 20B

**Supported Models (Preview):**
- `meta-llama/llama-4-maverick-17b-128e-instruct` - Llama 4 (preview)
- `meta-llama/llama-4-scout-17b-16e-instruct` - Llama 4 Scout (preview)

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Perplexity Lyrics
Lyrics generation using Perplexity Sonar models.

**Supported Models:**
- `sonar` - Standard model
- `sonar-pro` - Professional variant
- `sonar-reasoning` - Reasoning-focused
- `sonar-reasoning-pro` - Professional reasoning

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Cohere Lyrics
Lyrics generation using Cohere Command models.

**Supported Models:**
- `command-a-03-2025` - Latest Command A (upcoming)
- `command-r7b-12-2024` - December 2024 update
- `command-r-plus-08-2024` - R+ August 2024
- `command-r-08-2024` - R August 2024
- `command-r-plus` - Latest R+ variant
- `command-r` - Latest R variant
- `command-a-translate-08-2025` - Specialized translation model
- `command-a-reasoning-08-2025` - Reasoning-focused variant

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Replicate Lyrics
Lyrics generation using Replicate API models.

**Supported Models:**
- `meta/llama-3.1-405b-instruct` - 405B instruction-tuned
- `meta/llama-3.1-70b-instruct` - 70B instruction-tuned
- `meta/llama-3.1-8b-instruct` - 8B instruction-tuned
- `meta/llama-3-70b-instruct` - Llama 3 70B
- `meta/llama-2-70b-chat` - Llama 2 chat 70B
- `mistralai/mistral-7b-instruct-v0.3` - Mistral 7B v0.3
- `mistralai/mistral-small-24b-instruct-2501` - Mistral Small 24B
- `mistralai/mixtral-8x7b-instruct-v0.1` - Mixtral MoE

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step HuggingFace Lyrics
Lyrics generation using HuggingFace Inference API.

**Supported Models:**
- `meta-llama/Llama-3.1-405B-Instruct` - Large instruction-tuned
- `meta-llama/Llama-3.1-70B-Instruct` - 70B instruction-tuned
- `mistralai/Mistral-7B-Instruct-v0.2` - Mistral 7B
- `deepseek-ai/deepseek-v3` - DeepSeek V3
- `qwen/Qwen2.5-72B-Instruct` - Qwen 2.5 72B
- `HuggingFaceH4/zephyr-7b-beta` - Zephyr 7B
- `tiiuae/falcon-7b-instruct` - Falcon 7B

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Together AI Lyrics
Lyrics generation using Together AI serverless models.

**Supported Models (Selection):**
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` - Llama 3.3 70B
- `meta-llama/Llama-3.1-405B-Instruct-Turbo` - Llama 3.1 405B
- `meta-llama/Llama-3.1-70B-Instruct-Turbo` - Llama 3.1 70B
- `meta-llama/Llama-3.1-8B-Instruct-Turbo` - Llama 3.1 8B
- `mistralai/Mistral-Small-24B-Instruct-2501` - Mistral Small 24B
- `mistralai/Ministral-3-8B-Instruct-2512` - Ministral 3 8B
- `deepseek-ai/DeepSeek-V3.1` - DeepSeek V3.1
- `deepseek-ai/DeepSeek-R1` - DeepSeek R1 reasoning
- `Qwen/Qwen3-235B-A22B-Instruct-2507` - Qwen 3 235B
- `moonshotai/Kimi-K2-Instruct-0905` - Kimi K2 instruction
- `google/gemma-3-27b-it` - Gemma 3 27B
- Plus 50+ additional models available

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Fireworks Lyrics
Lyrics generation using Fireworks AI models (100+ available).

**Supported Models (Selection):**
- `deepseek-ai/deepseek-v3p2` - DeepSeek V3 P2
- `deepseek-ai/deepseek-r1` - DeepSeek R1 reasoning
- `Qwen/Qwen3-235B-A22B-Instruct-2507` - Qwen 3 235B
- `Qwen/Qwen3-Next-80B-A3B-Instruct` - Qwen 3 Next 80B
- `meta-llama/Llama-3.3-70B-Instruct` - Llama 3.3 70B
- `meta-llama/Llama-3.1-405B-Instruct` - Llama 3.1 405B
- `meta-llama/Llama-3.1-70B-Instruct` - Llama 3.1 70B
- `mistralai/Mistral-Large-3-675B-Instruct-2512` - Mistral Large 675B
- `mistralai/Mistral-Small-24B-Instruct-2501` - Mistral Small 24B
- `mistralai/Mistral-Nemo-Instruct-2407` - Mistral Nemo
- `mistralai/Mixtral-8x22B-Instruct` - Mixtral 8x22B
- `zai-org/GLM-4.6` - GLM 4.6
- Plus 90+ additional models available

**Category:** `JK AceStep Nodes/Lyrics`

---

### Ace-Step Save Text
Text saver with auto-incremented filenames and folder support. Works with any lyrics generator.

**Category:** `JK AceStep Nodes/IO`

---

## 🎨 JKASS Custom Sampler

**J**ust **K**eep **A**udio **S**ampling **S**imple - custom sampler optimized for audio generation.

### Available Variants

- **`jkass_quality`** - Second-order Heun method for maximum audio quality
  - Superior accuracy and detail preservation
  - Recommended for final renders
  - ~2x slower than fast variant

- **`jkass_fast`** - First-order Euler method for faster generation
  - Optimized for speed with vectorized operations
  - Good quality with reduced computation time
  - Best for iterations and previews

### Key Features
- No noise normalization (preserves audio dynamics)
- Clean sampling path (prevents word cutting/stuttering)
- Optimized for long-form audio

Select your preferred variant from any sampler dropdown (default: `jkass_quality`).

---

## 📊 Recommended Settings

- **Sampler:** `jkass_quality` (for best quality) or `jkass_fast` (for speed)
- **Scheduler:** `sgm_uniform`
- **Steps:** 80-100
- **CFG:** 4.0-4.5
- **Anti-Autotune:** 0.25-0.35 (vocals), 0.0-0.15 (instruments)

---

## 🎯 Quality Check Feature

Automatically tests multiple step counts to find optimal settings for your prompt.

**Important:** Quality scores are **comparative only**. Compare within same prompt/style. Electronic music naturally scores lower than acoustic (both can be excellent quality).

---

## 🔧 Troubleshooting

- **Word cutting/stuttering:** Use `jkass_quality` sampler, disable advanced optimizations
- **Metallic voice:** Increase `anti_autotune_strength` to 0.3-0.4
- **Poor quality:** Increase steps (80-120), use CFG 4.0-4.5, enable APG, try `jkass_quality` sampler

---

## 📁 Project Structure

```
JK-AceStep-Nodes/
├── __init__.py                    # Central node aggregator
├── ace_step_ksampler.py           # Samplers (Basic + Advanced)
├── ace_step_prompt_gen.py         # Prompt generator (150+ styles)
├── lyrics_nodes.py                # 10 lyrics generators consolidated
├── ace_step_save_text.py          # Text saver node
├── requirements.txt
└── py/
    └── jkass_sampler.py           # Custom audio sampler
```

### Available Lyrics Generators

- **OpenAI** - gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo, and more
- **Anthropic Claude** - Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- **Google Gemini** - gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro/flash
- **Groq** - Llama 3.3 70B, Llama 3.1 8B, Llama Guard 3, GPT-OSS (120B/20B), and Llama 4 preview models
- **Perplexity** - Sonar, Sonar Pro, Sonar Reasoning (with 128k context)
- **Cohere** - Command A/R+ (with reasoning & vision), Aya (multilingual)
- **Replicate** - Llama 3.1 (405B/70B/8B), Mistral Small/Nemo, Mixtral
- **HuggingFace** - Llama 3.1, Mistral, DeepSeek, Qwen, Falcon, and 100+ more
- **Together AI** - Llama 3.3/3.1, DeepSeek, Qwen 3, Mistral variants, and 50+ more
- **Fireworks AI** - DeepSeek V3/R1, Qwen 3, Llama 3.3/3.1, Mistral Large/Small, GLM, and 90+ more

---

## 📄 License

MIT License

---

**Version:** 2.3  
**Last Updated:** December 2025

🎵 Happy music generation!
