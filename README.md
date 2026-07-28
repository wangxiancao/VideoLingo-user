<div align="center">

<img src="/docs/logo.png" alt="VideoLingo Logo" height="140">

# Connect the World, Frame by Frame

<a href="https://trendshift.io/repositories/12200" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12200" alt="Huanshere%2FVideoLingo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

## 🌟 Overview ([Try VL Now!](https://videolingo.io))

VideoLingo is an all-in-one video translation, localization, and dubbing tool aimed at generating Netflix-quality subtitles. It eliminates stiff machine translations and multi-line subtitles while adding high-quality dubbing, enabling global knowledge sharing across language barriers.

Key features:
- 🎥 YouTube video download via yt-dlp

- **🎙️ Word-level subtitle recognition with Whisper-large-v3-turbo (99 languages, DTW cross-attention word timestamps)**

- **📝 NLP and AI-powered subtitle segmentation**

- **📚 Custom + AI-generated terminology for coherent translation**

- **🔄 3-step Translate-Reflect-Adaptation for cinematic quality**

- **✅ Netflix-standard, Single-line subtitles Only**

- **🗣️ Dubbing with IndexTTS2 and custom TTS**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

Difference from similar projects: **Single-line subtitles only, superior translation quality, seamless dubbing experience**

## 🎥 Demo

<table>
<tr>
<td width="50%">

### Dual Subtitles
---
https://github.com/user-attachments/assets/a5c3d8d1-2b29-4ba9-b0d0-25896829d951

</td>
<td width="50%">

### Voice Clone Dubbing
---
https://github.com/user-attachments/assets/e065fe4c-3694-477f-b4d6-316917df7c0a

</td>
</tr>
</table>

### Language Support

**Input Language Support (99 languages via Whisper):**

🇺🇸 English 🤩 | 🇨🇳 Chinese 🤩 | 🇯🇵 Japanese 🤩 | 🇫🇷 French 🤩 | 🇩🇪 German 🤩 | 🇮🇹 Italian 🤩 | 🇪🇸 Spanish 🤩 | 🇷🇺 Russian 😊 | 🇰🇷 Korean 🤩 | 🇵🇹 Portuguese 🤩 | 🇻🇳 Vietnamese | 🇹🇭 Thai | 🇮🇩 Indonesian | 🇲🇾 Malay | 🇹🇷 Turkish | 🇳🇱 Dutch | 🇵🇱 Polish | 🇸🇪 Swedish | 🇫🇮 Finnish | 🇩🇰 Danish | 🇨🇿 Czech | 🇭🇺 Hungarian | 🇷🇴 Romanian | 🇬🇷 Greek | 🇸🇦 Arabic | 🇮🇳 Hindi | 🇮🇷 Persian | 🇵🇭 Filipino | 🇲🇰 Macedonian | 🇨🇦 Cantonese + 70 more languages

> Whisper also supports **code-switching** (mixed-language audio, e.g. Chinese-English).

**Translation supports all languages, while dubbing language depends on the chosen TTS method.**

## Installation

Meet any problem? Chat with our free online AI agent [**here**](https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh) to help you.

> **Note:** For Windows users with NVIDIA GPU, follow these steps before installation:
> 1. Install [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)
> 2. Install [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)
> 3. Add `C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6` to your system PATH
> 4. Restart your computer

> **Note:** FFmpeg is required. Please install it via package managers:
> - Windows: ```choco install ffmpeg``` (via [Chocolatey](https://chocolatey.org/))
> - macOS: ```brew install ffmpeg``` (via [Homebrew](https://brew.sh/))
> - Linux: ```sudo apt install ffmpeg``` (Debian/Ubuntu)

1. Clone the repository

```bash
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo
```

2. Install dependencies(requires `python=3.10`)

```bash
conda create -n videolingo python=3.10.0 -y
conda activate videolingo
python install.py
```

3. Start the application

```bash
streamlit run st.py
```

### Docker
Alternatively, you can use Docker (requires CUDA 12.4+ and NVIDIA Driver version >550), see [Docker docs](/docs/pages/docs/docker.en-US.md):

```bash
docker build -t videolingo .
docker run -d -p 8501:8501 --gpus all videolingo
```

## Architecture

VideoLingo's pipeline consists of 12 numbered stages, each producing intermediate artifacts:

| Stage | Module | Description |
|---|---|---|
| 1 | `_1_ytdlp.py` | Video download / file import |
| 2 | `_2_asr.py` | **Whisper-large-v3-turbo** transcription with word-level DTW timestamps |
| 3 | `_3_1_split_nlp.py` / `_3_2_split_meaning.py` | NLP + AI-powered sentence segmentation |
| 4 | `_4_1_summarize.py` / `_4_2_translate.py` | AI summary + 3-step translation (Reflect-Adapt) |
| 5 | `_5_split_sub.py` | Subtitle length optimization |
| 6 | `_6_gen_sub.py` | Timestamp alignment + subtitle generation |
| 7 | `_7_sub_into_vid.py` | Burn subtitles into video |
| 8-12 | `_8`–`_12` | TTS dubbing pipeline (IndexTTS2 / custom TTS) |

**ASR**: [Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (809M params) via `transformers` pipeline with `return_timestamps="word"`. No external ASR packages required — just `transformers` + `torch`.

**TTS**: [IndexTTS2](https://github.com/index-tts/index-tts) voice cloning (runs as separate HTTP worker) or user-implementable `custom_tts` hook.

## APIs
VideoLingo supports OpenAI-Like API format and various TTS interfaces:
- LLM: `claude-3-5-sonnet`, `gpt-4.1`, `deepseek-v3`, `gemini-2.0-flash`, ... (sorted by performance, be cautious with gemini-2.5-flash...)
- ASR: **Whisper-large-v3-turbo** (local GPU, transformers pipeline with DTW word timestamps)
- TTS: `custom-tts`, **`indextts2`**

> **Note:** VideoLingo works with **[302.ai](https://gpt302.saaslink.net/C2oHR9)** for LLM API. Or run fully locally with Whisper + a local LLM + IndexTTS2 for free, no API needed!

For detailed installation, API configuration, and batch mode instructions, please refer to the documentation: [English](/docs/pages/docs/start.en-US.md) | [中文](/docs/pages/docs/start.zh-CN.md)

### CLI Usage

```bash
# Subtitles only (default)
python translate_video.py /path/to/video.mp4

# Subtitles + dubbing
python translate_video.py --mode tts /path/to/video.mp4

# From YouTube URL
python translate_video.py --mode tts https://www.youtube.com/watch?v=xxx
```

## Current Limitations

1. ASR transcription quality depends on audio clarity. For videos with loud background music, please enable Voice Separation Enhancement (Demucs). Whisper-large-v3-turbo provides word-level timestamps via DTW cross-attention alignment.

2. Using weaker models can lead to errors during processes due to strict JSON format requirements for responses (tried my best to prompt llm😊). If this error occurs, please delete the `output` folder and retry with a different LLM, otherwise repeated execution will read the previous erroneous response causing the same error.

3. The dubbing feature may not be 100% perfect due to differences in speech rates and intonation between languages, as well as the impact of the translation step. However, this project has implemented extensive engineering processing for speech rates to ensure the best possible dubbing results.

4. **Code-switching support**: Whisper handles mixed-language audio (e.g. Chinese-English), but downstream translation and TTS stages currently process a single primary language. Full multilingual pipeline support is a work in progress.

5. **Speaker diarization**: The pipeline does not currently separate multiple speakers for independent dubbing. Whisper does not include native diarization; this would require an external system (e.g. pyannote).

## 📄 License

This project is licensed under the Apache 2.0 License. Special thanks to the following open source projects for their contributions:

[Whisper](https://github.com/openai/whisper), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [json_repair](https://github.com/mangiucugna/json_repair), [BELLE](https://github.com/LianjiaTech/BELLE)

## 📬 Contact Me

- Submit [Issues](https://github.com/Huanshere/VideoLingo/issues) or [Pull Requests](https://github.com/Huanshere/VideoLingo/pulls) on GitHub
- DM me on Twitter: [@Huanshere](https://twitter.com/Huanshere)
- Email me at: team@videolingo.io

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Huanshere/VideoLingo&type=Timeline)](https://star-history.com/#Huanshere/VideoLingo&Timeline)

---

<p align="center">If you find VideoLingo helpful, please give me a ⭐️!</p>
