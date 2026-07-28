import os

# vLLM forks its engine-core subprocess by default. Since this process already
# initializes CUDA (ForcedAligner load / is_available() check), a forked child
# inherits that state and torch refuses to re-init CUDA there, raising
# "Cannot re-initialize CUDA in forked subprocess". vLLM reads this env var via
# get_mp_context() — NOT Python's global set_start_method — so we set it here,
# before qwen_asr is imported (that import pulls in vllm).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import librosa
from qwen_asr import Qwen3ASRModel
from core.utils import *
from core.utils.models import *
from rich import print as rprint

# ----------------------------------------
# Language name to ISO 639-1 mapping
# ----------------------------------------
LANG_NAME_TO_CODE = {
    'chinese': 'zh', 'english': 'en', 'japanese': 'ja',
    'french': 'fr', 'german': 'de', 'spanish': 'es',
    'russian': 'ru', 'italian': 'it', 'korean': 'ko',
    'portuguese': 'pt', 'arabic': 'ar', 'hindi': 'hi',
    'thai': 'th', 'vietnamese': 'vi', 'indonesian': 'id',
    'malay': 'ms', 'turkish': 'tr', 'dutch': 'nl',
    'polish': 'pl', 'romanian': 'ro', 'greek': 'el',
    'swedish': 'sv', 'finnish': 'fi', 'danish': 'da',
    'czech': 'cs', 'hungarian': 'hu', 'macedonian': 'mk',
    'persian': 'fa', 'filipino': 'tl', 'cantonese': 'yue',
}

# ----------------------------------------
# Singleton model cache
# ----------------------------------------
_MODEL = None


def get_model():
    """Load Qwen3-ASR model once and cache it globally. Supports transformers and vLLM backends."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_name = load_key("qwen3_asr.model")
    aligner_name = load_key("qwen3_asr.forced_aligner")
    device_cfg = load_key("qwen3_asr.device")
    dtype_cfg = load_key("qwen3_asr.dtype")
    backend = load_key("qwen3_asr.backend")

    # Resolve device
    if device_cfg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # Resolve dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    # ForcedAligner always runs on transformers (NAR model, lightweight)
    aligner_kwargs = dict(device_map=device)
    if dtype_cfg in dtype_map:
        aligner_kwargs["dtype"] = dtype_map[dtype_cfg]

    # vLLM requires CUDA; fallback to transformers on CPU
    if backend == "vllm" and (device == "cpu" or not torch.cuda.is_available()):
        rprint("[yellow]⚠️ vLLM backend requires CUDA, falling back to transformers[/yellow]")
        backend = "transformers"

    if backend == "vllm":
        gpu_mem_util = load_key("qwen3_asr.gpu_memory_utilization")
        rprint(f"[cyan]🔄 Loading Qwen3-ASR (vLLM native): {model_name}, GPU mem util={gpu_mem_util}[/cyan]")
        rprint(f"[cyan]🔄 Forced aligner: {aligner_name} (transformers, device={device})[/cyan]")

        # qwen-asr registers a pre-v0.16 vLLM shim (Qwen3ASRForConditionalGeneration)
        # into vLLM's ModelRegistry at import time (qwen_asr/__init__.py). On vLLM >=0.16
        # that shim's _get_data_parser API was removed, breaking engine init. Re-register
        # vLLM's native class so the engine builds with the current API. We bypass
        # Qwen3ASRModel.LLM() (which couples to the shim) and construct the wrapper
        # directly with a native vllm.LLM, reusing qwen-asr's .transcribe() orchestration.
        from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration as _NativeQwen3ASR
        from vllm import ModelRegistry
        ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", _NativeQwen3ASR)

        from vllm import LLM as vLLM, SamplingParams
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

        # Release GPU memory cached by prior stages (e.g. Demucs vocal
        # separation) so vLLM profiles a cleaner free-memory state.
        torch.cuda.empty_cache()

        llm = vLLM(
            model=model_name,
            gpu_memory_utilization=gpu_mem_util,
            limit_mm_per_prompt={"audio": 1},
            # Model default is 65536 (≈20 min audio). ForcedAligner chunks at
            # 180s (≈10K tokens), so 32768 (≈600s) is ample and keeps the KV
            # cache small enough to coexist with the ForcedAligner on one GPU.
            max_model_len=32768,
        )
        processor = Qwen3ASRProcessor.from_pretrained(model_name, fix_mistral_regex=True)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
        forced_aligner_model = Qwen3ForcedAligner.from_pretrained(aligner_name, **aligner_kwargs)

        _MODEL = Qwen3ASRModel(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            # Process one audio chunk at a time. With -1 (qwen-asr's default for
            # vLLM), all ~6 chunks (180s each) are batched into one forward pass,
            # producing a logits tensor of batch×seq×vocab ≈ 5 GB that OOMs
            # alongside vLLM. Batch size 1 keeps peak ForcedAligner memory ~2-3 GB.
            max_inference_batch_size=1,
            max_new_tokens=None,
        )
    else:
        dtype_kwargs = {}
        if dtype_cfg in dtype_map:
            dtype_kwargs["dtype"] = dtype_map[dtype_cfg]

        rprint(f"[cyan]🔄 Loading Qwen3-ASR (transformers): {model_name}[/cyan]")
        rprint(f"[cyan]🔄 Device={device}, Dtype={dtype_cfg}, Forced aligner={aligner_name}[/cyan]")

        _MODEL = Qwen3ASRModel.from_pretrained(
            model_name,
            device_map=device,
            forced_aligner=aligner_name,
            forced_aligner_kwargs=aligner_kwargs,
            **dtype_kwargs,
        )

    rprint(f"[green]✅ Qwen3-ASR loaded ({backend} backend)[/green]")
    return _MODEL


def map_language(qwen_lang: str) -> str:
    """Convert Qwen3 language output (e.g. 'Chinese', 'English', 'Chinese,English') to ISO 639-1."""
    if not qwen_lang:
        return 'en'
    # For code-switch like "Chinese,English", take the first language
    first_lang = qwen_lang.split(',')[0].strip().lower()
    return LANG_NAME_TO_CODE.get(first_lang, 'en')


@except_handler("Qwen3-ASR processing error:")
def transcribe_audio_qwen3(raw_audio_file, vocal_audio_file, start, end) -> dict:
    """
    Transcribe audio using Qwen3-ASR with forced alignment.

    Args:
        raw_audio_file: Path to raw audio file (unused, kept for API compatibility)
        vocal_audio_file: Path to vocal audio file
        start: Start time in seconds (offset)
        end: End time in seconds

    Returns:
        Dict with 'language' and 'segments' matching the expected output contract.
    """
    # Load audio segment at 16kHz mono
    audio, sr = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=end - start, mono=True)

    # Get or create the model singleton
    model = get_model()

    # Prepare optional context (hot words)
    hot_words = load_key("qwen3_asr.hot_words")
    transcribe_kwargs = dict(
        return_time_stamps=True,
        language=None,  # auto-detect
    )
    if hot_words:
        transcribe_kwargs["context"] = hot_words

    # Transcribe
    rprint(f"[green]▶️ Qwen3-ASR transcribing segment {start:.2f}s to {end:.2f}s...[/green]")
    results = model.transcribe(
        audio=(audio, sr),
        **transcribe_kwargs,
    )

    # Qwen3 returns a list; take the first result
    qwen_result = results[0]

    # Map language
    raw_lang = getattr(qwen_result, 'language', None) or ''
    iso_code = map_language(raw_lang)
    rprint(f"[cyan]🌐 Detected language: {raw_lang} -> {iso_code}[/cyan]")

    # Save detected language to config
    update_key("whisper.detected_language", iso_code)
    whisper_lang = load_key("whisper.language")
    if whisper_lang == 'auto':
        update_key("whisper.language", iso_code)

    # Build word list from time_stamps
    time_stamps = getattr(qwen_result, 'time_stamps', [])
    words = []
    for ts in time_stamps:
        word_text = getattr(ts, 'text', '') or ''
        word_start = getattr(ts, 'start_time', None)
        word_end = getattr(ts, 'end_time', None)

        if word_start is not None and word_end is not None:
            # Offset timestamps by chunk start
            word_start += start
            word_end += start

        words.append({
            'word': word_text,
            'start': word_start if word_start is not None else 0.0,
            'end': word_end if word_end is not None else 0.0,
        })

    # Build one segment with all words
    if words:
        segment = {
            'start': words[0]['start'],
            'end': words[-1]['end'],
            'text': getattr(qwen_result, 'text', '') or '',
            'words': words,
        }
    else:
        segment = {
            'start': start,
            'end': end,
            'text': getattr(qwen_result, 'text', '') or '',
            'words': [],
        }

    return {
        'language': iso_code,
        'segments': [segment],
    }
