import re

from pipeline.contracts import ConfigError, StageError

ALIGNER_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}

LANGUAGE_ALIASES = {
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "cantonese": "yue",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}

KANA_RE = re.compile(r"[\u3040-\u30ff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_language(value):
    text = str(value).strip().lower()
    if not text or text == "auto":
        raise ConfigError("asr.language must not be 'auto' here")
    if text in ALIGNER_LANGUAGES:
        return text
    if text in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[text]
    raise ConfigError(
        f"unsupported asr.language: {value}; supported: auto, "
        + ", ".join(sorted(ALIGNER_LANGUAGES))
    )


def _detect(text):
    if KANA_RE.search(text):
        return "ja"
    if HANGUL_RE.search(text):
        return "ko"
    if HAN_RE.search(text):
        return "zh"
    if CYRILLIC_RE.search(text):
        return "ru"
    if LATIN_RE.search(text):
        return "en"
    return None


def detect_language(text):
    language = _detect(text)
    if language is None:
        raise StageError(
            "cannot detect language from the transcript; set asr.language explicitly"
        )
    return language


def resolve_language(configured, texts):
    if configured is not None:
        value = str(configured).strip()
        if value and value.lower() != "auto":
            return normalize_language(value)

    for text in texts:
        language = _detect(text)
        if language is not None:
            return language
    raise StageError(
        "cannot detect language from the transcript; set asr.language explicitly"
    )


def aligner_language(code):
    try:
        return ALIGNER_LANGUAGES[code]
    except KeyError as exc:
        raise ConfigError(f"language is not supported by the forced aligner: {code}") from exc