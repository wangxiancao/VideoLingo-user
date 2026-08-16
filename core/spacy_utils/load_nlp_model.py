import re
import spacy
from spacy.cli import download
from core.utils import rprint, load_key, except_handler, get_joiner

SPACY_MODEL_MAP = load_key("spacy_model_map")

# ------------------------
# Language helpers
# ------------------------
# CJK Unified Ideographs (+ Ext A), Hiragana/Katakana, Hangul (syllables + Jamo),
# CJK symbols/punctuation and full-width forms. Used to detect scripts that the
# (often English-only fallback) spacy model cannot segment properly.
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0x3000, 0x303F),
    (0xFF00, 0xFF60),
)


def is_cjk_char(ch):
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def is_cjk_text(text):
    """Return True when the text is dominated by CJK characters."""
    if not text:
        return False
    meaningful = [ch for ch in text if not ch.isspace()]
    if not meaningful:
        return False
    cjk = sum(1 for ch in meaningful if is_cjk_char(ch))
    return (cjk / len(meaningful)) > 0.25


def resolve_language():
    """Working source language for NLP splitting: the real
    whisper.detected_language first, then the configured whisper.language,
    then 'en'. Degrades gracefully if the config keys are missing."""
    for key in ("whisper.detected_language", "whisper.language"):
        try:
            lang = (load_key(key) or "").strip().lower()
        except KeyError:
            continue
        if lang and lang not in ("auto", "unknown", "none"):
            return lang
    return "en"


def get_joiner_safe(language, text=""):
    """Joiner (" " or "") used when joining word-level tokens.

    A detected CJK script in the actual text always wins (""), because a stale
    or missing 'en' language setting in the config must not cause Chinese,
    Japanese or Korean text to be joined with spaces. Otherwise the configured
    split lists decide, falling back to " " for unsupported codes."""
    if text and is_cjk_text(text):
        return ""
    try:
        return get_joiner(language)
    except ValueError:
        pass
    return " "


_CJK_SENT_END_RE = re.compile(r"[。！？…]")


def split_cjk_by_punctuation(text, min_part_len=5):
    """Split CJK text at sentence-ending punctuation (kept attached), used when
    the loaded spacy model cannot segment the CJK script (en fallback)."""
    parts = []
    current = []
    for ch in text:
        current.append(ch)
        if len(current) >= min_part_len and _CJK_SENT_END_RE.fullmatch(ch):
            parts.append("".join(current))
            current = []
    if current:
        tail = "".join(current)
        if tail.strip():
            parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


def get_spacy_model(language: str):
    model = SPACY_MODEL_MAP.get(language.lower(), "en_core_web_md")
    if language.lower() not in SPACY_MODEL_MAP:
        rprint(f"[yellow]Spacy model does not support '{language}', using en_core_web_md model as fallback...[/yellow]")
    return model


@except_handler("Failed to load NLP Spacy model")
def init_nlp():
    language = resolve_language()
    model = get_spacy_model(language)
    rprint(f"[blue]⏳ Loading NLP Spacy model: <{model}> ...[/blue]")
    nlp = None
    try:
        nlp = spacy.load(model)
    except Exception as e:
        rprint(f"[yellow]Failed to load {model}: {e}[/yellow]")
        rprint(f"[yellow]Downloading {model} model...[/yellow]")
        try:
            download(model)
            nlp = spacy.load(model)
        except Exception as e2:
            rprint(f"[yellow]Download/load of {model} failed too ({e2}).[/yellow]")
            nlp = None
    if nlp is None:
        rprint(f"[yellow]Falling back to en_core_web_md for '{language}' text. "
               f"POS/DP-based splitting will be degraded; CJK text will use "
               f"punctuation/character-based splitting instead.[/yellow]")
        nlp = spacy.load("en_core_web_md")
    rprint("[green]✅ NLP Spacy model loaded successfully![/green]")
    return nlp


# --------------------
# define the intermediate files
# --------------------
SPLIT_BY_COMMA_FILE = "output/log/split_by_comma.txt"
SPLIT_BY_CONNECTOR_FILE = "output/log/split_by_connector.txt"
SPLIT_BY_MARK_FILE = "output/log/split_by_mark.txt"