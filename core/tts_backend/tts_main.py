import os
import re
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.cosyvoice_tts import cosyvoice_tts
from core.tts_backend.custom_tts import custom_tts
from core.tts_backend.indextts2_tts import indextts2_tts
from core.prompts import get_correct_text_prompt
from core.utils import *

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def _write_srctext_sidecar(sidecar_file, text):
    """Persist the ACTUAL text synthesized beside the audio file."""
    try:
        with open(sidecar_file, 'w', encoding='utf-8') as f:
            f.write(text)
    except OSError as e:
        rprint(f"[yellow]⚠️ Could not write srctext sidecar {sidecar_file}: {e}[/yellow]")

def _read_srctext_sidecar(sidecar_file):
    """Return the text a previous run recorded for this file, or None."""
    try:
        with open(sidecar_file, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError:
        return None

def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Sidecar records which text was ACTUALLY synthesized (possibly GPT-corrected).
    # The temp path is content-addressed on the ORIGINAL text, so an existing
    # file alone does not prove the cached synthesis matches this run: a prior
    # run may have cached corrected audio under the original-text key. Skip only
    # when the sidecar exists AND equals the current text, i.e. the previous run
    # synthesized this exact (uncorrected) input.
    sidecar_file = f"{save_as}.srctext"

    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        _write_srctext_sidecar(sidecar_file, text)
        return

    # Skip if an identical run already produced this exact text. Files without a
    # matching sidecar (legacy or corrected) are re-synthesized conservatively.
    if os.path.exists(save_as) and _read_srctext_sidecar(sidecar_file) == text:
        return
    
    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")
                # Bypass the prompt cache: the correction must reflect the
                # current text, not a cached response for a different one.
                correct_text = ask_gpt(get_correct_text_prompt(text), resp_type="json", log_title='tts_correct_text', use_cache=False)
                if 'text' not in correct_text or not str(correct_text.get('text', '')).strip():
                    raise ValueError(f"GPT text correction returned an invalid result for: {text}")
                text = correct_text['text']
            if TTS_METHOD == 'custom_tts':
                custom_tts(text, save_as)
            elif TTS_METHOD == 'indextts2':
                indextts2_tts(text, save_as, number, task_df)
            elif TTS_METHOD == 'cosyvoice':
                cosyvoice_tts(text, save_as, number, task_df)
                
            # Check generated audio duration: anything below 0.2s is treated as
            # a failed synthesis (retry, then raise), never as a success.
            duration = get_audio_duration(save_as)
            if duration >= 0.2:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if os.path.exists(sidecar_file):
                    os.remove(sidecar_file)
                if attempt == max_retries - 1:
                    raise Exception(f"Generated audio is too short ({duration:.3f}s < 0.2s) for text: {text}")
                print(f"Attempt {attempt + 1} produced too-short audio ({duration:.3f}s), retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")

    # Record the exact text that was synthesized so a later run can distinguish
    # a pure-function cache hit from corrected audio cached under an original key.
    _write_srctext_sidecar(sidecar_file, text)
