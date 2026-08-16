from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import (
    process_transcription,
    convert_video_to_audio,
    split_audio,
    save_results,
    normalize_audio_volume,
    build_and_save_sentences,
)
from core._1_ytdlp import find_video_files
from core.utils.models import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips (Whisper-large-v3-turbo with word-level timestamps)
    all_results = []
    from core.asr_backend.whisper_asr import transcribe_audio_whisper as ts
    rprint("[cyan]🎤 Transcribing audio with Whisper...[/cyan]")

    for start, end in segments:
        # Isolate per-clip failures so one bad chunk doesn't kill the stage
        try:
            result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        except Exception as e:
            rprint(f"[red]❌ Failed to transcribe clip {start:.2f}s-{end:.2f}s: {e}; skipping clip.[/red]")
            continue
        if result:
            all_results.append(result)
    
    # 5. Combine results
    combined_result = {'segments': []}
    total_words = 0
    for result in all_results:
        for segment in result.get('segments', []):
            combined_result['segments'].append(segment)
            total_words += len(segment.get('words', []))

    if not all_results or total_words == 0:
        raise RuntimeError(
            "ASR produced no text: Whisper returned zero words across all clips."
        )
    
    # 6. Process df
    df = process_transcription(combined_result)
    # save_results filters drops (empty/long words, duplicate degenerate
    # windows) and writes cleaned_chunks.xlsx; reuse the SAME filtered rows for
    # sentences.xlsx so downstream exact-match alignment never diverges.
    df = save_results(df)

    # 7. Build and save sentence-level data
    build_and_save_sentences(df)
        
if __name__ == "__main__":
    transcribe()