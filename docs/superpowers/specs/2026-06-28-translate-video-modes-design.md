# Translate Video Modes Design

## Goal

Update `translate_video.py` to support exactly two modes:

- `no-tts`: translate and generate subtitles only.
- `tts`: translate first, then generate dubbed audio and final dubbed video.

There is no direct/source-language TTS mode.

## CLI

Supported commands:

```bash
python translate_video.py <video_path_or_url>
python translate_video.py --mode no-tts <video_path_or_url>
python translate_video.py --mode tts <video_path_or_url>
```

The default mode remains `no-tts` for backward compatibility.

## `no-tts` Flow

The existing text/subtitle pipeline remains unchanged:

1. Prepare `output/`.
2. Process input file or URL.
3. Run ASR.
4. Split sentences with NLP and LLM.
5. Summarize and translate.
6. Process and align subtitles.
7. Merge subtitles into video.
8. Archive with `cleanup("history")`.

## `tts` Flow

The `tts` mode always runs the full translation/subtitle pipeline first, then appends the existing dubbing pipeline:

1. Generate audio tasks.
2. Generate dubbing chunks.
3. Extract reference audio.
4. Generate TTS audio.
5. Merge full audio.
6. Merge dubbing into video.
7. Archive with `cleanup("history")`.

If `indextts2.reference_mode` is `videolingo`, `output/audio/refers/1.wav` is produced by the reference-audio step.

## Worker Cleanup

At script exit, `translate_video.py` should stop local optional TTS workers it started. The first implementation targets the IndexTTS2 worker. Cleanup must run on both success and failure in `tts` mode.
