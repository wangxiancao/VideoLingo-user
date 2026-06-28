# IndexTTS2 Worker Design

## Goal

Add `IndexTeam/IndexTTS-2` as an optional local VideoLingo dubbing backend named `indextts2`. The backend must only start when a user selects the IndexTTS2 TTS method and begins a dubbing task. It must not install packages into `/root/soft/anaconda3/envs/videolingo`.

## Architecture

IndexTTS2 will run as an isolated local HTTP worker. VideoLingo remains in its existing runtime and never imports IndexTTS modules directly.

Components:

- `external/index-tts`: official IndexTTS repository, managed as a git submodule.
- `external/index-tts/.venv`: isolated `uv` environment created by IndexTTS.
- `external/index-tts/checkpoints`: downloaded `IndexTeam/IndexTTS-2` model files.
- `scripts/indextts2_worker.py`: VideoLingo-owned HTTP worker script.
- `core/tts_backend/indextts2_tts.py`: VideoLingo TTS adapter.
- `config.yaml`: `indextts2` configuration block and optional default selection.
- `core/st_utils/sidebar_setting.py`: adds `indextts2` to the TTS selector.

## Configuration

Add this configuration block:

```yaml
indextts2:
  repo_dir: 'external/index-tts'
  host: '127.0.0.1'
  port: 18721
  reference_audio: 'output/audio/refers/1.wav'
  use_fp16: true
  use_cuda_kernel: false
  use_deepspeed: false
  startup_timeout: 180
  request_timeout: 300
```

## Data Flow

When `tts_method` is `indextts2`, `tts_main()` dispatches to the new adapter. The adapter checks `http://127.0.0.1:18721/health`. If no healthy worker exists, it starts one with `uv run` from `external/index-tts`.

The worker loads one `IndexTTS2` instance:

```python
IndexTTS2(
    cfg_path='checkpoints/config.yaml',
    model_dir='checkpoints',
    use_fp16=use_fp16,
    use_cuda_kernel=use_cuda_kernel,
    use_deepspeed=use_deepspeed,
)
```

For each subtitle line, the adapter sends `text`, `output_path`, and `spk_audio_prompt='output/audio/refers/1.wav'` to `/synthesize`. The worker writes the WAV file directly to `output_path`. VideoLingo then keeps its existing duration check, retry, speed adjustment, and merge behavior.

If the reference audio is missing, the adapter calls `extract_refer_audio_main()` to create `output/audio/refers/1.wav`. If extraction fails, the adapter raises a clear error and does not silently fall back to cloud TTS.

## Installation

Implementation will add the official repository as a submodule:

```bash
git submodule add https://github.com/index-tts/index-tts.git external/index-tts
```

Then, inside `external/index-tts`, use the upstream-supported setup:

```bash
uv sync --all-extras
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

## Worker API

Required endpoints:

- `GET /health`: returns readiness, model-loaded state, and basic configuration.
- `POST /synthesize`: accepts text, reference audio path, and output path; returns success or structured error details.

The worker should expose a minimal web/HTTP surface so the user can manually verify it after startup at `http://127.0.0.1:18721`.

## Verification

Minimum automated verification:

- Start the worker.
- Confirm `/health` succeeds.
- Synthesize one short sentence to a temporary WAV file.
- Confirm VideoLingo can read a non-zero audio duration.

Manual verification:

- Leave the worker running after the minimum check.
- Report the local URL so the user can inspect or call the worker manually.

Out of scope:

- Full video translation.
- Full-video dubbing generation.
- Subjective voice-quality evaluation.

## Risks

IndexTTS officially supports `uv`; non-`uv` dependency layouts are out of scope. First setup requires network access and large model downloads. Startup may be slow because the model loads once into the worker. CUDA, VRAM, or custom kernel issues may require changing `use_fp16`, `use_cuda_kernel`, or `use_deepspeed`.
