import platform
import subprocess

import cv2
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'


def escape_sub_filter_arg(value):
    """Escape a path for use inside an ffmpeg ``subtitles=`` filter argument.

    Per ffmpeg filtergraph rules, backslash, colon, comma and single quote must
    be backslash-escaped so they survive filter argument parsing.
    """
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
    )


def get_media_duration(path):
    """Return media duration in seconds via ffprobe, or None if it cannot be read."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.decode('utf-8', errors='ignore').strip())
    except ValueError:
        return None


def amix_supports_normalize():
    """Report whether this ffmpeg's amix filter accepts the ``normalize`` option.

    ``normalize`` was added in ffmpeg 4.4. Pass ``normalize=0`` on ffmpeg >= 4.4
    so the already-normalized dub keeps full loudness (a plain sum). Older amix
    rejects the option and instead always averages its inputs by dividing with
    the input count, so its output must be compensated with ``volume=<input
    count>`` to match the ``normalize=0`` sum loudness (see merge_video_audio()).

    Returns True when the option is accepted, False when the probe failed and
    the stderr identifies ``normalize`` as an unknown option/argument (an old
    ffmpeg), and None when the probe failed for an unrelated reason. On None the
    caller must NOT guess the legacy volume compensation (it would double the
    loudness on a modern build whose probe merely failed); the real merge then
    fails fast with a clear stderr if ``normalize`` is truly unsupported.
    """
    probe = subprocess.run(
        ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'anoisesrc=d=0.1:r=8000',
         '-f', 'lavfi', '-i', 'anoisesrc=d=0.1:r=8000',
         '-filter_complex', '[0:a][1:a]amix=inputs=2:normalize=0[a]',
         '-map', '[a]', '-f', 'null', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if probe.returncode == 0:
        return True
    stderr_text = probe.stderr.decode('utf-8', errors='ignore').lower()
    unknown_markers = ("not found", "invalid", "unknown", "unrecognized",
                       "no such option")
    if "normalize" in stderr_text and any(m in stderr_text for m in unknown_markers):
        # Old amix: the option itself is rejected (e.g. 4.2.7 prints
        # "Option 'normalize' not found" / "Error initializing filter 'amix'").
        return False
    rprint(
        f"[bold yellow]Warning: amix normalize probe failed with exit code "
        f"{probe.returncode} and no unknown-option error; skipping the legacy "
        f"volume compensation. If your ffmpeg truly lacks 'normalize' the merge "
        f"below will fail with a clear error instead of doubling loudness.[/bold yellow]"
    )
    return None


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    burn_subtitles = load_key("burn_subtitles")

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    # Resolution and dimensions always come from the source video. The real video
    # is always produced; the subtitles= filter is appended only when burning.
    video_chain = (
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2'
    )
    if burn_subtitles:
        video_chain += (
            f",subtitles={escape_sub_filter_arg(DUB_SUB_FILE)}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
    video_chain += '[v]'

    # Pad a short dub so the mixed track covers the whole video (no silent tail),
    # and mix the already-normalized dub without reapplying amix attenuation.
    # amix termination semantics: use a FINITE pad and duration=longest; an
    # unlimited apad + duration=longest never reaches EOF and hangs the mux.
    # Loudness: on ffmpeg >= 4.4 amix runs with normalize=0 (plain sum); older
    # amix always averages inputs, so the compensation below is applied there.
    video_duration = get_media_duration(VIDEO_FILE)
    dub_duration = get_media_duration(normalized_dub_audio)
    if video_duration and dub_duration:
        pad_seconds = max(0.0, video_duration - dub_duration)
    elif video_duration:
        # Dub duration unknown: apad's pad_dur APPENDS its argument to the
        # input (it does not top up to a total), so pad by the whole video
        # length and rely on the unconditional atrim below to clamp.
        pad_seconds = video_duration
    else:
        # Video duration unknown: no finite pad can be trusted; atrim below is
        # skipped, so the -shortest output option is the only backstop for a
        # dub that overruns the video end.
        pad_seconds = 0.0

    # amix normalization: ffmpeg >= 4.4 can ask for normalize=0 (plain sum).
    # Older amix has no normalize option and always averages by dividing with the
    # input count, so compensate with volume=<number of inputs> (2 here).
    amix_inputs = 2
    amix = f'amix=inputs={amix_inputs}:duration=longest:dropout_transition=3'
    normalize_support = amix_supports_normalize()
    if normalize_support is True:
        amix += ':normalize=0'
    elif normalize_support is False:
        amix += f',volume={amix_inputs}.0'
    # None (probe failed for an unrelated reason): keep plain amix, never guess.
    # If `normalize` is genuinely unsupported the mux fails fast below with a
    # clear ffmpeg stderr; if it is supported, amix's default normalize=1 only
    # mirrors the legacy averaging behavior (softer dub, no doubling).

    # Only pad when the dub is actually shorter. On older ffmpeg `apad=pad_dur=0`
    # is treated as "no pad duration" and appends infinite silence, which makes
    # duration=longest run forever, so skip the apad branch when there is no need.
    if pad_seconds > 0.0:
        audio_filters = f'[2:a]apad=pad_dur={pad_seconds:.3f}[a2];[1:a][a2]{amix}'
    else:
        # A dub longer than the video must not let duration=longest let the mixed
        # audio overrun the video end: clamp the mixed audio to the video length.
        audio_filters = f'[1:a][2:a]{amix}'
    # Clamp the mix to the video length whenever it is known. With exact known
    # durations this is a no-op; when a duration probe failed it fixes both the
    # pad branch (apad-append overrun) and the no-pad branch (dub longer than
    # the video). Without a known video duration -shortest is the only backstop.
    if video_duration:
        audio_filters += f',atrim=end={video_duration:.3f}'
    audio_filters += '[a]'
    filter_complex = f'{video_chain};{audio_filters}'

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex', filter_complex,
    ]

    # When either duration probe failed the pad/atrim logic may not bound the
    # audio to the video end, so add -shortest as a backup mux bound. Verified
    # on ffmpeg 4.2.7 with synthetic 3s clips: it does NOT truncate the video
    # stream early in the equal-length case, though on 4.2.7 it is a no-op for
    # filtergraph audio (newer builds apply it), hence the atrim above.
    if not (video_duration and dub_duration):
        cmd.extend(['-shortest'])
        if video_duration is None:
            rprint(
                "[bold yellow]Warning: video duration unknown; cannot clamp the "
                "mixed audio with atrim. Relying on -shortest to bound the "
                "output — on ffmpeg builds that only apply it to unfiltered "
                "streams the dubbing may overrun the video end.[/bold yellow]"
            )

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr_tail = result.stderr.decode('utf-8', errors='ignore')[-500:]
        raise RuntimeError(
            f"FFmpeg failed to merge video and audio (exit code {result.returncode}):\n{stderr_tail}"
        )
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

    # Verify the mixed audio length matches the video within tolerance in either
    # direction: too short leaves a silent tail, too long overruns the video end.
    mixed_duration = get_media_duration(DUB_VIDEO)
    if video_duration and mixed_duration and abs(mixed_duration - video_duration) > 0.5:
        relation = "shorter" if mixed_duration < video_duration else "longer"
        rprint(
            f"[bold yellow]Warning: output audio duration ({mixed_duration:.2f}s) is {relation} "
            f"than video duration ({video_duration:.2f}s) — the audio and video tails "
            f"may be misaligned.[/bold yellow]"
        )

if __name__ == '__main__':
    merge_video_audio()