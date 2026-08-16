import os, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import platform
from core.utils import *

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"

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

def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    burn_subtitles = load_key("burn_subtitles")

    if burn_subtitles and (not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT)):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    # Resolution must come from the source video, never hardcoded.
    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    # Always produce the real video; the subtitles= filters are only appended
    # when burning. The original audio stream is kept (default stream selection).
    vf_parts = [
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
    ]
    if burn_subtitles:
        vf_parts.append(
            f"subtitles={escape_sub_filter_arg(SRC_SRT)}:force_style='FontSize={SRC_FONT_SIZE},FontName={FONT_NAME},"
            f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1'"
        )
        vf_parts.append(
            f"subtitles={escape_sub_filter_arg(TRANS_SRT)}:force_style='FontSize={TRANS_FONT_SIZE},FontName={TRANS_FONT_NAME},"
            f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )

    ffmpeg_cmd = ['ffmpeg', '-i', video_file, '-vf', ','.join(vf_parts)]

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        _, stderr = process.communicate()
        if process.returncode != 0:
            stderr_tail = stderr.decode('utf-8', errors='ignore')[-500:]
            raise RuntimeError(
                f"FFmpeg failed with exit code {process.returncode}:\n{stderr_tail}"
            )
    except Exception as e:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise

    rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    merge_subtitles_to_video()