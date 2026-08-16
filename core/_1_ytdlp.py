import os,sys
import glob
import re
import shutil
import subprocess
from core.utils import *

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best' if resolution == 'best' else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
    }

    # yt-dlp only enables deno by default; YouTube extraction needs a JS runtime
    # for full format/signature support, otherwise downloads can 403/throttle.
    if shutil.which("node"):
        ydl_opts["js_runtimes"] = {"node": {}}

    # Read Youtube Cookie File
    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        # Datacenter IPs often get HTTP 403 from the web player client (PO-token
        # bot detection). Fall back to mobile player clients, which trade some
        # bitrate for reliability.
        rprint(f"[yellow]Default YouTube clients failed ({e}); retrying with mobile player clients[/yellow]")
        ydl_opts["extractor_args"] = {"youtube": {"player_client": ["android", "ios", "mweb"]}}
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    # Exclude only the pipeline's own generated videos (output/output_sub.mp4 and
    # output/output_dub.mp4), which live directly inside the output directory.
    # Compare resolved paths so relative/absolute (and OS) path forms all match,
    # instead of filtering by an "output/output" name prefix that would also hide
    # a legitimate source file named e.g. output_final.mp4.
    real_output_dir = os.path.normcase(os.path.realpath(save_path))
    pipeline_outputs = {
        os.path.join(real_output_dir, name)
        for name in ("output_sub.mp4", "output_dub.mp4")
    }
    video_files = [
        file for file in video_files
        if os.path.normcase(os.path.realpath(file)) not in pipeline_outputs
    ]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
