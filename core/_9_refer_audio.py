import os
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils import *
from core.utils.models import *
import pandas as pd
import soundfile as sf
from core.asr_backend.demucs_vl import demucs_audio

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Audio extraction, clamped/validated so an empty wav is never written"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    total_samples = len(audio_data)
    # Clamp the requested window into the available audio range
    start = min(max(start, 0), total_samples)
    end = min(max(end, 0), total_samples)
    if end <= start:
        # Invalid/inverted window: fall back to a neighbor 1s window from start
        rprint(f"[yellow]⚠️ Refer window {start_time} -> {end_time} for {os.path.basename(out_file)} is empty, using fallback 1s window[/yellow]")
        start = min(start, total_samples)
        end = min(total_samples, start + sr)
    if end <= start:
        # Still empty (e.g. zero-length source audio): skip instead of writing an empty file
        rprint(f"[yellow]⚠️ Skipping empty refer extraction for {os.path.basename(out_file)}[/yellow]")
        return
    sf.write(out_file, audio_data[start:end], sr)

def extract_refer_audio_main():
    demucs_audio() #!!! in case demucs not run
    if os.path.exists(os.path.join(_AUDIO_REFERS_DIR, '1.wav')):
        rprint(Panel("Audio segments already exist, skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(_8_1_AUDIO_TASK)
    data, sr = sf.read(_VOCAL_AUDIO_FILE)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))
        
        for _, row in df.iterrows():
            out_file = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
            extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {_AUDIO_REFERS_DIR}", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()