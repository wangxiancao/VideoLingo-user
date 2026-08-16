import ast
import hashlib
import os
import pandas as pd
import subprocess
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.utils import *
from core.utils.models import *
console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

def gen_file_key(number, line_index, line_text):
    """Stable, content-addressed file key for TTS audio segments.

    MUST stay in sync with the identical helper in core/_10_gen_audio.py so the
    segment files written there are found here. Numbers are normalized
    (1.0 -> 1) so pandas dtype differences never change a file key.
    """
    text_hash = hashlib.md5(str(line_text).encode('utf-8')).hexdigest()[:10]
    try:
        num = int(number) if float(number).is_integer() else number
    except (TypeError, ValueError):
        num = number
    return f"{num}_{line_index}_{text_hash}"

def load_and_flatten_data(excel_file):
    """Load and flatten Excel data"""
    df = pd.read_excel(excel_file)
    lines = [ast.literal_eval(line) if isinstance(line, str) else line for line in df['lines'].tolist()]
    lines = [item for sublist in lines for item in sublist]
    
    new_sub_times = []
    for idx, time in enumerate(df['new_sub_times'].tolist()):
        if isinstance(time, str):
            try:
                parsed = ast.literal_eval(time)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Row {idx} (number={df['number'].iloc[idx]!r}) has an "
                    f"unparseable new_sub_times cell: {time!r}. Numpy scalars in "
                    f"the cell (numpy 2.x serializes them as 'np.float64(...)') "
                    f"indicate stage 10 produced a non-float value; re-run the "
                    f"dubbing pipeline from that stage."
                ) from e
        else:
            parsed = time
        new_sub_times.append(parsed)
    new_sub_times = [item for sublist in new_sub_times for item in sublist]
    
    return df, lines, new_sub_times

def get_audio_files(df):
    """Generate a list of audio file paths (content-addressed, matching _10)"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        lines = ast.literal_eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        for line_index, line in enumerate(lines):
            audios.append(OUTPUT_FILE_TEMPLATE.format(gen_file_key(number, line_index, line)))
    return audios

def process_audio_segment(audio_file):
    """Process a single audio segment with MP3 compression"""
    temp_file = f"{audio_file}_temp.mp3"
    # Remove a leftover temp file from a crashed earlier attempt so a retry can
    # never silently mix stale audio into the dub.
    if os.path.exists(temp_file):
        os.remove(temp_file)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file,
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '64k',
        temp_file
    ]
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_tail = result.stderr.decode('utf-8', errors='ignore')[-500:]
        raise RuntimeError(
            f"FFmpeg failed to compress audio segment {audio_file} "
            f"(exit code {result.returncode}):\n{stderr_tail}"
        )
    audio_segment = AudioSegment.from_mp3(temp_file)
    os.remove(temp_file)
    return audio_segment

def merge_audio_segments(audios, new_sub_times, sample_rate):
    merged_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))
        
        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            if not os.path.exists(audio_file):
                console.print(f"[bold yellow]⚠️  Warning: File {audio_file} does not exist, skipping...[/bold yellow]")
                progress.advance(merge_task)
                continue
                
            audio_segment = process_audio_segment(audio_file)
            start_time, end_time = time_range
            
            # Fit the segment exactly into its [start, end] slot before appending:
            # truncate if the audio is longer than the slot, pad with silence if
            # shorter. Using round() (not int-truncation) prevents cumulative drift.
            slot_duration = max(float(end_time) - float(start_time), 0.0)
            segment_duration = len(audio_segment) / 1000.0
            if segment_duration > slot_duration:
                audio_segment = audio_segment[:int(round(slot_duration * 1000))]
            elif segment_duration < slot_duration:
                audio_segment = audio_segment + AudioSegment.silent(
                    duration=int(round((slot_duration - segment_duration) * 1000)),
                    frame_rate=sample_rate,
                )
            
            # Pad from the current end of the assembled audio up to the slot start
            current_duration = len(merged_audio) / 1000.0
            if float(start_time) > current_duration:
                merged_audio += AudioSegment.silent(
                    duration=int(round((float(start_time) - current_duration) * 1000)),
                    frame_rate=sample_rate,
                )
            merged_audio += audio_segment
            progress.advance(merge_task)
    
    return merged_audio

def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    
    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
    
    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")

def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")
    
    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    console.print("[bold green]✅ Data loaded successfully[/bold green]")
    
    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")
    
    with console.status("[bold cyan]📝 Generating subtitle file...[/bold cyan]"):
        create_srt_subtitle()
    
    if not audios:
        raise RuntimeError("No audio segments were generated (empty audio list); cannot merge the dub track")
    if not os.path.exists(audios[0]):
        raise RuntimeError(
            f"First audio file {audios[0]} does not exist; TTS generation likely "
            "produced no usable segments. Refusing to merge a silent/stale dub."
        )
    
    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)
    
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(DUB_VOCAL_FILE, format="mp3", parameters=["-b:a", "64k"])
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()
