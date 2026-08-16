import ast
import hashlib
import math
import os
import time
import shutil
import subprocess
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def gen_file_key(number, line_index, line_text):
    """Stable, content-addressed file key for TTS audio segments.

    Includes a short hash of the subtitle text so a stale temp/output file from
    an earlier run can never be reused for different text. Numbers are
    normalized (1.0 -> 1) so pandas dtype differences across xlsx round-trips
    never change a file key. KEEP IN SYNC with the identical helper in
    core/_11_merge_audio.py.
    """
    text_hash = hashlib.md5(str(line_text).encode('utf-8')).hexdigest()[:10]
    try:
        num = int(number) if float(number).is_integer() else number
    except (TypeError, ValueError):
        num = number
    return f"{num}_{line_index}_{text_hash}"

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return
        
    atempo = speed_factor
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = ast.literal_eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(gen_file_key(number, line_index, line))
        tts_main(line, temp_file, number, tasks_df)
        real_dur += get_audio_duration(temp_file)
    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))
        
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e
        
        tts_method = load_key("tts_method")
        if tts_method == "indextts2":
            from core.tts_backend.indextts2_tts import request_concurrency
            max_workers = request_concurrency()
        elif tts_method == "cosyvoice":
            from core.tts_backend.cosyvoice_tts import request_concurrency
            max_workers = request_concurrency()
        else:
            max_workers = load_key("max_workers")
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]❌ Error: {str(e)}[/red]")
                        raise e

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    # A negative final tolerance (overlapping subtitles) already shrinks the
    # row's tol_dur; subtracting it again would inflate the available window.
    durations = tol_durs - max(chunk_df.iloc[-1]['tolerance'], 0.0)
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1

    # Floor denominators so tiny/negative or zero windows never yield an
    # invalid (infinite/nan/negative) speed factor.
    duration_floor = max(durations - speed_var_error, 0.1)
    tol_dur_floor = max(tol_durs - speed_var_error, 0.1)

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = (chunk_durs + all_gaps) / duration_floor
    elif chunk_durs / accept < durations:
        speed_factor = chunk_durs / duration_floor
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = (chunk_durs + all_gaps) / tol_dur_floor
    else:
        speed_factor = chunk_durs / tol_dur_floor
        keep_gaps = False

    # Guard non-finite results, then clamp into [min_speed, 100.0] (ffmpeg's
    # atempo supports up to 100x; anything beyond must not reach ffmpeg).
    if not math.isfinite(speed_factor) or speed_factor <= 0:
        speed_factor = min_speed
    speed_factor = min(max(speed_factor, min_speed), 100.0)
        
    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            # Cast to Python float: np.float64 survives the pandas row lookup, and
            # numpy 2.x serializes it into Excel as `np.float64(...)`, which stage
            # 11's ast.literal_eval cannot parse.
            chunk_end_time = float(parse_df_srt_time(chunk_df.iloc[-1]['end_time'])) + float(chunk_df.iloc[-1]['tolerance']) # 加上tolerance才是这一块的结束
            cur_time = chunk_start_time
            last_audio_file = None
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    # float() keeps cur_time a plain Python float; a pandas
                    # np.float64 here would taint cur_time and, through the
                    # Step-5/Step-6 reassignments, the serialized new_sub_times.
                    cur_time += float(chunk_df.iloc[i-1]['gap'])/speed_factor
                new_sub_times = []
                number = row['number']
                lines = ast.literal_eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    file_key = gen_file_key(number, line_index, line)
                    temp_file = TEMP_FILE_TEMPLATE.format(file_key)
                    output_file = OUTPUT_FILE_TEMPLATE.format(file_key)
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([
                        float(cur_time),
                        float(cur_time + ad_dur),
                    ])
                    cur_time += ad_dur
                    last_audio_file = output_file
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                if len(new_sub_times) != len(lines):
                    raise RuntimeError(
                        f"Row number {row['number']} (chunk row {i}, chunk {chunk_start}-{index}) "
                        f"produced {len(new_sub_times)} subtitle times for {len(lines)} lines"
                    )
                number_idx = tasks_df.index[tasks_df['number'] == row['number']]
                if len(number_idx) > 0:
                    main_df_idx = number_idx[0]
                else:
                    # Fall back to the chunk's positional index inside tasks_df
                    main_df_idx = chunk_start + i
                    rprint(f"[yellow]⚠️ No task row matched number {row['number']}, falling back to its positional row {main_df_idx}[/yellow]")
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Fill the remaining chunk window so the dubbed timeline
            # never silently compresses when TTS audio is shorter than the slot.
            if cur_time < chunk_end_time and last_audio_file:
                remainder = chunk_end_time - cur_time
                audio = AudioSegment.from_wav(last_audio_file)
                padded_audio = audio + AudioSegment.silent(duration=int(round(remainder * 1000)))
                padded_audio.export(last_audio_file, format="wav")
                last_times = tasks_df.at[index, 'new_sub_times']
                if last_times:
                    last_times[-1][1] = float(chunk_end_time)
                    tasks_df.at[index, 'new_sub_times'] = last_times
                cur_time = chunk_end_time
                rprint(f"[cyan]⏱️ Chunk {chunk_start} to {index} was {remainder:.3f}s shorter than its window, filled trailing silence[/cyan]")
            # 🔄 Step6: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= 0.6:  # If exceeding time is within 0.6 seconds, truncate the last audio
                    rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = ast.literal_eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
                    last_line_index = len(last_lines) - 1
                    last_key = gen_file_key(last_number, last_line_index, last_lines[-1])
                    last_file = OUTPUT_FILE_TEMPLATE.format(last_key)
                    
                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                    trimmed_audio.export(last_file, format="wav")
                    
                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    if last_times:
                        last_times[-1][1] = float(chunk_end_time)
                        tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1
    
    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    
    # 🎯 Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)
    
    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    
    # 💾 Step5: Save results
    # Belt-and-braces: force every new_sub_times cell to plain Python floats.
    # Even with the float() casts in merge_chunks, any stray numpy scalar that
    # reaches this point would serialize under numpy 2.x as `np.float64(...)`,
    # which stage 11's ast.literal_eval rejects. Empty cells (None/NaN rows
    # never covered by a chunk) are preserved as-is.
    def _normalize_sub_times(v):
        if v is None:
            return None
        try:
            if math.isnan(v):
                return None
        except TypeError:
            pass  # not a scalar; treat as a list of pairs below
        return [[float(a), float(b)] for a, b in v]

    tasks_df['new_sub_times'] = tasks_df['new_sub_times'].apply(_normalize_sub_times)
    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
