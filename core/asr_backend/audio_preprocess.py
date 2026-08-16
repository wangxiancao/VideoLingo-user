import os, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from core.spacy_utils.load_nlp_model import is_cjk_text
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint

# Whisper word-level DTW groups whole clauses into single CJK "word" chunks
# (31-40 chars is normal for Chinese), so the generic 30-char cap must not
# apply to CJK. Worse, long-form Whisper merges CONTINUOUS speech into giant
# single CJK chunks (hundreds of chars across many seconds, one timestamp
# pair) which used to be dropped wholesale. Such chunks are now sub-split
# into clause-sized rows on CJK punctuation instead of being dropped.
MAX_NON_CJK_WORD_LEN = 30
# A CJK word chunk longer than this (chars) or spanning more than this many
# seconds is sub-split into rows of at most MAX_WORD_CHARS characters, with
# the chunk's [start, end] distributed proportionally by character count.
MAX_WORD_CHARS = 60
MAX_LONG_WORD_SPAN = 15.0

# CJK punctuation used when sub-splitting giant Whisper word chunks:
# sentence-ending marks (kept attached to their clause) and clause-level
# separators (also kept attached).
_CJK_SENT_END_CHARS = "。！？；…"
_CJK_CLAUSE_CHARS = "，、"
_CJK_EMPHASIS_CHARS = "！？"


def _split_cjk_by_sentence(text):
    """Split CJK text at sentence-ending punctuation, keeping it attached.

    A leading emphasis mark run such as "？" immediately followed by "！" (or
    vice versa) is kept as one unit attached to the clause."""
    parts = []
    current = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        current.append(ch)
        if ch in _CJK_SENT_END_CHARS:
            # Keep a paired emphasis mark ("？！" / "！？") on the same clause.
            if ch in _CJK_EMPHASIS_CHARS and i + 1 < n and text[i + 1] in _CJK_EMPHASIS_CHARS:
                current.append(text[i + 1])
                i += 1
            parts.append("".join(current))
            current = []
        i += 1
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _sub_split_cjk_part(part, max_chars):
    """Split a CJK part that exceeds max_chars on clause separators (，、),
    keeping the separators attached; hard-chunk any piece still too long."""
    if len(part) <= max_chars:
        return [part]
    pieces = []
    current = []
    for ch in part:
        current.append(ch)
        if ch in _CJK_CLAUSE_CHARS:
            pieces.append("".join(current))
            current = []
    if current:
        pieces.append("".join(current))
    out = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
    return [p for p in out if p.strip()]


def _split_long_cjk_word(text, max_chars=MAX_WORD_CHARS):
    """Split a giant CJK word chunk into pieces of at most max_chars characters.

    Prefers sentence-ending punctuation boundaries, then clause separators,
    then hard character chunks — so each piece is a plausible subtitle unit
    and never exceeds the character cap."""
    parts = []
    for clause in _split_cjk_by_sentence(text):
        parts.extend(_sub_split_cjk_part(clause, max_chars))
    return parts


def _subdivide_cjk_word(text, start, end, max_chars=MAX_WORD_CHARS):
    """Split a giant CJK word chunk into (text, start, end) sub-rows.

    The chunk's [start, end] span is distributed across the pieces
    proportionally to each piece's character count, so the sub-rows are
    monotonic, contiguous and together cover exactly the original span.

    Zero/negative spans are returned unsplit: distributing a collapsed window
    would create identical zero-width rows that the downstream duplicate
    filter would then drop (losing text the old code kept).
    """
    span = float(end) - float(start)
    if span <= 0:
        return [(text, float(start), float(end))]
    pieces = _split_long_cjk_word(text, max_chars)
    total_chars = sum(len(p) for p in pieces)
    if total_chars == 0:
        return [(text, float(start), float(end))]
    rows = []
    acc = 0
    for piece in pieces:
        n_chars = len(piece)
        piece_start = float(start) + span * (acc / total_chars)
        acc += n_chars
        piece_end = float(start) + span * (acc / total_chars)
        rows.append((piece, piece_start, piece_end))
    return rows

def normalize_audio_volume(audio_path, output_path, target_db = -20.0, format = "wav"):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format=format)
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def convert_video_to_audio(video_file: str):
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(_RAW_AUDIO_FILE):
        rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-ar', '16000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: float = 30*60, win: float = 60) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len+win] 区间内用 pydub 检测静默，切分音频
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        return [(0, duration)]
    segments, pos = [], 0.0
    safe_margin = 0.5  # 静默点前后安全边界，单位秒

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration)); break

        threshold = pos + target_len
        ws, we = int((threshold - win) * 1000), int((threshold + win) * 1000)
        
        # 获取完整的静默区域
        silence_regions = detect_silence(audio[ws:we], min_silence_len=int(safe_margin*1000), silence_thresh=-30)
        silence_regions = [(s/1000 + (threshold - win), e/1000 + (threshold - win)) for s, e in silence_regions]
        # 筛选长度足够（至少1秒）且位置适合的静默区域
        valid_regions = [
            (start, end) for start, end in silence_regions 
            if (end - start) >= (safe_margin * 2) and threshold <= start + safe_margin <= threshold + win
        ]
        
        if valid_regions:
            start, end = valid_regions[0]
            split_at = start + safe_margin  # 在静默区域起始点后0.5秒处切分
        else:
            rprint(f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]")
            split_at = threshold
            
        segments.append((pos, split_at)); pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)

        for word in segment['words']:
            # Clean the word text before validation
            # ! For French, we need to convert guillemets to empty strings
            cleaned = word["word"].replace('»', '').replace('«', '').strip()
            if not cleaned:
                continue

            # Check word length: non-CJK words longer than 30 characters are
            # Whisper garbage (hallucinated/joined filler) — drop them. CJK
            # chunks are NEVER dropped for length; giant ones are sub-split
            # into clause-sized rows below.
            if len(cleaned) > MAX_NON_CJK_WORD_LEN and not is_cjk_text(cleaned):
                rprint(f"[yellow]⚠️ Warning: Detected non-CJK word longer than {MAX_NON_CJK_WORD_LEN} characters, skipping: {cleaned}[/yellow]")
                continue

            # Resolve timestamps before any collapse decision so the
            # degenerate-window check compares real values.
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    w_start = w_end = all_words[-1]['end']
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        w_start, w_end = next_word["start"], next_word["end"]
                    else:
                        # Don't abort the whole transcription because of one bad chunk
                        rprint(f"[yellow]⚠️ Warning: No next word with timestamp found for word {cleaned}, skipping it.[/yellow]")
                        continue
            else:
                # Normal case, with start and end times
                w_start = word.get('start', all_words[-1]['end'] if all_words else 0)
                w_end = word['end']

            # Whisper long-form merges CONTINUOUS speech into giant single CJK
            # word chunks (hundreds of chars, one timestamp pair spanning many
            # seconds). Sub-split such chunks into clause-sized rows on CJK
            # punctuation; the chunk's [start, end] is distributed
            # proportionally by character count so the whole span stays
            # covered. Each sub-row flows through the same degenerate-window
            # collapse check as a normal word.
            if is_cjk_text(cleaned) and (
                len(cleaned) > MAX_WORD_CHARS
                or (float(w_end) - float(w_start)) > MAX_LONG_WORD_SPAN
            ):
                sub_rows = _subdivide_cjk_word(cleaned, w_start, w_end)
            else:
                sub_rows = [(cleaned, w_start, w_end)]

            for sub_text, sub_start, sub_end in sub_rows:
                # Collapse consecutive identical words ONLY when their
                # timestamp window is degenerate (end does not advance past
                # the previous kept word, or the exact same window). Whisper
                # hallucination loops repeat a word on a collapsed/receding
                # timestamp, while legitimate repeats ("好的好的", "no no")
                # advance the window and must be preserved.
                if all_words:
                    prev = all_words[-1]
                    if sub_text == prev['text'] and sub_end <= prev['end']:
                        rprint(f"[yellow]⚠️ Skipping repeated consecutive word with degenerate window: '{sub_text}'[/yellow]")
                        continue

                word_dict = {
                    'text': sub_text,
                    'start': sub_start,
                    'end': sub_end,
                    'speaker_id': speaker_id,
                }
                all_words.append(word_dict)

    df = pd.DataFrame(all_words)
    df.insert(0, 'word_id', range(len(df)))
    return df

def save_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter `df` (empty/long words, duplicate degenerate windows), write the
    cleaned result to _2_CLEANED_CHUNKS, and return the FILTERED frame so the
    caller builds sentences.xlsx from exactly the same rows it wrote."""
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")

    # Check for and remove words longer than 30 characters. CJK chunks were
    # already sub-split into sane clause-sized rows by process_transcription,
    # so only non-CJK garbage words are dropped here (the length rule is
    # language-aware).
    long_words = df[df['text'].str.len() > MAX_NON_CJK_WORD_LEN]
    if not long_words.empty:
        non_cjk_flags = ~df['text'].apply(is_cjk_text)
        non_cjk_long = df[non_cjk_flags & (df['text'].str.len() > MAX_NON_CJK_WORD_LEN)]
        if not non_cjk_long.empty:
            rprint(f"[yellow]⚠️ Warning: Detected {len(non_cjk_long)} non-CJK word(s) longer than {MAX_NON_CJK_WORD_LEN} characters. These will be removed.[/yellow]")
            df = df[~non_cjk_flags | (df['text'].str.len() <= MAX_NON_CJK_WORD_LEN)]

    # Hallucination/repetition filtering (A4):
    #  - drop consecutive rows that share the exact same (start, end) window
    #  - drop zero/negative-width rows that merely repeat the previous text
    df['start'] = pd.to_numeric(df['start'], errors='coerce').fillna(0.0)
    df['end'] = pd.to_numeric(df['end'], errors='coerce').fillna(0.0)
    keep_flags = []
    prev_start = prev_end = prev_text = None
    for _, row in df.iterrows():
        start, end = float(row['start']), float(row['end'])
        text = str(row['text'])
        duplicate = (
            prev_start is not None
            and (
                (start == prev_start and end == prev_end)
                or (end <= start and text == prev_text)
            )
        )
        keep_flags.append(not duplicate)
        if not duplicate:
            prev_start, prev_end, prev_text = start, end, text
    dropped = sum(1 for k in keep_flags if not k)
    if dropped:
        rprint(f"[yellow]⚠️ Dropped {dropped} row(s) with duplicate/zero-width timestamps.[/yellow]")
        df = df.iloc[[i for i, k in enumerate(keep_flags) if k]]

    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]📊 Excel file saved to {_2_CLEANED_CHUNKS}[/green]")
    return df

# Max time span (seconds) allowed for one sentence produced in stage 2; longer
# spans are re-split at word boundaries so downstream NLP/LLM splitting stages
# don't receive multi-minute "sentences".
MAX_SENTENCE_SPAN = 15.0

def _append_sentence(sentences, subset):
    sentences.append({
        'sentence_id': len(sentences),
        'text': ' '.join(subset['text'].str.strip('"').str.strip().tolist()),
        'start_word_id': int(subset.iloc[0]['word_id']),
        'end_word_id': int(subset.iloc[-1]['word_id']),
        'start': float(subset.iloc[0]['start']),
        'end': float(subset.iloc[-1]['end']),
    })

def _append_sentences_span_limited(sentences, subset, max_span=MAX_SENTENCE_SPAN):
    """Append `subset` as sentences, re-splitting at word boundaries (greedy
    cumulative-time distribution) whenever one sentence would span more than
    `max_span` seconds. Keeps the xlsx sentence schema unchanged."""
    subset = subset.reset_index(drop=True)
    if len(subset) == 0:
        return
    span = float(subset.iloc[-1]['end']) - float(subset.iloc[0]['start'])
    if len(subset) <= 1 or span <= max_span:
        _append_sentence(sentences, subset)
        return
    group_start = 0
    for i in range(1, len(subset) + 1):
        if i == len(subset):
            _append_sentence(sentences, subset.iloc[group_start:i])
            break
        new_span = float(subset.iloc[i]['end']) - float(subset.iloc[group_start]['start'])
        if new_span > max_span:
            _append_sentence(sentences, subset.iloc[group_start:i])
            group_start = i

def build_and_save_sentences(df):
    """Group words into sentences by sentence-ending punctuation, save to _2_SENTENCES."""
    import re
    sentences = []
    sent_start_idx = 0
    sent_end_puncts = re.compile(r'[.!?。！？…]')

    for idx, row in df.iterrows():
        text = str(row['text']).strip().strip('"').strip()
        if sent_end_puncts.search(text):
            subset = df.iloc[sent_start_idx:idx+1]
            _append_sentences_span_limited(sentences, subset)
            sent_start_idx = idx + 1

    # Handle remaining words after last punctuation
    if sent_start_idx < len(df):
        subset = df.iloc[sent_start_idx:]
        _append_sentences_span_limited(sentences, subset)

    import os
    os.makedirs('output/log', exist_ok=True)
    pd.DataFrame(sentences).to_excel(_2_SENTENCES, index=False)
    rprint(f"[green]📝 Sentences saved to {_2_SENTENCES}[/green]")

