import pandas as pd
import os
import re
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

# Forward-scan window used when a matched sentence's ASR word timestamps are
# degenerate (end <= start): only a nearby word may extend the cue, so a single
# subtitle never jumps across an arbitrarily long silent span.
MAX_SCAN_WORDS = 8
MAX_EXTEND_SECONDS = 3.0

# A single ASR word chunk spanning more than this (seconds) is treated as a
# wide word (Whisper long-form merges continuous speech into giant CJK chunks
# with one timestamp pair). Legacy constant kept for reference; per-sentence
# cues are now derived from word-relative char positions, which naturally
# distributes multiple sentences inside one wide word without special pooling.
WIDE_MATCHED_WORD_SECONDS = 10.0

# Sane per-line duration cap, consistent with the ASR-side MAX_SENTENCE_SPAN.
# Longer cues are clamped before writing SRTs.
MAX_SUBTITLE_DURATION = 15.0

# Minimal positive duration kept when clamping fully-collapsed overlapping rows.
MIN_DURATION_EPS = 0.1

# get_sentence_timestamps returns a list of (start, end) tuples. After the
# wide-word proportional distribution, a sentence may still resolve earlier
# than the row before it (e.g. it starts inside a wide word whose earlier part
# was already consumed by a grouped run, then spills into the next word). The
# resolver enforces monotonicity by construction before normalize_timestamps:
# each row starts at least at the previous row's end, and keeps positive width.
def _enforce_monotonic_timestamps(stamps):
    """Return a monotonic copy of (start, end) cues.

    Defensive floor against pathological overlapping/regressing word
    timestamps: start_i = max(start_i, start_{i-1}) so a cue never starts
    BEFORE the one before it (validate_positive_timestamps rejects strictly
    regressing starts), and every cue keeps positive width. Equal starts are
    preserved (legitimate repeated/degenerate cues share the same point; the
    overlap clamp in normalize_timestamps resolves them later). With
    word-relative char-timing the resolver output is already ordered, so this
    pass only ever nudges genuinely out-of-order rows.
    """
    out = []
    prev_start = None
    for start, end in stamps:
        start = float(start)
        end = float(end)
        if prev_start is not None and start < prev_start:
            start = prev_start
        if end <= start:
            end = start + MIN_DURATION_EPS
        out.append((start, end))
        prev_start = start
    return out

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings."""
    min_len = min(len(str1), len(str2))
    diff_positions = []

    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)

    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))

    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print(
        "Position markers: "
        + "".join(
            "^" if i in diff_positions else " "
            for i in range(max(len(str1), len(str2)))
        )
    )
    print(f"Difference indices: {diff_positions}")

def get_sentence_timestamps(df_words, df_sentences):
    """Match normalized source text exactly against the sequential ASR character stream."""
    full_words_str = ''
    position_to_word_idx = {}
    word_char_start = []
    word_char_end = []

    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(word.lower())
        word_char_start.append(len(full_words_str))
        full_words_str += clean_word
        word_char_end.append(len(full_words_str))
        for pos in range(word_char_start[idx], word_char_end[idx]):
            position_to_word_idx[pos] = idx

    def char_time(pos):
        """Absolute time at character index `pos` (word-relative, linear).

        Each character range of a word maps linearly onto that word's
        [start, end], so a sentence that only covers PART of a word — e.g. a
        sentence starting in the tail of one giant CJK chunk and spilling
        into the next — receives a proportional slice instead of snapping to
        the word's absolute start (which previously rewound past already
        distributed cues and produced timestamp regressions)."""
        if not position_to_word_idx:
            return 0.0
        if pos <= 0:
            widx = position_to_word_idx[0]
            frac = 0.0
        elif pos >= len(full_words_str):
            widx = position_to_word_idx[len(full_words_str) - 1]
            frac = 1.0
        else:
            widx = position_to_word_idx[pos]
            wl = float(word_char_end[widx] - word_char_start[widx])
            frac = (pos - word_char_start[widx]) / wl if wl > 0 else 0.0
        ts = float(df_words['start'][widx])
        te = float(df_words['end'][widx])
        return ts + (te - ts) * frac

    def char_end_time(ce):
        """Absolute time at the trailing boundary of the LAST matched char."""
        if ce <= 0 or not position_to_word_idx:
            return char_time(0)
        widx = position_to_word_idx[min(ce - 1, len(full_words_str) - 1)]
        wl = float(word_char_end[widx] - word_char_start[widx])
        frac = (min(ce, word_char_end[widx]) - word_char_start[widx]) / wl if wl > 0 else 1.0
        ts = float(df_words['start'][widx])
        te = float(df_words['end'][widx])
        return ts + (te - ts) * frac

    fallback_duration = 3.0
    try:
        fallback_duration = float(load_key("min_subtitle_duration"))
    except (KeyError, TypeError, ValueError):
        pass  # keep the 3s default when the config key is missing

    time_stamp_list = []
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        # Guard against stage-5 schema padding / empty rows: `None` (or NaN)
        # crashes str(None).lower(), and empty strings loop forever. Skip them.
        if pd.isna(sentence):
            console.print(f"[yellow]⚠️ Skipping subtitle row {idx}: no source text.[/yellow]")
            continue
        sentence = str(sentence)
        clean_sentence = remove_punctuation(sentence.lower()).replace(" ", "")
        sentence_len = len(clean_sentence)
        if sentence_len == 0:
            console.print(f"[yellow]⚠️ Skipping subtitle row {idx}: no source text.[/yellow]")
            continue

        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos:current_pos + sentence_len] == clean_sentence:
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                start_time = char_time(current_pos)
                end_time = char_end_time(current_pos + sentence_len)
                if end_time <= start_time:
                    # Degenerate ASR timestamps (e.g. Whisper collapsing repeated
                    # words onto one point): best-effort extend the span to the end
                    # of a nearby word that actually ends later. The scan is
                    # bounded (a few words / a few seconds) so a cue never jumps
                    # arbitrarily far; otherwise fall back to a sane default span.
                    scan_end = min(end_word_idx + 1 + MAX_SCAN_WORDS, len(df_words))
                    for later_idx in range(end_word_idx + 1, scan_end):
                        candidate = float(df_words['end'][later_idx])
                        if candidate > start_time and candidate - start_time <= MAX_EXTEND_SECONDS:
                            end_time = candidate
                            break
                    if end_time <= start_time:
                        end_time = start_time + fallback_duration
                time_stamp_list.append((start_time, end_time))
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1

        if not match_found:
            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(
                clean_sentence,
                full_words_str[current_pos:current_pos + len(clean_sentence)],
            )
            print("\nOriginal sentence:", df_sentences['Source'][idx])
            raise ValueError("❎ No match found for sentence.")

    # Monotonic by construction: char-position-linear timing already keeps the
    # cues ordered, but a defensive floor (start_i = max(start_i, prev_end))
    # guarantees validate_positive_timestamps can never see a regression even
    # for pathological overlapping word timestamps.
    return _enforce_monotonic_timestamps(time_stamp_list)

def validate_positive_timestamps(df_trans_time):
    """Reject invalid or regressed timestamps before writing subtitles."""
    invalid = df_trans_time[
        df_trans_time['Source'].fillna('').astype(str).str.strip().ne('')
        & df_trans_time['duration'].le(0)
    ]
    if not invalid.empty:
        details = "; ".join(
            f"row {idx + 1}: {row['Source']!r}, timestamp={row['timestamp']}"
            for idx, row in invalid.head(5).iterrows()
        )
        raise ValueError(f"Non-empty subtitles must have positive duration: {details}")

    start_times = df_trans_time['timestamp'].apply(lambda value: float(value[0]))
    regressed = start_times.diff().lt(0)
    if regressed.any():
        raise ValueError(
            f"Subtitle timestamps regress in {int(regressed.sum())} row(s)"
        )

def normalize_timestamps(df_trans_time):
    """Guarantee monotonic, non-overlapping, duration-capped subtitle rows.

    Runs after matching so downstream SRT consumers never see negative gaps,
    overlapping cues or absurd multi-minute lines. Applied once for both the
    display path and the audio/remerged path (they share align_timestamp()).

    (a) Clamp overlaps: end_i = min(end_i, start_{i+1}), keeping duration > 0.
    (b) Cap absurd per-line durations to MAX_SUBTITLE_DURATION with a warning.
    """
    timestamps = [[float(x[0]), float(x[1])] for x in df_trans_time['timestamp']]

    # (a) absorb small gaps and clamp overlaps / negative gaps
    for i in range(len(timestamps) - 1):
        cur = timestamps[i]
        nxt = timestamps[i + 1]
        delta_time = nxt[0] - cur[1]
        if 0 < delta_time < 1:
            cur[1] = nxt[0]  # absorb small gaps by extending the previous cue
        if cur[1] > nxt[0]:
            cur[1] = nxt[0]  # clamp overlap: never start the next cue early
        if cur[1] - cur[0] <= 0:
            cur[1] = cur[0] + MIN_DURATION_EPS  # keep duration > 0

    # (b) cap absurd per-line durations
    for i in range(len(timestamps)):
        start, end = timestamps[i]
        if end - start > MAX_SUBTITLE_DURATION:
            console.print(
                f"[yellow]⚠️ Subtitle row {i + 1} duration {end - start:.1f}s exceeds "
                f"{MAX_SUBTITLE_DURATION:.0f}s, capping it.[/yellow]"
            )
            timestamps[i][1] = start + MAX_SUBTITLE_DURATION

    df_trans_time['timestamp'] = [tuple(ts) for ts in timestamps]
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Drop padding / empty rows before matching: rows without source text cannot
    # be aligned, and `None` (stage-5 schema padding) would crash the exact-match
    # search. They are skipped here with a warning instead of failing the stage.
    empty_mask = df_trans_time['Source'].isna() | (df_trans_time['Source'].astype(str).str.strip() == '')
    if empty_mask.any():
        console.print(
            f"[yellow]⚠️ Skipping {int(empty_mask.sum())} empty subtitle row(s) "
            f"with no source text.[/yellow]"
        )
        df_trans_time = df_trans_time[~empty_mask].reset_index(drop=True)

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_trans_time)
    if len(time_stamp_list) != len(df_trans_time):
        raise ValueError(
            f"Timestamp count mismatch: matched {len(time_stamp_list)} "
            f"sentence(s) for {len(df_trans_time)} subtitle row(s)"
        )
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Normalize: absorb small gaps, clamp overlaps and cap absurd durations so
    # the SRTs stay monotonic and non-overlapping. Shared by display and audio.
    normalize_timestamps(df_trans_time)

    # Reject invalid or regressed timestamps before writing any subtitle.
    validate_positive_timestamps(df_trans_time)

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', x).strip())

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()
