import math
import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from core.spacy_utils.load_nlp_model import get_joiner_safe
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    language = load_key("whisper.language")
    # Route through the safe joiner: real detected codes (e.g. ko/pt/tr) that
    # are missing from the config split lists must never crash get_joiner().
    # CJK target text still joins without spaces.
    tr_stream = "".join(tr_parts)
    joiner = get_joiner_safe(language, tr_stream)
    tr_remerged = joiner.join(tr_parts)

    # The LLM can return a DIFFERENT number of target parts than the source
    # was split into, or even skip lines entirely. Enforce equal counts now,
    # before any caller `zip` (flattening in split_align_subs, per-row pairing
    # in stage 6) could silently truncate the longer side. '' padding preserves
    # the text stream on BOTH sides: "".join(src_parts) equals src_part with its
    # '\n' separators stripped (src_part holds one line per source part, see the
    # split('\n') above, so join() alone never re-equals src_part), and the
    # target stream `tr_stream` is unchanged. Comparing the newline-normalized
    # form is what makes the invariant genuinely meaningful for successful multi-line
    # LLM splits instead of an assert that always fails and sends every line
    # to the deterministic fallback. tr_remerged is computed above from the
    # real (unpadded) parts so no extra joiner separators wrap '' pieces.
    src_parts, tr_parts = _pad_equal_parts(src_parts, tr_parts)
    assert "".join(src_parts) == src_part.replace("\n", ""), "source stream altered by padding"
    assert "".join(tr_parts) == tr_stream, "target stream altered by padding"
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def chunk_by_char_count(source: str, n: int) -> List[str]:
    """Split `source` into at most n sequential pieces by character count."""
    source = str(source)
    if n <= 1 or not source:
        return [source]
    size = math.ceil(len(source) / n)
    return [source[i:i + size] for i in range(0, len(source), size)]


def split_by_weight_limit(target: str, max_weight: float) -> List[str]:
    """Strict greedy split of `target` so every part satisfies
    `calc_len(part) <= max_weight`. Used for the deterministic fallback."""
    target = str(target)
    if calc_len(target) <= max_weight:
        return [target]
    parts = []
    cur = ''
    cur_w = 0.0
    for ch in target:
        w = calc_len(ch)
        if cur and cur_w + w > max_weight:
            parts.append(cur)
            cur = ch
            cur_w = w
        else:
            cur += ch
            cur_w += w
    if cur:
        parts.append(cur)
    return parts


def _pad_equal_parts(src_parts: List[str], tr_parts: List[str]) -> Tuple[List[str], List[str]]:
    """Pad the SHORTER parallel side with '' pieces so both sides have equal counts.

    '' pieces are length/weight zero, preserve the original text stream
    ("".join(parts) is unchanged on either side) and are safely skipped by
    stage 6. Without this, any downstream `zip` silently truncates the longer
    side and drops text.
    """
    n = max(len(src_parts), len(tr_parts))
    if len(src_parts) < n:
        src_parts = src_parts + [''] * (n - len(src_parts))
    if len(tr_parts) < n:
        tr_parts = tr_parts + [''] * (n - len(tr_parts))
    return src_parts, tr_parts


def split_pair_to_fit(source: str, target: str, max_len: int, multiplier: float) -> Tuple[List[str], List[str]]:
    """Deterministic fallback/force split producing equal-length parallel parts.

    The target is strictly split by `calc_len` weight (every piece within the
    configured limit); the source is split into at least as many pieces by
    character count. Both sides then get the same number of pieces — the short
    side is padded with '' pieces, which preserves the original text stream
    ("".join(parts) is unchanged) and is safely skipped by stage 6. Used when
    the LLM split/alignment fails or when a line would otherwise stay over-length.
    """
    source, target = str(source), str(target)
    max_target_weight = max_len / multiplier
    tr_parts = split_by_weight_limit(target, max_target_weight)
    src_parts_needed = max(1, math.ceil(len(source) / max_len))
    n = max(src_parts_needed, len(tr_parts))
    src_parts = chunk_by_char_count(source, n)
    return _pad_equal_parts(src_parts, tr_parts)


def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    # Zip-safety invariant: every level below pairs src/tr by index (detection
    # loop, per-line replacement, flattening, final pass). Normalize a
    # mismatched input to equal counts up front so no level ever silently
    # truncates the longer side. '' padding preserves both text streams
    # ("".join) and is safely skipped by stage 6, exactly like
    # split_pair_to_fit / _pad_equal_parts. Entries are left as-is (no str()
    # coercion) so existing NaN/empty cell semantics flow through untouched.
    src_lines, tr_lines = _pad_equal_parts(src_lines.copy(), tr_lines.copy())

    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    def process(i):
        try:
            split_src = split_sentence(src_lines[i], num_parts=2).strip()
            src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
            src_lines[i] = src_parts
            tr_lines[i] = tr_parts
            remerged_tr_lines[i] = tr_remerged
        except Exception as e:
            console.print(
                f"[red]⚠️ LLM split/align failed for line {i}: {e}.[/red] "
                f"[yellow]Falling back to length-based split.[/yellow]"
            )
            original_tr = str(tr_lines[i])
            src_parts, tr_parts = split_pair_to_fit(src_lines[i], original_tr, MAX_SUB_LENGTH, TARGET_SUB_MULTIPLIER)
            src_lines[i] = src_parts
            tr_lines[i] = tr_parts
            remerged_tr_lines[i] = original_tr
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        futures = [executor.submit(process, i) for i in to_split]
        # Consume every future so worker exceptions are surfaced instead of
        # being silently swallowed by an unconsumed iterator.
        for future in futures:
            try:
                future.result()
            except Exception as e:
                console.print(f"[red]⚠️ split_align_subs worker raised: {e}[/red]")
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    # Zip-safety for the final pass below: never truncate the longer side.
    # Stream preservation is by construction everywhere: flattening only
    # expands the per-line parts in place, split_pair_to_fit chunks without
    # dropping text, and align_subs pads with '' (asserted inline). The only
    # intentional deviation remains split_sentence(...).strip() inside the LLM
    # path, which predates this change.
    src_lines, tr_lines = _pad_equal_parts(src_lines, tr_lines)

    # Defensive last pass: never write over-length lines. Anything still too
    # long is force-split with a deterministic character/length split.
    final_src, final_tr = [], []
    for src, tr in zip(src_lines, tr_lines):
        if len(src) <= MAX_SUB_LENGTH and calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH:
            final_src.append(src)
            final_tr.append(tr)
        else:
            src_parts, tr_parts = split_pair_to_fit(src, tr, MAX_SUB_LENGTH, TARGET_SUB_MULTIPLIER)
            final_src.extend(src_parts)
            final_tr.extend(tr_parts)
    
    return final_src, final_tr, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
