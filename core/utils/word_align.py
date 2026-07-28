"""
Word-level alignment utilities.

Replaces the fragile character-level string matching in get_sentence_timestamps()
with robust sequential word-level matching. Sentences are matched against the
word DataFrame in order, allowing minor ASR artifacts without raising errors.
"""

import re
from typing import List, Tuple
from difflib import SequenceMatcher


def normalize_word(text: str) -> str:
    """Normalize a single word for matching: strip quotes/punctuation, lowercase."""
    text = str(text).strip().strip('"').strip()
    text = re.sub(r'[^\w]', '', text, flags=re.UNICODE)
    return text.lower()


def normalize_sentence_to_words(text: str) -> List[str]:
    """Split a sentence into a list of normalized lowercased word tokens."""
    text = str(text)
    raw_words = re.findall(r'\w+', text, flags=re.UNICODE)
    return [w.lower() for w in raw_words]


def get_timestamps_for_sentences(
    df_words,
    sentences: List[str],
    look_ahead: int = 8,
    fuzzy_threshold: float = 0.75,
) -> List[Tuple[float, float]]:
    """
    Match each sentence to a word range in df_words using sequential
    word-level matching.

    This replaces the old get_sentence_timestamps() which joined all words
    into a giant string and did exact character-level substring matching
    (raising ValueError on any mismatch). This version:

    - Matches word-by-word in sequential order (sentences follow words).
    - Allows minor ASR artifacts via look-ahead and fuzzy matching.
    - NEVER raises — always returns a best-effort timestamp.

    Args:
        df_words: DataFrame with 'text', 'start', 'end' columns.
        sentences: List of sentence texts to match.
        look_ahead: How many words forward to search when a direct match fails.
        fuzzy_threshold: Similarity ratio above which two words are considered equal.

    Returns:
        List of (start_time, end_time) tuples, one per sentence.
    """
    n_words = len(df_words)
    if n_words == 0:
        return [(0.0, 0.0)] * len(sentences)

    # Pre-normalize all words once
    word_norms = [normalize_word(t) for t in df_words['text'].tolist()]
    word_starts = df_words['start'].tolist()
    word_ends = df_words['end'].tolist()

    timestamps = []
    word_idx = 0  # current scan position in the word list

    for sentence in sentences:
        target_words = normalize_sentence_to_words(sentence)

        # Handle empty sentence
        if not target_words:
            if word_idx < n_words:
                t = float(word_starts[word_idx])
                timestamps.append((t, t))
            else:
                last_t = float(word_ends[n_words - 1]) if n_words > 0 else 0.0
                timestamps.append((last_t, last_t))
            continue

        # If we've consumed all words, assign the last word's time
        if word_idx >= n_words:
            last_t = float(word_ends[n_words - 1])
            timestamps.append((last_t, last_t))
            continue

        start_idx = word_idx
        target_i = 0
        wi = word_idx

        while wi < n_words and target_i < len(target_words):
            target_w = target_words[target_i]

            # Direct match
            if word_norms[wi] == target_w:
                target_i += 1
                wi += 1
                continue

            # Skip empty normalized words (ASR artifacts)
            if word_norms[wi] == '':
                wi += 1
                continue

            # Look ahead for a match within the window
            found = False
            for la in range(1, min(look_ahead, n_words - wi)):
                if word_norms[wi + la] == target_w:
                    # Skip the non-matching words (they are likely ASR insertions)
                    wi += la + 1
                    target_i += 1
                    found = True
                    break
            if found:
                continue

            # Fuzzy match (minor spelling difference)
            sim = SequenceMatcher(None, word_norms[wi], target_w).ratio()
            if sim >= fuzzy_threshold:
                target_i += 1
                wi += 1
                continue

            # No match — advance word pointer, keep target pointer
            # This handles extra words in ASR output not present in the sentence
            wi += 1

        end_idx = min(wi - 1, n_words - 1)
        if end_idx < start_idx:
            end_idx = start_idx

        # Ensure at least one word is consumed for forward progress
        if end_idx < word_idx:
            end_idx = min(word_idx, n_words - 1)

        timestamps.append((float(word_starts[start_idx]), float(word_ends[end_idx])))
        word_idx = end_idx + 1

    return timestamps



