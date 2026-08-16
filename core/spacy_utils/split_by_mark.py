import os
import pandas as pd
import warnings
from core.spacy_utils.load_nlp_model import (
    init_nlp,
    SPLIT_BY_MARK_FILE,
    resolve_language,
    get_joiner_safe,
    is_cjk_text,
    split_cjk_by_punctuation,
)
from rich import print as rprint

warnings.filterwarnings("ignore", category=FutureWarning)

def split_by_mark(nlp):
    language = resolve_language()
    chunks = pd.read_excel("output/log/cleaned_chunks.xlsx")
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))
    raw_texts = chunks.text.to_list()

    # Detect the script from the actual content so unsupported language codes
    # still get the right joiner ("" for CJK, " " otherwise).
    sample = "".join(raw_texts)
    joiner = get_joiner_safe(language, sample)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")

    # join with joiner
    input_text = joiner.join(raw_texts)

    # skip - and ...
    sentences_by_mark = []
    current_sentence = []

    model_lang = (getattr(nlp, "meta", {}) or {}).get("lang", "") or ""
    if is_cjk_text(sample) and model_lang not in ("zh", "ja", "ko"):
        # The language-specific spacy model is missing (only en_core_web_md is
        # installed), so doc.sents would treat a whole CJK line as one giant
        # sentence. Split at CJK sentence-ending punctuation instead.
        rprint(f"[yellow]⚠️ Spacy model <{model_lang}> cannot segment CJK text; "
               f"using punctuation-based fallback.[/yellow]")
        sentences_by_mark = split_cjk_by_punctuation(input_text)
    else:
        doc = nlp(input_text)
        assert doc.has_annotation("SENT_START")

        # iterate all sentences
        for sent in doc.sents:
            text = sent.text.strip()

            # check if the current sentence ends with - or ...
            if current_sentence and (
                text.startswith('-') or
                text.startswith('...') or
                current_sentence[-1].endswith('-') or
                current_sentence[-1].endswith('...')
            ):
                current_sentence.append(text)
            else:
                if current_sentence:
                    sentences_by_mark.append(' '.join(current_sentence))
                    current_sentence = []
                current_sentence.append(text)

        # add the last sentence
        if current_sentence:
            sentences_by_mark.append(' '.join(current_sentence))

    with open(SPLIT_BY_MARK_FILE, "w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(sentences_by_mark):
            if i > 0 and sentence.strip() in [',', '.', '，', '。', '？', '！']:
                # ! If the current line contains only punctuation, merge it with the previous line, this happens in Chinese, Japanese, etc.
                output_file.seek(output_file.tell() - 1, os.SEEK_SET)  # Move to the end of the previous line
                output_file.write(sentence)  # Add the punctuation
            else:
                output_file.write(sentence + "\n")
    
    rprint(f"[green]💾 Sentences split by punctuation marks saved to →  `{SPLIT_BY_MARK_FILE}`[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    split_by_mark(nlp)
