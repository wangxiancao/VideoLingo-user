import os
import string
import warnings
from core.spacy_utils.load_nlp_model import (
    init_nlp,
    SPLIT_BY_CONNECTOR_FILE,
    resolve_language,
    get_joiner_safe,
    is_cjk_text,
    split_cjk_by_punctuation,
)
from core.utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP

warnings.filterwarnings("ignore", category=FutureWarning)

def split_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)
    
    # dynamic programming array, dp[i] represents the optimal split scheme from the start to the ith token
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    # record optimal split points
    prev = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):  # limit search range to avoid overly long sentences
            if i - j >= 30:  # ensure sentence length is at least 30
                token = doc[i-1]
                if j == 0 or (token.is_sent_end or token.pos_ in ['VERB', 'AUX'] or token.dep_ == 'ROOT'):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j
    
    # rebuild sentences based on optimal split points
    sentences = []
    i = n
    language = resolve_language()
    joiner = get_joiner_safe(language, doc.text)
    while i > 0:
        j = prev[i]
        sentences.append(joiner.join(tokens[j:i]).strip())
        i = j
    
    return sentences[::-1]  # reverse list to keep original order

def split_extremely_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)
    
    num_parts = (n + 59) // 60  # round up
    
    part_length = n // num_parts
    
    sentences = []
    language = resolve_language()
    joiner = get_joiner_safe(language, doc.text)
    for i in range(num_parts):
        start = i * part_length
        end = start + part_length if i < num_parts - 1 else n
        sentence = joiner.join(tokens[start:end])
        sentences.append(sentence)
    
    return sentences


def split_long_cjk_text(text, max_len=60):
    """Split CJK text: first at sentence-ending punctuation, then redistribute
    any still-too-long part into even character chunks. Used when the loaded
    spacy model cannot segment the CJK script, so POS/ROOT-driven DP never
    fires on these lines."""
    parts = split_cjk_by_punctuation(text) or [text]
    result = []
    for part in parts:
        if len(part) <= max_len:
            result.append(part)
            continue
        n = len(part)
        num_parts = (n + max_len - 1) // max_len
        step = (n + num_parts - 1) // num_parts
        result.extend(part[i * step:(i + 1) * step] for i in range(num_parts))
    return [p for p in result if p.strip()]


def split_long_by_root_main(nlp):
    with open(SPLIT_BY_CONNECTOR_FILE, "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue

        # CJK scripts collapse into a single token under the (English) fallback
        # spacy model, so the token-count check and the POS/ROOT DP below never
        # fire and multi-minute spans stay unsplit. Use character-level
        # splitting for CJK lines instead.
        if is_cjk_text(text):
            if len(text) > 60:
                split_sentences = split_long_cjk_text(text)
                all_split_sentences.extend(split_sentences)
                rprint(f"[yellow]✂️  Splitting long CJK sentence by characters: {text[:30]}...[/yellow]")
            else:
                all_split_sentences.append(text)
            continue

        doc = nlp(text)
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc)
            if any(len(nlp(sent)) > 60 for sent in split_sentences):
                split_sentences = [subsent for sent in split_sentences for subsent in split_extremely_long_sentence(nlp(sent))]
            all_split_sentences.extend(split_sentences)
            rprint(f"[yellow]✂️  Splitting long sentences by root: {text[:30]}...[/yellow]")
        else:
            all_split_sentences.append(text)

    punctuation = string.punctuation + "'" + '"'  # include all punctuation and apostrophe ' and "

    with open(_3_1_SPLIT_BY_NLP, "w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(all_split_sentences):
            stripped_sentence = sentence.strip()
            if not stripped_sentence or all(char in punctuation for char in stripped_sentence):
                rprint(f"[yellow]⚠️  Warning: Empty or punctuation-only line detected at index {i}[/yellow]")
                if i > 0:
                    all_split_sentences[i-1] += sentence
                continue
            output_file.write(sentence + "\n")

    # delete the original file
    os.remove(SPLIT_BY_CONNECTOR_FILE)   

    rprint(f"[green]💾 Long sentences split by root saved to →  {_3_1_SPLIT_BY_NLP}[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    split_long_by_root_main(nlp)
    # raw = "平口さんの盛り上げごまが初めて売れました本当に嬉しいです本当にやっぱり見た瞬間いいって言ってくれるそういうコマを作るのがやっぱりいいですよねその2ヶ月後チコさんが何やらそわそわしていましたなんか気持ち悪いやってきたのは平口さんの駒の評判を聞きつけた愛知県の収集家ですこの男性師匠大沢さんの駒も持っているといいますちょっと褒めすぎかなでも確実にファンは広がっているようです自信がない部分をすごく感じてたのでこれで自信を持って進んでくれるなっていう本当に始まったばっかりこれからいろいろ挑戦していってくれるといいなと思って今月平口さんはある場所を訪れましたこれまで数々のタイトル戦でコマを提供してきた老舗5番手平口さんのコマを扱いたいと言いますいいですねぇ困ってだんだん成長しますので大切に使ってそういう長く良い駒になる駒ですね商談が終わった後店主があるものを取り出しましたこの前の名人戦で使った駒があるんですけど去年、名人銭で使われた盛り上げごま低く盛り上げて品良くするというのは難しい素晴らしいですね平口さんが目指す高みですこういった感じで作れればまだまだですけどただ、多分、咲く。"
    # nlp = init_nlp()
    # doc = nlp(raw.strip())
    # for sent in split_still_long_sentence(doc):
    #     print(sent, '\n==========')