"""V4-1 spike: measured benchmark for HY-MT2-1.8B as the local translation LLM.

Drives bench/v4_1_server.py through the real pipeline contracts (segment
sentences JSON, translate faithful/refine translations JSON) and measures
cold start, VRAM, thinking leakage, JSON stability under the original and
example-anchored prompts, throughput, and zh<->en quality samples for human
review. One-off harness per spec #32 V4-1; not product code.

Usage:
    .venv/bin/python bench/v4_1_spike.py \
        --model-dir _model_cache/HY-MT2-1.8B \
        --words-json bench/results/native-asr-m1/e2e-60min-words.json \
        --out bench/results/v4-1/spike.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")
# NOTE: the segment stage moved to deterministic rules in V4-3 and no longer
# has a system prompt; this spike keeps only the translate prompts below.
from pipeline.stages.translate import (  # noqa: E402
    SYSTEM_FAITHFUL as FAITHFUL_ORIG,
    SYSTEM_REFINE as REFINE_ORIG,
)

# Example-anchored variants: same JSON I/O contract as the originals; wording
# adds a filled example so HY-MT2 puts the translation inside "target" instead
# of inventing an "extra" key (orig prompt fails deterministically, see report).
SEGMENT_V2 = (
    "You split an ordered ASR word stream into natural sentences. Output ONLY a "
    'JSON object: {"sentences":[{"start":<integer>,"end":<integer>}, ...]} where '
    "indices are half-open [start, end), strictly increasing, first start is 0, "
    "last end is the total word count, and every word is covered exactly once in "
    'order. Example: words ["今天","天气","很好","我们","出门"] -> '
    '{"sentences":[{"start":0,"end":3},{"start":3,"end":5}]}. The stream is '
    "Chinese: adjacent characters arrive as separate words without spaces."
)
FAITHFUL_V2 = (
    "You translate subtitle segments faithfully into the target language. Output "
    'ONLY a JSON object: {"translations":[{"id":<integer>,"target":"<translation '
    'string>"}, ...]} with one entry per input id, the full translation inside '
    '"target", and no additional keys. Example input {"segments":[{"id":1,'
    '"text":"你好世界"}]} -> example output {"translations":[{"id":1,"target":'
    '"Hello world."}]}. Preserve names, numbers and meaning.'
)
REFINE_V2 = (
    "You refine subtitle translations for natural, fluent target-language "
    'speech. Output ONLY a JSON object: {"translations":[{"id":<integer>,'
    '"target":"<refined translation string>"}, ...]} with one entry per input '
    'id, the full refined translation inside "target", and no additional keys. '
    'Example input {"segments":[{"id":1,"source":"Hello world.","draft":'
    '"你好世界。"}]} -> example output {"translations":[{"id":1,"target":'
    '"你好，世界。"}]}. Do not add facts or explanations; keep every id.'
)

NO_PROXY_ENV = {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

# Representative en->zh subtitle lines (project direction: foreign video to
# 简体中文). Marked as synthetic in the report; zh->en lines are real.
EN_ZH_LINES = [
    "We need to talk about what happened yesterday.",
    "The engine overheats when you push it past eighty.",
    "She said she'd call back before the meeting starts.",
    "This is the last train to Stockholm tonight.",
    "I've been working on this for three months straight.",
    "Turn left at the second traffic light, then go straight.",
    "The report says global shipments dropped by twelve percent.",
    "Are you sure you locked the door before we left?",
    "My grandmother used to make dumplings every Sunday.",
    "In this episode we'll visit the old harbor district.",
    "The doctor recommended surgery within two weeks.",
    "Nobody expected the storm to arrive so early.",
]

THINKING_MARKERS = [
    "<think", "</think", "reasoning_content", "As an AI", "I cannot",
    "Let me think", "让我想一想", "好的，我来翻译",
]


def parse_json_like(text):
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def post_chat(base_url, system, user, max_tokens=2048, timeout=300):
    payload = json.dumps(
        {"model": "hy-mt2-1.8b",
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
         "temperature": 0, "max_tokens": max_tokens}).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def nvidia_smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    used, free = (int(part.split()[0]) for part in out.split(","))
    return {"used_mib": used, "free_mib": free}


def group_lines(words, gap_s=0.9, max_line_s=6.0):
    lines, current, start, current_end = [], [], None, None
    for word in words:
        if not current:
            start, current_end = word["start"], word["end"]
            current = [word["text"]]
            continue
        if word["start"] - current_end > gap_s or word["end"] - start > max_line_s:
            lines.append("".join(current))
            current, start = [word["text"]], word["start"]
        else:
            current.append(word["text"])
        current_end = word["end"]
    if current:
        lines.append("".join(current))
    return [line.strip() for line in lines if len(line.strip()) >= 8]


def check_segment(content, total_words):
    sentences = parse_json_like(content)["sentences"]
    ordered = all(0 <= s["start"] < s["end"] for s in sentences)
    contiguous = (sentences[0]["start"] == 0
                  and all(sentences[i]["end"] == sentences[i + 1]["start"]
                          for i in range(len(sentences) - 1))
                  and sentences[-1]["end"] == total_words)
    return {"ok": bool(ordered and contiguous), "n_sentences": len(sentences),
            "covers_all_words_in_order": contiguous, "raw_excerpt": content[:300]}


def check_translations(content, expected_ids):
    """Pipeline-grade validation mirroring stages/translate._parse_translations."""
    items = parse_json_like(content)["translations"]
    result = {}
    extra_keys = sorted({k for item in items for k in item if k not in ("id", "target")})
    for item in items:
        result[int(item["id"])] = str(item.get("target", "")).strip()
    return {
        "ok": (sorted(result) == sorted(expected_ids)
               and all(result[i] for i in expected_ids)),
        "ids_complete": sorted(result) == sorted(expected_ids),
        "all_targets_nonempty": all(result[i] for i in result),
        "extra_keys": extra_keys,
    }


def faithful(base_url, system, lines, source_language, target_language):
    request = {"source_language": source_language, "target_language": target_language,
               "segments": [{"id": i + 1, "text": line}
                            for i, line in enumerate(lines)]}
    response = post_chat(base_url, system, json.dumps(request, ensure_ascii=False))
    content = response["choices"][0]["message"]["content"]
    parsed = parse_json_like(content)
    by_id = {int(item["id"]): str(item.get("target", "")).strip()
             for item in parsed["translations"]}
    return [by_id[i + 1] for i in range(len(lines))], content, response["_timing"]


def refine(base_url, system, lines, drafts, source_language, target_language):
    request = {"source_language": source_language, "target_language": target_language,
               "segments": [{"id": i + 1, "source": line, "draft": draft}
                            for i, (line, draft) in enumerate(zip(lines, drafts))]}
    response = post_chat(base_url, system, json.dumps(request, ensure_ascii=False))
    content = response["choices"][0]["message"]["content"]
    parsed = parse_json_like(content)
    by_id = {int(item["id"]): str(item.get("target", "")).strip()
             for item in parsed["translations"]}
    return [by_id[i + 1] for i in range(len(lines))], content, response["_timing"]


def scan_thinking(outputs):
    hits = []
    for label, text in outputs:
        for marker in THINKING_MARKERS:
            if marker.lower() in str(text).lower():
                hits.append({"where": label, "marker": marker,
                             "excerpt": str(text)[:120]})
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--words-json", required=True)
    parser.add_argument("--zh-split-log", default="output/log/split_by_nlp.txt",
                        help="old-pipeline real split lines for zh->en samples")
    parser.add_argument("--port", type=int, default=9931)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"

    smi_before = nvidia_smi()
    server = subprocess.Popen(
        [".venv/bin/python", "bench/v4_1_server.py",
         "--model-dir", args.model_dir, "--port", str(args.port),
         # Model config ships repetition_penalty=1.05 for sampling chat; under
         # greedy JSON decoding it causes deterministic token degeneration
         # ("target","id":10,"target":...) at the same offset every run.
         "--repetition-penalty", "1.0"],
        cwd=".", env={**os.environ, **NO_PROXY_ENV},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    report = {"schema_version": 3, "tool": "v4_1_spike",
              "model": "HY-MT2-1.8B BF16 transformers FastAPI",
              "decoding": "greedy, repetition_penalty=1.0"}
    outputs, failures = [], []
    try:
        health = None
        for _ in range(240):
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
                    health = json.loads(r.read())
                    break
            except (urllib.error.URLError, ConnectionError):
                if server.poll() is not None:
                    raise RuntimeError("server exited: " + server.stdout.read()[-2000:])
                time.sleep(1)
        if health is None:
            raise RuntimeError("server did not become healthy in 240s")
        report["cold_load"] = {"server_reported_load_s": health["load_s"],
                               "torch_max_mem_gb": health["torch_max_mem_gb"],
                               "smi_before": smi_before, "smi_after": nvidia_smi()}

        words = json.load(open(args.words_json, encoding="utf-8"))
        zh_lines = group_lines(words)[:12]
        sample_words = [w["text"].strip() for w in words[:60] if w["text"].strip()]
        # Quality zh->en samples: the old pipeline's real NLP split product for
        # this same 60-min video (production workload, what the translate stage
        # actually receives), not synthetic windows.
        zh_quality = [line.strip() for line in
                      open(args.zh_split_log, encoding="utf-8").read().splitlines()
                      if len(line.strip()) >= 15][:12]

        # --- segment contract: orig vs v2 ---
        seg_request = json.dumps(
            {"language": "zh",
             "words": [{"index": i, "text": t} for i, t in enumerate(sample_words)]},
            ensure_ascii=False)
        report["segment_contract"] = {}
        for tag, system in (("v2", SEGMENT_V2),):
            response = post_chat(base_url, system, seg_request)
            content = response["choices"][0]["message"]["content"]
            outputs.append((f"segment_{tag}", content))
            try:
                report["segment_contract"][tag] = check_segment(
                    content, len(sample_words))
            except Exception as exc:  # noqa: BLE001
                report["segment_contract"][tag] = {"ok": False,
                                                   "error": str(exc)[:200],
                                                   "raw_excerpt": content[:300]}

        # --- translate contract: orig vs v2, both directions, 12-seg ---
        ids12 = list(range(1, 13))
        report["faithful_contract"] = {}
        direction_requests = {
            "zh_en": {"source_language": "Chinese", "target_language": "English",
                      "segments": [{"id": i + 1, "text": zh_lines[i]}
                                   for i in range(12)]},
            "en_zh": {"source_language": "English", "target_language": "简体中文",
                      "segments": [{"id": i + 1, "text": EN_ZH_LINES[i]}
                                   for i in range(12)]},
        }
        for direction, faithful_request in direction_requests.items():
            for tag, system in (("orig", FAITHFUL_ORIG), ("v2", FAITHFUL_V2)):
                response = post_chat(base_url, system,
                                     json.dumps(faithful_request, ensure_ascii=False))
                content = response["choices"][0]["message"]["content"]
                outputs.append((f"faithful_{direction}_{tag}", content))
                try:
                    entry = check_translations(content, ids12)
                    entry["timing"] = response["_timing"]
                    report["faithful_contract"][f"{direction}_{tag}"] = entry
                except Exception as exc:  # noqa: BLE001
                    report["faithful_contract"][f"{direction}_{tag}"] = {
                        "ok": False, "error": str(exc)[:200],
                        "raw_excerpt": content[:300]}

        # --- throughput: 6 sequential 12-seg faithful batches on v2 ---
        timings = []
        for round_index in range(6):
            request = {**direction_requests["zh_en"],
                       "segments": [{"id": i + 1,
                                     "text": zh_lines[(round_index + i) % 12]}
                                    for i in range(12)]}
            result = post_chat(base_url, FAITHFUL_V2,
                               json.dumps(request, ensure_ascii=False))
            timings.append(result["_timing"])
        report["throughput_12seg_v2"] = {
            "runs": timings,
            "tok_per_s_median": statistics.median(t["tok_per_s"] for t in timings),
            "decode_s_median": statistics.median(t["decode_s"] for t in timings),
        }

        # --- quality: zh->en and en->zh, faithful(v2) vs refine(v2/orig) ---
        quality = {}
        for source_language, target_language, lines, tag in (
                ("Chinese", "English", zh_quality, "zh_en"),
                ("English", "简体中文", EN_ZH_LINES, "en_zh")):
            try:
                drafts, faith_content, _ = faithful(
                    base_url, FAITHFUL_V2, lines, source_language, target_language)
                outputs.append((f"faithful_v2_{tag}", faith_content))
                refined, ref_content, _ = refine(
                    base_url, REFINE_V2, lines, drafts, source_language,
                    target_language)
                outputs.append((f"refine_v2_{tag}", ref_content))
                quality[tag] = {
                    "samples": [
                        {"source": source, "faithful": draft, "refined": better,
                         "sanity": {"nonempty": bool(draft.strip() and better.strip()),
                                    "changed_by_refine": draft.strip() != better.strip()}}
                        for source, draft, better in zip(lines, drafts, refined)]}
            except Exception as exc:  # noqa: BLE001
                failures.append({"case": f"quality_{tag}", "error": str(exc)[:200]})
        report["quality_for_human_review"] = quality

        # --- refine orig prompt on v2 drafts (does orig wording refine OK?) ---
        try:
            drafts, _, _ = faithful(base_url, FAITHFUL_V2, zh_lines[:12], "Chinese",
                                    "English")
            refined, ref_content, _ = refine(base_url, REFINE_ORIG, zh_lines[:12],
                                             drafts, "Chinese", "English")
            outputs.append(("refine_orig_zh_en", ref_content))
            report["refine_orig_contract"] = check_translations(ref_content, ids12)
        except Exception as exc:  # noqa: BLE001
            report["refine_orig_contract"] = {"ok": False, "error": str(exc)[:200]}

        report["thinking_scan"] = {"markers": THINKING_MARKERS,
                                   "hits": scan_thinking(outputs)}
        report["failures"] = failures
        report["smi_final"] = nvidia_smi()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {k: v for k, v in report.items()
               if k not in ("quality_for_human_review", "segment_contract")}
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3500])
    print("\nsegment_contract:", json.dumps(report["segment_contract"],
                                            ensure_ascii=False)[:600])
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
