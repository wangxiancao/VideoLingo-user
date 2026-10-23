"""V4-7 O1 probe: Q4 (llama.cpp) vs BF16 (transformers) side by side.

Drives the V4-1-spike contract matrix (both directions, example-anchored
prompts, pipeline-grade validation) against two already-running local
endpoints, measures decode throughput, and saves side-by-side outputs for
maintainer blind review. One-off evidence harness per spec #54 / ticket #55;
not product code.

Usage:
    .venv/bin/python bench/v4_7_q4_probe.py \
        --q4-url http://127.0.0.1:9940/v1 --bf16-url http://127.0.0.1:9931/v1 \
        --out bench/results/v4-7/q4-probe.json
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())

from bench.v4_1_spike import (  # noqa: E402
    EN_ZH_LINES,
    FAITHFUL_V2,
    REFINE_V2,
    check_translations,
    parse_json_like,
)


def post_chat(base_url, system, user, max_tokens=2048, timeout=300):
    payload = json.dumps(
        {"model": "probe",
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
         "temperature": 0, "max_tokens": max_tokens}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - started
    usage = result.get("usage", {})
    completion = usage.get("completion_tokens")
    timing = result.get("_timing") or {}
    tok_per_s = timing.get("tok_per_s")
    if tok_per_s is None and completion and wall:
        tok_per_s = round(completion / wall, 1)
    return {
        "content": result["choices"][0]["message"]["content"],
        "wall_s": round(wall, 3),
        "completion_tokens": completion,
        "tok_per_s": tok_per_s,
    }


def run_matrix(base_url, zh_lines):
    """Faithful contract matrix both directions + refine, pipeline-grade."""
    results = {}
    outputs = {}
    for direction, source_language, lines in (
            ("zh_en", "Chinese", zh_lines), ("en_zh", "English", EN_ZH_LINES)):
        ids = list(range(1, len(lines) + 1))
        request = json.dumps(
            {"source_language": source_language,
             "target_language": "English" if direction == "zh_en" else "简体中文",
             "segments": [{"id": i + 1, "text": line}
                          for i, line in enumerate(lines)]},
            ensure_ascii=False)
        response = post_chat(base_url, FAITHFUL_V2, request)
        outputs[f"faithful_{direction}"] = response["content"]
        try:
            entry = check_translations(response["content"], ids)
        except Exception as exc:  # noqa: BLE001
            entry = {"ok": False, "error": str(exc)[:200],
                     "raw_excerpt": response["content"][:200]}
        entry["timing"] = {k: response[k] for k in ("wall_s", "completion_tokens",
                                                    "tok_per_s")}
        results[direction] = entry
    # refine pass on zh->en drafts
    try:
        drafts = [item["target"] for item in
                  parse_json_like(outputs["faithful_zh_en"])["translations"]]
        request = json.dumps(
            {"source_language": "Chinese", "target_language": "English",
             "segments": [{"id": i + 1, "source": s, "draft": d}
                          for i, (s, d) in enumerate(zip(zh_lines, drafts))]},
            ensure_ascii=False)
        response = post_chat(base_url, REFINE_V2, request)
        outputs["refine_zh_en"] = response["content"]
        results["refine_zh_en"] = check_translations(
            response["content"], list(range(1, len(zh_lines) + 1)))
    except Exception as exc:  # noqa: BLE001
        results["refine_zh_en"] = {"ok": False, "error": str(exc)[:200]}
    return results, outputs


def throughput(base_url, zh_lines, rounds=6):
    speeds, walls = [], []
    for round_index in range(rounds):
        request = json.dumps(
            {"source_language": "Chinese", "target_language": "English",
             "segments": [{"id": i + 1,
                           "text": zh_lines[(round_index + i) % len(zh_lines)]}
                          for i in range(12)]},
            ensure_ascii=False)
        response = post_chat(base_url, FAITHFUL_V2, request)
        speeds.append(response["tok_per_s"])
        walls.append(response["wall_s"])
    return {"tok_per_s_median": statistics.median(speeds),
            "wall_s_median_12seg": statistics.median(walls),
            "runs": speeds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-url", required=True)
    parser.add_argument("--bf16-url", default=None,
                        help="skip the BF16 side when the server is not running")
    parser.add_argument("--zh-split-log", default="output/log/split_by_nlp.txt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zh_lines = [line.strip() for line in
                open(args.zh_split_log, encoding="utf-8").read().splitlines()
                if len(line.strip()) >= 15][:12]

    report = {"schema_version": 1, "tool": "v4_7_q4_probe",
              "zh_lines": zh_lines, "en_lines": EN_ZH_LINES}
    q4_matrix, q4_outputs = run_matrix(args.q4_url, zh_lines)
    report["q4"] = {"matrix": q4_matrix,
                    "throughput": throughput(args.q4_url, zh_lines)}
    report["q4_outputs"] = q4_outputs
    if args.bf16_url:
        bf_matrix, bf_outputs = run_matrix(args.bf16_url, zh_lines)
        report["bf16"] = {"matrix": bf_matrix,
                          "throughput": throughput(args.bf16_url, zh_lines)}
        report["bf16_outputs"] = bf_outputs

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {side: {"matrix": {k: v.get("ok") for k, v in
                                 report[side]["matrix"].items()},
                      "throughput": report[side]["throughput"]["tok_per_s_median"]}
               for side in ("q4", "bf16") if side in report}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
