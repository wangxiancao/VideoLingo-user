"""
Sequential queue driver for translate_video.py over all videos in a folder.

Usage:
    python scripts/queue_translate.py            # uses TTS_WORKERS env or default 3
    TTS_WORKERS=2 python scripts/queue_translate.py

State persisted to queue_state.json — safe to interrupt and rerun; already-accepted
videos are skipped. On CUDA OOM, TTS workers auto-downgrade by 1 for that video's
next attempt (won't rerun automatically; rerun the script to retry).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ------------
# Config
# ------------
VIDEO_DIR = Path("/root/buzz_file/pi_agent")
PROJECT = Path("/root/VideoLingo")
HISTORY = PROJECT / "history"
STATE_FILE = PROJECT / "queue_state.json"
LOG_DIR = PROJECT / "queue_logs"
PYTHON = "/root/soft/anaconda3/envs/videolingo/bin/python"

DEFAULT_TTS_WORKERS = int(os.environ.get("TTS_WORKERS", "3"))
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"}


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def history_dirs():
    return set(p.name for p in HISTORY.iterdir() if p.is_dir()) if HISTORY.exists() else set()


def detect_oom(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore").lower()
    return "out of memory" in text or "cuda error" in text


def run_one(video, state, idx, total):
    rec = state.get(video.name, {})
    workers = rec.get("workers", DEFAULT_TTS_WORKERS)
    accepted = rec.get("accepted", False)
    if accepted:
        print(f"[{idx}/{total}] SKIP (accepted): {video.name}")
        return

    print(f"\n[{idx}/{total}] {video.name}  (tts-workers={workers})")

    before = history_dirs()
    log_path = LOG_DIR / f"{video.stem}.log"
    LOG_DIR.mkdir(exist_ok=True)

    cmd = [
        PYTHON, "translate_video.py",
        "--mode", "tts",
        "--tts-workers", str(workers),
        str(video),
    ]

    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(
            cmd, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT, text=True
        )
    elapsed = time.time() - t0

    after = history_dirs()
    new_folders = after - before

    rec["workers"] = workers
    rec["exit_code"] = result.returncode
    rec["elapsed_sec"] = round(elapsed, 1)
    rec["last_attempt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["new_history_folders"] = sorted(new_folders)
    rec["accepted"] = bool(new_folders)

    if not rec["accepted"] and detect_oom(log_path):
        rec["workers"] = max(1, workers - 1)
        rec["oom_downgraded"] = True
        print(f"  -> OOM detected, downgraded to workers={rec['workers']} for next run")

    state[video.name] = rec
    save_state(state)

    status = "ACCEPTED" if rec["accepted"] else "FAILED"
    print(f"  -> {status}  exit={result.returncode}  time={rec['elapsed_sec']}s  new={list(new_folders)}")


def main():
    if not VIDEO_DIR.is_dir():
        sys.exit(f"VIDEO_DIR not found: {VIDEO_DIR}")
    if not PYTHON.exists():
        sys.exit(f"Python not found: {PYTHON}")

    videos = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"No videos in {VIDEO_DIR}")

    state = load_state()
    total = len(videos)
    for idx, v in enumerate(videos, 1):
        run_one(v, state, idx, total)

    accepted = sum(1 for r in state.values() if r.get("accepted"))
    print(f"\n=== Summary: {accepted}/{total} accepted ===")
    for name, rec in state.items():
        mark = "OK" if rec.get("accepted") else "FAIL"
        print(f"  [{mark}] {name}  exit={rec.get('exit_code')}  time={rec.get('elapsed_sec')}s")


if __name__ == "__main__":
    main()
