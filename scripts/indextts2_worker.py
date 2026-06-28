#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TTS = None
MODEL_LOCK = threading.Lock()
STARTUP_ERROR = None
SERVER_CONFIG = {}


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler, status, body):
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


class IndexTTS2Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[indextts2-worker] " + fmt % args + "\n")

    def do_GET(self):
        if self.path == "/health":
            _json_response(self, 200, {
                "ok": STARTUP_ERROR is None and TTS is not None,
                "model_loaded": TTS is not None,
                "startup_error": STARTUP_ERROR,
                "config": SERVER_CONFIG,
            })
            return

        if self.path in ("/", "/index.html"):
            _text_response(self, 200, """<!doctype html>
<html>
<head><meta charset="utf-8"><title>IndexTTS2 Worker</title></head>
<body>
<h1>IndexTTS2 Worker</h1>
<p>Status endpoint: <a href="/health">/health</a></p>
<form id="synth-form">
  <p><label>Text<br><textarea id="text" rows="4" cols="80">Hello, this is an IndexTTS2 worker test.</textarea></label></p>
  <p><label>Reference audio<br><input id="spk_audio_prompt" size="100" placeholder="/absolute/path/to/reference.wav"></label></p>
  <p><label>Output path<br><input id="output_path" size="100" placeholder="/absolute/path/to/output.wav"></label></p>
  <p><button type="submit">Synthesize</button></p>
</form>
<pre id="result"></pre>
<script>
document.getElementById("synth-form").addEventListener("submit", async function(event) {
  event.preventDefault();
  const payload = {
    text: document.getElementById("text").value,
    spk_audio_prompt: document.getElementById("spk_audio_prompt").value,
    output_path: document.getElementById("output_path").value,
    verbose: false
  };
  const result = document.getElementById("result");
  result.textContent = "Running...";
  const response = await fetch("/synthesize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  result.textContent = JSON.stringify(await response.json(), null, 2);
});
</script>
</body>
</html>
""")
            return

        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/synthesize":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        if STARTUP_ERROR is not None or TTS is None:
            _json_response(self, 503, {"ok": False, "error": STARTUP_ERROR or "model not loaded"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload["text"]
            spk_audio_prompt = payload["spk_audio_prompt"]
            output_path = payload["output_path"]
            verbose = bool(payload.get("verbose", False))
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "error": f"invalid request: {exc}"})
            return

        try:
            if not os.path.exists(spk_audio_prompt):
                raise FileNotFoundError(f"reference audio not found: {spk_audio_prompt}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with MODEL_LOCK:
                TTS.infer(
                    spk_audio_prompt=spk_audio_prompt,
                    text=text,
                    output_path=output_path,
                    verbose=verbose,
                )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"IndexTTS2 did not create a valid output file: {output_path}")

            _json_response(self, 200, {"ok": True, "output_path": output_path})
        except Exception as exc:
            _json_response(self, 500, {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })


def parse_args():
    parser = argparse.ArgumentParser(description="VideoLingo IndexTTS2 HTTP worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18721)
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--use-deepspeed", action="store_true")
    return parser.parse_args()


def load_model(args):
    global TTS, STARTUP_ERROR, SERVER_CONFIG

    model_dir = Path(args.model_dir).resolve()
    cfg_path = model_dir / "config.yaml"
    SERVER_CONFIG = {
        "model_dir": str(model_dir),
        "cfg_path": str(cfg_path),
        "use_fp16": args.use_fp16,
        "use_cuda_kernel": args.use_cuda_kernel,
        "use_deepspeed": args.use_deepspeed,
    }

    try:
        if not cfg_path.exists():
            raise FileNotFoundError(f"IndexTTS2 config not found: {cfg_path}")

        from indextts.infer_v2 import IndexTTS2

        TTS = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_fp16=args.use_fp16,
            use_cuda_kernel=args.use_cuda_kernel,
            use_deepspeed=args.use_deepspeed,
        )
    except Exception as exc:
        STARTUP_ERROR = f"{exc}\n{traceback.format_exc()}"
        raise


def main():
    args = parse_args()
    load_model(args)
    server = ThreadingHTTPServer((args.host, args.port), IndexTTS2Handler)
    print(f"IndexTTS2 worker listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
