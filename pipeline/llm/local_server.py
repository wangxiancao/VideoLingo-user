"""Local OpenAI-compatible LLM server for the managed HY-MT2 service (V4-2).

Runs as a subprocess of the pipeline, spawned and supervised by
``pipeline.llm.local_service.LocalLlmService``. Decoding is fixed to greedy
with ``repetition_penalty=1.0``: the model's shipped default of 1.05 is
tuned for sampling chat and degenerates JSON output deterministically
under greedy decoding (V4-1 spike, docs/agents/v4-1-spike.md §2.2).

Usage:
    python -m pipeline.llm.local_server --model-dir <dir> --port <port>
    [--host 127.0.0.1] [--die-with-parent] [--stub] [--fail-start]

``--stub`` serves canned responses without loading the model (lifecycle
tests). ``--fail-start`` exits immediately with a message (startup-failure
tests). ``--die-with-parent`` sets PR_SET_PDEATHSIG so the OS kills this
process even if the parent is SIGKILLed.
"""

import argparse
import ctypes
import json
import os
import signal
import threading
import time

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

GENERATE_LOCK = threading.Lock()
LOAD_SECONDS = None
TOKENIZER = None
MODEL = None
STUB = False

PR_SET_PDEATHSIG = 1


def _enable_die_with_parent():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("prctl(PR_SET_PDEATHSIG) failed")


def load(model_dir: str):
    global LOAD_SECONDS, TOKENIZER, MODEL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    # HY-MT2 ships its chat template as a sidecar file; the tokenizer does
    # not carry it in tokenizer_config.json.
    with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8") as handle:
        TOKENIZER.chat_template = handle.read()
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype="bfloat16", device_map="cuda"
    )
    MODEL.eval()
    LOAD_SECONDS = time.perf_counter() - started


def create_app():
    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()
    generate_lock = GENERATE_LOCK

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "hy-mt2-1.8b"
        messages: list[Message]
        temperature: float | None = None
        max_tokens: int | None = None

    def stub_reply(request: ChatRequest) -> dict:
        user = request.messages[-1].content if request.messages else ""
        payload = {"translations": [{"id": index + 1, "target": f"stub-{index}"}
                                    for index in range(3)]}
        return {
            "id": "local-llm", "object": "chat.completion",
            "created": int(time.time()), "model": request.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": json.dumps(payload,
                                                           ensure_ascii=False)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(user), "completion_tokens": 8,
                      "total_tokens": len(user) + 8},
            "_timing": {"decode_s": 0.01, "tok_per_s": 800.0},
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        if STUB:
            return stub_reply(request)
        messages = [message.model_dump() for message in request.messages]
        prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
        prompt_tokens = int(inputs.input_ids.shape[1])
        # 1024 truncated 12-segment batches in the V4-1 spike; 2048 covers
        # the longest subtitle batches with margin.
        max_new_tokens = request.max_tokens or 2048
        started = time.perf_counter()
        with generate_lock, torch.inference_mode():
            output = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
            )
        elapsed = time.perf_counter() - started
        generated = output[0][prompt_tokens:]
        text = TOKENIZER.decode(generated, skip_special_tokens=True)
        completion_tokens = int(generated.shape[0])
        return {
            "id": "local-llm", "object": "chat.completion",
            "created": int(time.time()), "model": request.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
            "_timing": {"decode_s": round(elapsed, 3),
                        "tok_per_s": round(completion_tokens / elapsed, 1)
                        if elapsed else None},
        }

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "hy-mt2-1.8b",
                                            "object": "model"}]}

    @app.get("/health")
    def health():
        payload = {"status": "ok", "stub": STUB, "pid": os.getpid()}
        if LOAD_SECONDS is not None:
            payload["load_s"] = round(LOAD_SECONDS, 2)
        return payload

    return app


def main():
    global STUB
    parser = argparse.ArgumentParser(prog="pipeline.llm.local_server")
    parser.add_argument("--model-dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--die-with-parent", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--fail-start", action="store_true")
    args = parser.parse_args()
    STUB = args.stub

    if args.die_with_parent:
        _enable_die_with_parent()
    if args.fail_start:
        print("local_server: simulated startup failure", flush=True)
        raise SystemExit(3)
    if not args.stub:
        if not args.model_dir:
            raise SystemExit("--model-dir is required unless --stub")
        load(args.model_dir)
        print(f"model loaded in {LOAD_SECONDS:.1f}s", flush=True)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
