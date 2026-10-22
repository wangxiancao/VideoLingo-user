"""V4-1 spike: minimal OpenAI-compatible chat server for HY-MT2-1.8B.

One-off benchmark harness for spec #32 V4-1 (research report docs/agents/
v4-1-local-llm-research.md §5.2/§5.3). Not product code: the production
service lifecycle is V4-2. Serves /v1/chat/completions with greedy decoding
so the pipeline's OpenAICompatibleClient can call it unchanged.

Usage:
    .venv/bin/python bench/v4_1_server.py --model-dir _model_cache/HY-MT2-1.8B \
        --port 9931
"""

import argparse
import os
import threading
import time

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

LOAD_SECONDS = None
GENERATE_LOCK = threading.Lock()
GENERATE_KWARGS = {}


def load(model_dir: str):
    global LOAD_SECONDS
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # HY-MT2 ships its chat template as a sidecar file; the tokenizer does
    # not carry it in tokenizer_config.json.
    template_path = os.path.join(model_dir, "chat_template.jinja")
    with open(template_path, encoding="utf-8") as handle:
        tokenizer.chat_template = handle.read()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    LOAD_SECONDS = time.perf_counter() - started
    return tokenizer, model


TOKENIZER, MODEL = None, None
APP = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "hy-mt2-1.8b"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


@APP.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    messages = [message.model_dump() for message in request.messages]
    prompt = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    prompt_tokens = int(inputs.input_ids.shape[1])
    max_new_tokens = request.max_tokens or 1024
    started = time.perf_counter()
    with GENERATE_LOCK, torch.inference_mode():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **GENERATE_KWARGS,
        )
    elapsed = time.perf_counter() - started
    generated = output[0][prompt_tokens:]
    text = TOKENIZER.decode(generated, skip_special_tokens=True)
    completion_tokens = int(generated.shape[0])
    return {
        "id": "v4-1-spike",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "_timing": {
            "decode_s": round(elapsed, 3),
            "tok_per_s": round(completion_tokens / elapsed, 1) if elapsed else None,
        },
    }


@APP.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "hy-mt2-1.8b", "object": "model"}]}


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "load_s": round(LOAD_SECONDS, 2) if LOAD_SECONDS else None,
        "torch_max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "torch_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
    }


def main():
    global TOKENIZER, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=9931)
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="model config default is 1.05; 1.0 disables it")
    args = parser.parse_args()
    GENERATE_KWARGS["repetition_penalty"] = args.repetition_penalty
    TOKENIZER, MODEL = load(args.model_dir)
    print(f"model loaded in {LOAD_SECONDS:.1f}s", flush=True)
    uvicorn.run(APP, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
