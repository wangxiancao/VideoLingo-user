import os
import json
import re
from threading import Lock
import json_repair
import httpx
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'


def _is_local_qwen36(base_url, model):
    base_url = str(base_url or "").lower()
    model = str(model or "").lower()
    return (
        "127.0.0.1:18722" in base_url
        or "localhost:18722" in base_url
        or "qwen3.6" in model
    )


def _prepare_prompt_for_model(prompt, base_url, model):
    """Keep local Qwen3.x in fast non-thinking mode for pipeline calls."""
    if not _is_local_qwen36(base_url, model):
        return prompt
    stripped = prompt.lstrip()
    if stripped.startswith(("/no_think", "/think")):
        return prompt
    return (
        "/no_think\n"
        "Return only the requested final answer. Do not output hidden reasoning, "
        "analysis outside the requested JSON, or extra prose.\n\n"
        f"{prompt}"
    )


def _strip_thinking_blocks(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    request_prompt = _prepare_prompt_for_model(prompt, base_url, model)

    # Bypass SOCKS proxy (httpx doesn't support socks:// without socksio)
    http_client = httpx.Client(proxy=None, transport=httpx.HTTPTransport())
    client = OpenAI(api_key=load_key("api.key"), base_url=base_url, http_client=http_client)
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = [{"role": "user", "content": request_prompt}]

    params = dict(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=300
    )
    resp_raw = client.chat.completions.create(**params)

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = json_repair.loads(_strip_thinking_blocks(resp_content))
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
