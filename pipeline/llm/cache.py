import hashlib
import json
import os
from pathlib import Path


def cache_key(system, user):
    payload = (system + "\n" + user).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResponseCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.directory / f"{key}.json"

    def get(self, key):
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        response = data.get("response")
        return response if isinstance(response, str) else None

    def set(self, key, response):
        path = self._path(key)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)