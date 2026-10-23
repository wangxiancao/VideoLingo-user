"""Round-robin client across LLM service replicas (V4-7 O2).

The translate stage fires concurrent batch requests from a thread pool;
distributing them across N llama-server replicas multiplies decode
throughput. Failures surface as the underlying client's LlmError — no
silent fallback to a broken replica.
"""

import threading

from pipeline.llm.contracts import LlmError


class RoundRobinClient:
    """Distributes complete() calls across N LlmClient-compatible replicas."""

    def __init__(self, clients):
        self._clients = list(clients)
        if not self._clients:
            raise LlmError("round-robin client needs at least one replica")
        self._lock = threading.Lock()
        self._index = 0

    def complete(self, system: str, user: str) -> str:
        with self._lock:
            index = self._index
            self._index = (self._index + 1) % len(self._clients)
        return self._clients[index].complete(system, user)
