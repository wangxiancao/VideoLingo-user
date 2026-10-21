from pipeline.asr.factory import create_backend
from pipeline.asr.glm import _batched_results


class _Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


def test_default_batch_size_is_twenty():
    backend = create_backend(_Config())
    assert backend.batch_size == 20


def test_batched_results_halves_on_oom():
    calls = []

    def process(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        return list(batch)

    assert _batched_results(list(range(8)), 8, process, "test") == list(range(8))
    assert calls == [8, 4, 2, 2, 2, 2]