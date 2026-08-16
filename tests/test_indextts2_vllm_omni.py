import base64
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.tts_backend import indextts2_tts as client


class _SpeechResponse:
    def __init__(self, status_code=200, content=b"RIFFtest-wave"):
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size=8192):
        del chunk_size
        if self.content:
            yield self.content

    def json(self):
        raise ValueError("not JSON")


class IndexTTS2VllmOmniClientTests(unittest.TestCase):
    def setUp(self):
        client.SERVER_PROCESS = None
        client.SERVER_LOG_HANDLE = None
        client.REFERENCE_CACHE = None
        self.config = {
            "indextts2.host": "127.0.0.1",
            "indextts2.port": 18721,
            "indextts2.base_url": "http://127.0.0.1:18721",
            "indextts2.model": "IndexTeam/IndexTTS-2",
            "indextts2.max_concurrency": 3,
            "indextts2.request_timeout": 30,
            "indextts2.startup_timeout": 10,
            "indextts2.auto_start": True,
            "indextts2.api_key": "test-secret",
            "indextts2.vllm_omni_python": "/unused/python",
            "indextts2.preset_voice": "female",
            "indextts2.preset_voices": {"female": "unused-reference.wav"},
        }

    def load_key(self, key):
        if key in self.config:
            return self.config[key]
        raise KeyError(key)

    def test_indextts2_tts_preserves_the_existing_call_signature(self):
        signature = inspect.signature(client.indextts2_tts)

        self.assertEqual(
            list(signature.parameters),
            ["text", "save_path", "number", "task_df"],
        )
        self.assertIsNone(signature.parameters["number"].default)
        self.assertIsNone(signature.parameters["task_df"].default)

    def test_request_concurrency_returns_the_configured_value(self):
        with patch.object(client, "load_key", side_effect=self.load_key):
            self.assertEqual(client.request_concurrency(), 3)

    def test_request_concurrency_never_returns_less_than_one(self):
        self.config["indextts2.max_concurrency"] = 0

        with patch.object(client, "load_key", side_effect=self.load_key):
            self.assertEqual(client.request_concurrency(), 1)

    def test_ensure_server_does_not_start_a_process_when_health_check_succeeds(self):
        healthy = MagicMock(status_code=200)

        with (
            patch.object(client, "load_key", side_effect=self.load_key),
            patch.object(client.requests, "get", return_value=healthy),
            patch.object(client.subprocess, "Popen") as popen,
        ):
            client.ensure_server()

        popen.assert_not_called()

    def test_ensure_server_starts_a_process_when_health_check_fails(self):
        unhealthy = MagicMock(status_code=503)

        with (
            patch.object(client, "load_key", side_effect=self.load_key),
            patch.object(
                client.requests,
                "get",
                return_value=unhealthy,
            ),
            patch.object(client, "_start_server") as start_server,
        ):
            client.ensure_server()

        start_server.assert_called_once_with()

    def test_speech_request_uses_openai_audio_payload_with_reference_data_url(self):
        reference_bytes = b"RIFF-reference-wave"
        response = _SpeechResponse()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "voice.wav"
            reference.write_bytes(reference_bytes)
            output = tmp_path / "speech.wav"

            with (
                patch.object(client, "load_key", side_effect=self.load_key),
                patch.object(client, "ensure_server"),
                patch.object(
                    client,
                    "_resolve_reference_audio",
                    return_value=reference,
                ),
                patch.object(
                    client.requests,
                    "post",
                    return_value=response,
                ) as post,
            ):
                client.indextts2_tts("hello", output)

        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        headers = post.call_args.kwargs["headers"]
        self.assertTrue(url.endswith("/v1/audio/speech"))
        self.assertEqual(payload["model"], "IndexTeam/IndexTTS-2")
        self.assertEqual(payload["input"], "hello")
        self.assertEqual(payload["response_format"], "wav")
        prefix, encoded = payload["ref_audio"].split(",", 1)
        self.assertEqual(prefix, "data:audio/wav;base64")
        self.assertEqual(base64.b64decode(encoded), reference_bytes)
        self.assertEqual(headers["Authorization"], "Bearer test-secret")

    def test_successful_speech_response_is_written_as_a_complete_wav(self):
        wav_bytes = b"RIFF\x10\x00\x00\x00WAVEfmt "

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "voice.wav"
            reference.write_bytes(b"reference")
            output = tmp_path / "nested" / "speech.wav"

            with (
                patch.object(client, "load_key", side_effect=self.load_key),
                patch.object(client, "ensure_server"),
                patch.object(
                    client,
                    "_resolve_reference_audio",
                    return_value=reference,
                ),
                patch.object(
                    client.requests,
                    "post",
                    return_value=_SpeechResponse(content=wav_bytes),
                ),
            ):
                client.indextts2_tts("hello", output)

            self.assertEqual(output.read_bytes(), wav_bytes)
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_non_200_response_raises_without_replacing_an_existing_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "voice.wav"
            reference.write_bytes(b"reference")
            output = tmp_path / "speech.wav"
            output.write_bytes(b"previous-wave")

            with (
                patch.object(client, "load_key", side_effect=self.load_key),
                patch.object(client, "ensure_server"),
                patch.object(
                    client,
                    "_resolve_reference_audio",
                    return_value=reference,
                ),
                patch.object(
                    client.requests,
                    "post",
                    return_value=_SpeechResponse(
                        status_code=500,
                        content=b"server error",
                    ),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    client.indextts2_tts("hello", output)

            self.assertEqual(output.read_bytes(), b"previous-wave")

    def test_empty_response_raises_without_creating_a_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "voice.wav"
            reference.write_bytes(b"reference")
            output = tmp_path / "speech.wav"

            with (
                patch.object(client, "load_key", side_effect=self.load_key),
                patch.object(client, "ensure_server"),
                patch.object(
                    client,
                    "_resolve_reference_audio",
                    return_value=reference,
                ),
                patch.object(
                    client.requests,
                    "post",
                    return_value=_SpeechResponse(content=b""),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    client.indextts2_tts("hello", output)

            self.assertFalse(output.exists())

    def test_stop_does_not_touch_an_external_healthy_server(self):
        healthy = MagicMock(status_code=200)

        with (
            patch.object(client, "load_key", side_effect=self.load_key),
            patch.object(client.requests, "get", return_value=healthy),
            patch.object(client.subprocess, "Popen") as popen,
            patch.object(client.os, "killpg") as killpg,
        ):
            client.ensure_server()
            client.stop_indextts2_worker()

        popen.assert_not_called()
        killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
