import base64
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.routers import tts
from tests.support import build_client


def inline_audio_part(mime_type):
    return SimpleNamespace(inline_data=SimpleNamespace(mime_type=mime_type))


def tts_response(*, parts=None, candidates=None):
    return SimpleNamespace(parts=parts or [], candidates=candidates or [])


def candidate(parts=None, content=True):
    return SimpleNamespace(content=SimpleNamespace(parts=parts or []) if content else None)


class TtsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = build_client(tts.router)

    def test_pcm_to_wav_wraps_pcm_payload(self):
        wav = tts.pcm_to_wav(b"\x00\x01\x02\x03", sample_rate=16000)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:12])

    def test_synthesize_once_maps_voice_and_returns_audio_bytes(self):
        response = SimpleNamespace(content=b"audio")
        audio = SimpleNamespace(speech=SimpleNamespace(create=Mock(return_value=response)))
        client = SimpleNamespace(audio=audio)

        with patch("app.routers.tts.get_llm_client", return_value=client):
            data, mime_type = tts._synthesize_once("api-key", "hello", "Kore", None)

        self.assertEqual(data, b"audio")
        self.assertEqual(mime_type, "audio/mpeg")
        self.assertEqual(client.audio.speech.create.call_args.kwargs["voice"], "onyx")

    def test_tts_route_rejects_empty_text(self):
        response = self.client.post("/tts", json={"text": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error"], "Text must not be empty.")

    def test_tts_route_returns_audio_bytes(self):
        with patch("app.routers.tts.with_llm_retry_sync", return_value=(b"audio", "audio/mpeg")):
            response = self.client.post("/tts", json={"text": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"audio")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")

    def test_tts_route_maps_generation_failures(self):
        cases = (
            (RuntimeError("boom"), 500, "TTS generation failed."),
            (tts.HTTPException(status_code=429, detail={"error": "rate limited"}), 429, "rate limited"),
        )
        for side_effect, status, error in cases:
            with self.subTest(status=status), patch("app.routers.tts.with_llm_retry_sync", side_effect=side_effect):
                response = self.client.post("/tts", json={"text": "Hello"})
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["detail"]["error"], error)

