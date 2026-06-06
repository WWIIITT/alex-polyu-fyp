import base64
import io
import re
import struct

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from google.genai import types

from app.logger import get_logger
from app.api_helpers.service_helpers import error_detail
from app.utils.api_key_manager import get_llm_client, with_llm_retry_sync

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["tts"])


class TTSRequest(BaseModel):
    text: str
    voice_name: str | None = "Zephyr"
    model: str | None = None
    target_mime: str | None = "audio/wav"

def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_data)
    riff_chunk_size = 36 + data_size

    buffer = io.BytesIO()
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", riff_chunk_size))
    buffer.write(b"WAVE")
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<I", 16))
    buffer.write(struct.pack("<H", 1))
    buffer.write(struct.pack("<H", channels))
    buffer.write(struct.pack("<I", sample_rate))
    buffer.write(struct.pack("<I", byte_rate))
    buffer.write(struct.pack("<H", block_align))
    buffer.write(struct.pack("<H", sample_width * 8))
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(pcm_data)

    wav_data = buffer.getvalue()
    logger.debug("[WAV] sample_rate=%s total=%s", sample_rate, len(wav_data))
    return wav_data


def _synthesize_once(
    api_key: str,
    text: str,
    voice_name: str | None,
    model_name: str | None,
) -> tuple[bytes, str]:
    del model_name
    _model = "tts-1"

    client = get_llm_client(api_key)
    voice_map = {
        "Zephyr": "alloy",
        "Puck": "echo",
        "Charon": "fable",
        "Kore": "onyx",
        "Fenrir": "nova",
        "Aoede": "shimmer",
    }
    openai_voice = voice_map.get(voice_name or "Zephyr", "alloy")

    response = client.audio.speech.create(
        model=_model,
        voice=openai_voice,
        input=text,
        response_format="mp3",
    )
    audio_bytes = response.content
    mime_type = "audio/mpeg"
    logger.debug("[TTS] mime=%s raw_size=%s", mime_type, len(audio_bytes))
    return audio_bytes, mime_type


@router.post("/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail=error_detail("Text must not be empty."),
        )

    try:
        audio_bytes, mime_type = with_llm_retry_sync(
            "TTS generation",
            _synthesize_once,
            text,
            req.voice_name,
            req.model,
            error_type=HTTPException,
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.error("[TTS] generation failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_detail("TTS generation failed.", details=str(error)),
        ) from error

    logger.debug("[TTS] final_return mime=%s size=%s", mime_type, len(audio_bytes))
    return Response(content=audio_bytes, media_type=mime_type or "audio/mpeg")
