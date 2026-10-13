"""Voice features: speech to text and text to speech via Groq."""

import os
import tempfile

import config
from llm import client


def transcribe(audio_bytes: bytes) -> str:
    """Convert a recorded WAV clip into text."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        tmp.close()
        with open(tmp.name, "rb") as handle:
            result = client.audio.transcriptions.create(
                file=handle, model=config.STT_MODEL, response_format="text"
            )
    finally:
        os.unlink(tmp.name)
    return result if isinstance(result, str) else getattr(result, "text", str(result))


def synthesize(text: str) -> bytes:
    """Convert text into WAV audio bytes."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        response = client.audio.speech.create(
            model=config.TTS_MODEL, voice=config.TTS_VOICE, input=text, response_format="wav"
        )
        response.write_to_file(tmp.name)
        with open(tmp.name, "rb") as handle:
            return handle.read()
    finally:
        os.unlink(tmp.name)