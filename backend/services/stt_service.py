"""
Speech-to-Text Service – transcribe audio using OpenAI Whisper.
"""
import os
import logging
import tempfile
import whisper

logger = logging.getLogger(__name__)

# Load Whisper model once at import time (thread-safe after loading)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WHISPER_MODEL

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL)
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_audio_bytes(audio_bytes: bytes, audio_format: str = "webm") -> dict:
    """
    Transcribe raw audio bytes.
    Returns: {"text": str, "language": str}
    """
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = _get_model().transcribe(tmp_path)
        return {"text": result["text"].strip(), "language": result.get("language", "en")}
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise
    finally:
        os.unlink(tmp_path)


def transcribe_audio_file(file_path: str) -> dict:
    """Transcribe an audio file from disk."""
    result = _get_model().transcribe(file_path)
    return {"text": result["text"].strip(), "language": result.get("language", "en")}
