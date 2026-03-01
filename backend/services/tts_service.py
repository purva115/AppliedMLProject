"""
Text-to-Speech Service – converts text to MP3 using gTTS.
"""
import io
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)


def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    """
    Convert *text* to speech and return the raw MP3 bytes.
    Caller is responsible for streaming or saving the bytes.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.error("TTS conversion failed: %s", exc)
        raise


def text_to_speech_file(text: str, output_path: str, lang: str = "en") -> str:
    """Save speech MP3 to *output_path* and return the path."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    logger.info("TTS audio saved to %s", output_path)
    return output_path
