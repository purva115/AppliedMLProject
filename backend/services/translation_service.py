"""
Translation Service – auto-detect language and translate via deep-translator.
"""
import logging
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Return ISO-639-1 language code (e.g. 'en', 'es', 'fr')."""
    try:
        return detect(text)
    except LangDetectException:
        logger.warning("Language detection failed, defaulting to 'en'.")
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate *text* to English. Returns original text if already English."""
    lang = source_lang if source_lang != "auto" else detect_language(text)
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source=lang, target="en").translate(text)
    except Exception as exc:
        logger.error("Translation to English failed: %s", exc)
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate *text* from English to *target_lang*."""
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as exc:
        logger.error("Translation from English failed: %s", exc)
        return text
