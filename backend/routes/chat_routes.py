"""
Chat Routes – REST endpoints for the PharmaLLM chatbot.
"""
import io
import logging
from flask import Blueprint, request, jsonify, current_app, send_file

from services.nlp_service import preprocess_query
from services.translation_service import detect_language, translate_to_english, translate_from_english
from services.tts_service import text_to_speech_bytes
from services.stt_service import transcribe_audio_bytes
from models.inference import generate_response

chat_bp = Blueprint("chat", __name__)
logger = logging.getLogger(__name__)


@chat_bp.route("/text", methods=["POST"])
def chat_text():
    """
    POST /api/chat/text
    Body: {"message": str, "language": str (optional, default "en")}
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    target_lang = data.get("language", "en")
    model_loader = current_app.config["MODEL_LOADER"]

    # Translate to English if needed
    en_message = translate_to_english(user_message)

    # Preprocess & generate
    clean_query = preprocess_query(en_message)
    response_en = generate_response(model_loader, clean_query)

    # Translate response back to user's language
    response_out = translate_from_english(response_en, target_lang)

    return jsonify({"response": response_out, "detected_language": target_lang})


@chat_bp.route("/speech", methods=["POST"])
def chat_speech():
    """
    POST /api/chat/speech
    Form-data: audio file (field name 'audio')
    Returns JSON with transcription + text response + audio bytes URL
    """
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    audio_fmt = audio_file.filename.rsplit(".", 1)[-1] if "." in audio_file.filename else "webm"

    # Transcribe
    stt_result = transcribe_audio_bytes(audio_bytes, audio_format=audio_fmt)
    transcribed = stt_result["text"]
    lang = stt_result.get("language", "en")

    model_loader = current_app.config["MODEL_LOADER"]

    en_message = translate_to_english(transcribed, source_lang=lang)
    clean_query = preprocess_query(en_message)
    response_en = generate_response(model_loader, clean_query)
    response_out = translate_from_english(response_en, lang)

    # TTS
    audio_response = text_to_speech_bytes(response_out, lang=lang)

    return jsonify({
        "transcribed": transcribed,
        "response": response_out,
        "language": lang,
        "audio_base64": audio_response.hex(),   # frontend can decode
    })


@chat_bp.route("/tts", methods=["POST"])
def tts():
    """
    POST /api/chat/tts
    Body: {"text": str, "lang": str}
    Returns: MP3 audio stream
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    lang = data.get("lang", "en")
    audio_bytes = text_to_speech_bytes(text, lang=lang)
    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="response.mp3",
    )
