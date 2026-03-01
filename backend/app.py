"""
PharmaLLM – Flask Backend Entry Point
"""
from flask import Flask, jsonify
from flask_cors import CORS

from config import HOST, PORT, DEBUG, CORS_ORIGINS
from models.model_loader import ModelLoader
from routes.chat_routes import chat_bp

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# ── Load model ─────────────────────────────────────────────────────────────────
model_loader = ModelLoader()
app.config["MODEL_LOADER"] = model_loader

# ── Register blueprints ────────────────────────────────────────────────────────
app.register_blueprint(chat_bp, url_prefix="/api/chat")


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
    })


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
