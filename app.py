import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are KYRO, a helpful personal AI assistant.
Be friendly, clear, practical, and concise.
Your job is to help the user learn, build projects, solve problems, and brainstorm ideas.
Never pretend to have done something you cannot do.
"""

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/chat")
def chat():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is not configured yet."}), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    def generate():
        try:
            stream = client.models.generate_content_stream(
                model=os.getenv("KYRO_MODEL", "gemini-3.5-flash-lite"),
                contents=f"{SYSTEM_PROMPT}\n\nUser: {message}",
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"\n[ERROR] {exc}"

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
