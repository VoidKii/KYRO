import os
from flask import Flask, render_template, request, jsonify
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

    try:
        response = client.models.generate_content(
            model=os.getenv("KYRO_MODEL", "gemini-3.8-flash"),
            contents=f"{SYSTEM_PROMPT}\n\nUser: {message}",
        )
        return jsonify({"reply": response.text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
