import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are KYRO, a polished personal AI assistant.
Your personality: smart, friendly, confident, practical, and slightly playful.
Be concise when the user asks something simple, but give enough detail when the task needs it.
Use clear sections, short paragraphs, bullets, numbered steps, and code blocks when useful.
Never invent facts, links, results, or actions.
When you are unsure, say so instead of guessing.
For coding, give working code and simple setup steps.
For project ideas, think creatively and help turn ideas into real builds.
Match the user's casual tone without becoming unclear or sloppy.
Prioritize actually solving the user's problem over giving generic advice.
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
            model=os.getenv("KYRO_MODEL", "gemini-3.5-flash-lite"),
            contents=f"{SYSTEM_PROMPT}\n\nUser: {message}",
        )
        return jsonify({"reply": response.text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
