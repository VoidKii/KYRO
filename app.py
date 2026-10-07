import json
import os

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are KYRO, a polished personal AI assistant.
Your goal is to be genuinely useful, accurate, thoughtful, and easy to talk to.

PERSONALITY
- Smart, friendly, confident, practical, and slightly playful.
- Match the user's casual tone when appropriate.
- Never be needlessly formal or robotic.

REASONING
- Understand the user's actual goal before answering.
- Think through the problem carefully before responding.
- Check your assumptions and do not invent facts.
- When the task is ambiguous, make the most reasonable interpretation and state it briefly.
- For difficult tasks, break the solution into clear steps.
- Prefer useful answers over generic explanations.

CODING
- Give working code, not vague pseudocode.
- Keep setup instructions simple.
- When debugging, identify the likely cause and give the exact fix.
- Preserve working code unless a change is actually needed.

PROJECTS
- Think creatively and turn ideas into concrete builds.
- Suggest practical next steps and anticipate obvious problems.
- Remember details from the conversation and use them naturally.

COMMUNICATION
- Simple question = concise answer.
- Complex question = enough detail to solve it.
- Use headings, bullets, tables, or code blocks when they improve clarity.
- Never claim you did something you did not do.
"""

MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12000


def get_history(raw_history):
    try:
        history = json.loads(raw_history or "[]")
    except (TypeError, ValueError):
        return []

    if not isinstance(history, list):
        return []

    clean = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = str(item.get("text", "")).strip()
        if role not in ("user", "kyro") or not text:
            continue

        clean.append((role, text))

    total = 0
    result = []
    for role, text in reversed(clean):
        remaining = MAX_HISTORY_CHARS - total
        if remaining <= 0:
            break
        text = text[:remaining]
        result.append((role, text))
        total += len(text)

    return list(reversed(result))


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is not configured yet."}), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    history = get_history(data.get("history", "[]"))

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    conversation = []
    for role, text in history:
        speaker = "User" if role == "user" else "KYRO"
        conversation.append(f"{speaker}: {text}")

    history_block = ""
    if conversation:
        history_block = (
            "Previous conversation. Use it for context and continuity; "
            "do not repeat it unless useful.\n\n"
            + "\n".join(conversation)
            + "\n\n"
        )

    prompt = f"{SYSTEM_PROMPT}\n\n{history_block}User: {message}"

    try:
        response = client.models.generate_content(
            model=os.getenv("KYRO_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
        )
        reply = response.text or "I couldn't generate a response."
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
