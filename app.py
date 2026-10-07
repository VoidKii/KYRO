import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are KYRO, a powerful personal AI assistant.
Be friendly, direct, practical, and concise.
Help the user learn, code, build projects, brainstorm, analyze files, and solve problems.
Use conversation history and the user's saved memory when provided.
Never pretend you performed an action you did not perform.
When web search is enabled, use current web information when it improves accuracy and provide source-aware answers.
When code execution is enabled, use it for calculations, data analysis, and Python-based reasoning when useful.
When a file is attached, inspect it carefully and answer based on its contents.
"""

MAX_HISTORY = 16
MAX_FILE_BYTES = 8 * 1024 * 1024
ALLOWED_FILE_PREFIXES = ("image/", "text/")
ALLOWED_FILE_TYPES = {
    "application/pdf",
    "application/json",
    "application/javascript",
    "text/csv",
    "text/javascript",
    "application/xml",
}


def clean_history(raw_history):
    try:
        history = json.loads(raw_history or "[]")
    except (TypeError, ValueError):
        return []

    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = str(item.get("text", "")).strip()

        if role in ("user", "model") and text:
            cleaned.append({"role": role, "text": text})

    return cleaned


def extract_sources(response):
    sources = []
    seen = set()

    try:
        metadata = response.candidates[0].grounding_metadata
        chunks = metadata.grounding_chunks or []
    except (AttributeError, IndexError, TypeError):
        return sources

    for chunk in chunks:
        try:
            web = chunk.web
            uri = web.uri
            title = web.title or uri
        except AttributeError:
            continue

        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": title, "url": uri})

    return sources[:8]


def request_value(name, default=""):
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return data.get(name, default)
    return request.form.get(name, default)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is not configured yet."}), 500

    message = str(request_value("message", "")).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = clean_history(request_value("history", "[]"))
    memory = str(request_value("memory", "")).strip()
    use_web = str(request_value("web", "false")).lower() == "true"
    use_code = str(request_value("code", "false")).lower() == "true"

    uploaded = request.files.get("file")
    file_part = None

    if uploaded and uploaded.filename:
        uploaded.stream.seek(0, os.SEEK_END)
        file_size = uploaded.stream.tell()
        uploaded.stream.seek(0)

        if file_size > MAX_FILE_BYTES:
            return jsonify({"error": "File is too large. Maximum size is 8 MB."}), 400

        mime_type = uploaded.mimetype or "application/octet-stream"

        if not (
            mime_type in ALLOWED_FILE_TYPES
            or any(mime_type.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES)
        ):
            return jsonify({
                "error": "That file type is not supported yet. Use an image, PDF, text, JSON, CSV, or code file."
            }), 400

        file_part = types.Part.from_bytes(
            data=uploaded.read(),
            mime_type=mime_type,
        )

    try:
        # Keep the default path intentionally simple and close to the original
        # working KYRO implementation. Advanced features only opt into the
        # structured/tool path when the user explicitly enables them.
        advanced = bool(history or memory or use_web or use_code or file_part)

        if not advanced:
            response = client.models.generate_content(
                model=os.getenv("KYRO_MODEL", "gemini-3.5-flash-lite"),
                contents=f"{SYSTEM_PROMPT}\n\nUser: {message}",
            )
        else:
            contents = []

            for item in history:
                contents.append(
                    types.Content(
                        role=item["role"],
                        parts=[types.Part.from_text(text=item["text"])],
                    )
                )

            context_parts = []
            if memory:
                context_parts.append(
                    types.Part.from_text(
                        text="Saved user memory:\n" + memory[:6000]
                    )
                )

            context_parts.append(types.Part.from_text(text=message))

            if file_part is not None:
                context_parts.append(file_part)

            contents.append(types.Content(role="user", parts=context_parts))

            tools = []
            if use_web:
                tools.append(
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                )
            if use_code:
                tools.append(
                    types.Tool(
                        code_execution=types.ToolCodeExecution()
                    )
                )

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
                tools=tools or None,
            )

            response = client.models.generate_content(
                model=os.getenv("KYRO_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=config,
            )

        reply = getattr(response, "text", None) or "I couldn't generate a response."

        return jsonify({
            "reply": reply,
            "sources": extract_sources(response),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
