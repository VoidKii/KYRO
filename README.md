# KYRO

KYRO is a personal AI agent built with Flask and the Google Gen AI SDK.

## What KYRO can do

- Chat with persistent server-side Gemini interactions
- Search the web with Gemini's built-in Google Search tool
- Run code with Gemini's built-in code execution tool
- Remember useful user-scoped facts with SQLite-backed memory
- Inspect allowed GitHub repositories
- Create or update GitHub text files when GitHub write access is explicitly enabled
- Show agent activity in the UI
- Keep chat history locally in the browser

## Local setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Add your `GEMINI_API_KEY`.
5. Run `python app.py`.

## GitHub write mode

For GitHub editing, configure:

- `GITHUB_TOKEN`
- `KYRO_GITHUB_WRITE=true`
- `KYRO_GITHUB_ALLOWED_REPOS=owner/repository`

KYRO intentionally does not expose destructive repository operations.

## Health check

`GET /api/health` returns service and capability status.

## Deployment

The included `render.yaml` is configured for a Render web service using Gunicorn.
