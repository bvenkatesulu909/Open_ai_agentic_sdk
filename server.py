"""
server.py — FastAPI app.
Serves the demo UI and a /run endpoint that executes all 3 patterns.

Per your hands-free requirement: the web page auto-loads and auto-runs a
default prompt on open, then renders each pattern's live output. No buttons
to click (a "Run again" button is optional, not required).
"""
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

from runner import run_all, run_pattern
from config import has_key

app = FastAPI(
    title="OpenAI Agents SDK — Multi-Agent Demo",
    docs_url=None,        # disable auto-Swagger so our /docs = build docs page
    redoc_url=None,
    openapi_url=None,
)

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/docs", response_class=HTMLResponse)
async def docs():
    return FileResponse(os.path.join(BASE, "docs.html"))


@app.get("/api/run")
async def api_run(
    pattern: str = Query("all"),
    prompt: str = Query(
        "My payment failed and I was double-charged. Also, how do I reset my API key? "
        "And what plan fits a 5-person team on a budget?"
    ),
):
    if not has_key():
        return {
            "error": "NO_API_KEY",
            "message": "Set OPENROUTER_API_KEY in .env (see .env.example). "
                       "The server runs but agents need a key to call models.",
        }
    if pattern == "all":
        return {"results": await run_all(prompt)}
    return await run_pattern(pattern, prompt)


@app.get("/api/health")
async def health():
    return {"status": "ok", "has_key": has_key()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
