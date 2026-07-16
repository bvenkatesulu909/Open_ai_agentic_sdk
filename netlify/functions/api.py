"""
Netlify Function: serverless backend for the multi-agent demo.
Serves /api/run and /api/health. Netlify routes /api/* to this function.

Netlify Python functions use the `requests`-style handler:
    def handler(event, context) -> dict

We reuse the project's runner + agent definitions. The OpenRouter key comes
from the site's environment variables (set in Netlify dashboard), NOT committed.
"""
import json
import os
import asyncio

# Netlify injects env vars at runtime; load .env if present for local emulation.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Make sure the project modules are importable (function cwd is the repo root).
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import run_all, run_pattern
from config import has_key


def _respond(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps(body),
    }


def handler(event, context):
    try:
        path = event.get("path", "")
        qs = event.get("queryStringParameters") or {}
        # Netlify may pass the function at /api or /api/ ; normalize.
        pattern = (qs.get("pattern") or "all").lower()
        prompt = qs.get("prompt") or (
            "My payment failed and I was double-charged. Also, how do I reset "
            "my API key? And what plan fits a 5-person team on a budget?"
        )

        if path.rstrip("/").endswith("/health") or pattern == "health":
            return _respond(200, {"status": "ok", "has_key": has_key()})

        if not has_key():
            return _respond(200, {
                "error": "NO_API_KEY",
                "message": "Set OPENROUTER_API_KEY in Netlify site env vars.",
            })

        if pattern == "all":
            result = asyncio.run(run_all(prompt))
            return _respond(200, {"results": result})
        else:
            result = asyncio.run(run_pattern(pattern, prompt))
            return _respond(200, result)
    except Exception as e:  # never leak raw trace; return json error
        return _respond(500, {"error": "SERVER_ERROR", "message": str(e)[:300]})
