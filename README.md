# OpenAI Agents SDK — Multi-Agent Demo

A hands-free demo of **three** canonical OpenAI Agents SDK multi-agent patterns,
wired to **OpenRouter** (any model, one key) and served by a FastAPI web UI that
**auto-runs all patterns on page load** — no buttons to click.

## Patterns
1. **Handoffs** — `TriageAgent` routes to Billing / Technical / Sales specialists via `handoff()`.
2. **Manager + Workers** — `ManagerAgent` delegates to worker agents exposed as tools.
3. **Agents-as-Tools** — `CoordinatorAgent` calls a translator + sentiment agent as tools.

Per-agent fallback: `runner.py` tries `DEFAULT_MODEL`, then each `FALLBACK_MODELS`
entry on rate-limit / 429.

## Quick start
```bash
uv venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then set OPENROUTER_API_KEY (https://openrouter.ai/keys)
run.bat                       # Windows  (or ./run.sh on macOS/Linux)
```
Open http://127.0.0.1:8000 — it auto-runs. Build docs at /docs.

## Files
| File | Role |
|------|------|
| `config.py` | OpenRouter `AsyncOpenAI` client + `OpenRouterModelProvider` |
| `agents_defs.py` | The 3 patterns as SDK `Agent` objects |
| `runner.py` | Executes a pattern with default→fallback model chain |
| `server.py` | FastAPI app + `/api/run`, `/api/health` |
| `static/index.html` | UI, auto-runs on load |
| `docs.html` | Build documentation |

## API
```
GET /api/run?pattern=all&prompt=...      # run every pattern
GET /api/run?pattern=handoffs&prompt=... # single pattern
GET /api/health                          # {status, has_key}
```

## Note (Windows / Hermes venv gotcha)
The launch scripts `unset PYTHONPATH` (run.bat / run.sh) so the global Hermes
venv doesn't clobber this project's venv's pydantic/openai. If you run
`python server.py` directly, export `PYTHONPATH=` first.
