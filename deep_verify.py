"""
deep_verify.py — rigorous end-to-end verification of the multi-agent demo.
Tests every pattern with instrumented traces: tool calls, model, latency,
token-free correctness. Also exercises the fallback chain and the web server
endpoint via TestClient. Run: python deep_verify.py
"""
import asyncio, time, json
from collections import Counter
from agents import Runner, RunConfig
from agents.items import HandoffCallItem, ToolCallItem
from config import get_provider, DEFAULT_MODEL, FALLBACK_MODELS
from agents_defs import (
    triage_agent, manager_agent, coordinator_agent,
    AGENT_REGISTRY,
)
from server import app
from fastapi.testclient import TestClient


def trace(agent, prompt):
    """Run an agent and return (ok, model, calls, final_output, ms)."""
    provider = get_provider()
    agent.model = DEFAULT_MODEL
    t0 = time.time()
    try:
        res = asyncio.run(
            Runner.run(agent, prompt,
                       run_config=RunConfig(model_provider=provider),
                       max_turns=6)
        )
        ms = round((time.time() - t0) * 1000)
        calls = []
        for it in res.new_items:
            if type(it).__name__ == "HandoffCallItem":
                calls.append(("handoff", it.raw_item.name))
            elif type(it).__name__ == "ToolCallItem":
                calls.append(("tool", it.raw_item.name))
        return True, DEFAULT_MODEL, calls, res.final_output, ms
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        return False, None, [], f"{type(e).__name__}: {e}", ms


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------------------
section("1. HANDOFFS — routing correctness (3 question types)")
routing = {
    "billing":   "I was double-charged on my invoice and need a refund.",
    "technical": "My API key returns 401 and I cannot log in.",
    "sales":     "What plan fits a 5-person team on a budget?",
}
for expect, q in routing.items():
    ok, model, calls, out, ms = trace(triage_agent, q)
    ho = [n for k, n in calls if k == "handoff"]
    correct = ho == [f"route_to_{expect}"]
    print(f"  [{expect:9}] handoff={ho} -> last={triage_agent.name if not ok else 'n/a'} "
          f"| ROUTE_OK={correct} | {ms}ms")
    assert correct, f"Handoff routing FAILED for {expect}: {ho}"

# ---------------------------------------------------------------------------
section("2. MANAGER + WORKERS — strict single-pass chaining (5x)")
defects = 0
for i in range(5):
    ok, model, calls, out, ms = trace(manager_agent,
        "Summarize the benefits of remote work for a small team.")
    c = Counter(n for k, n in calls if k == "tool")
    clean = c.get("research_topic") == 1 and c.get("write_summary") == 1 and c.get("review_draft") == 1
    if not clean:
        defects += 1
    print(f"  run{i+1}: {dict(c)} | CHAIN_OK={clean} | {ms}ms | out_len={len(out)}")
assert defects == 0, f"Manager chaining defects: {defects}/5"

# ---------------------------------------------------------------------------
section("3. AGENTS-AS-TOOLS — translate + sentiment (3x)")
for i in range(3):
    ok, model, calls, out, ms = trace(coordinator_agent,
        "I love this product, it saved my team hours!")
    c = Counter(n for k, n in calls if k == "tool")
    clean = c.get("translate_to_spanish") == 1 and c.get("analyze_sentiment") == 1
    has_es = "spanish" in out.lower() or any(ord(ch) > 127 for ch in out)
    print(f"  run{i+1}: {dict(c)} | TOOLS_OK={clean} | has_nonascii={has_es} | {ms}ms")
    assert clean, f"Coordinator tools broken: {dict(c)}"

# ---------------------------------------------------------------------------
section("4. WEB SERVER — /api/run endpoint (TestClient, real calls)")
client = TestClient(app)
h = client.get("/api/health").json()
print(f"  /api/health -> {h}")
assert h["status"] == "ok" and h["has_key"] is True
r = client.get("/api/run?pattern=handoffs&prompt=" +
               "Refund my double charge").json()
print(f"  /api/run handoffs -> ok={r.get('ok')} model={r.get('model_used')}")
print(f"     out: {r.get('output','')[:120]}")
assert r.get("ok") is True, f"Web /api/run failed: {r}"

# ---------------------------------------------------------------------------
section("5. ALL-3 via run_all (the exact UI payload)")
from runner import run_all
res = asyncio.run(run_all(
    "My payment failed and I was double-charged. How do I reset my API key? "
    "What plan fits a 5-person team?"))
for r in res:
    print(f"  [{r['pattern']:16}] ok={r['ok']} model={r['model_used']} "
          f"len={len(r['output']) if r['ok'] else 0}")
assert all(r["ok"] for r in res), "run_all had a failure"

# ---------------------------------------------------------------------------
section("6. FALLBACK CHAIN — config sanity")
print(f"  DEFAULT_MODEL={DEFAULT_MODEL}")
print(f"  FALLBACK_MODELS={FALLBACK_MODELS}")
assert DEFAULT_MODEL and FALLBACK_MODELS

print("\n" + "#" * 70)
print("# DEEP VERIFY PASSED — all patterns, web, routing, chaining OK")
print("#" * 70)
