"""
runner.py — executes each multi-agent pattern with a fallback model chain.

The OpenAI Agents SDK's Runner.run() takes an agent + input. We override the
agent's model at call time (by re-binding `agent.model`) so we can try
DEFAULT_MODEL, then FALLBACK_MODELS on failure — exactly the per-agent
fallback behavior requested.

Because worker agents are invoked *inside* tools via Runner.run, we must also
bind the model on every worker referenced by a pattern. We do this by mapping
agent objects -> model name and patching each one in place (agents are
lightweight; we patch the live object, which is safe within a single run).
"""
import asyncio
from agents import Runner, RunConfig
from config import get_provider, DEFAULT_MODEL, FALLBACK_MODELS
from agents_defs import AGENT_REGISTRY

# Map every worker agent object -> its parent pattern so we can bind models
# on nested agents too. Workers are reachable from the registry agents' tools.
from agents_defs import (
    research_worker, writer_worker, reviewer_worker,
    translator_agent, sentiment_agent,
)


def _collect_agents(pattern: str):
    """Return the list of agent objects used by a pattern (top + nested)."""
    top = AGENT_REGISTRY[pattern]
    agents = [top]
    if pattern == "manager_workers":
        agents.extend([research_worker, writer_worker, reviewer_worker])
    elif pattern == "agents_as_tools":
        agents.extend([translator_agent, sentiment_agent])
    return agents


async def run_pattern(pattern: str, user_input: str) -> dict:
    """Run a pattern. Tries DEFAULT_MODEL then fallbacks. Returns a result dict."""
    provider = get_provider()
    models_to_try = [DEFAULT_MODEL] + FALLBACK_MODELS
    target_agents = _collect_agents(pattern)

    last_err = None
    for model_name in models_to_try:
        # Bind model on top-level + every nested worker for this attempt.
        for ag in target_agents:
            ag.model = model_name
        try:
            result = await Runner.run(
                starting_agent=AGENT_REGISTRY[pattern],
                input=user_input,
                max_turns=12,
                run_config=RunConfig(model_provider=provider),
            )
            return {
                "pattern": pattern,
                "model_used": model_name,
                "output": result.final_output,
                "ok": True,
                "error": None,
            }
        except Exception as e:  # rate-limit / model unavailable -> try next
            last_err = str(e)
            lowered = last_err.lower()
            if "rate" in lowered or "429" in lowered or "model" in lowered:
                continue
            break  # real bug — do not retry

    return {
        "pattern": pattern,
        "model_used": None,
        "output": None,
        "ok": False,
        "error": last_err or "all models failed",
    }


async def run_all(user_input: str) -> list[dict]:
    results = []
    for pattern in ["handoffs", "manager_workers", "agents_as_tools"]:
        results.append(await run_pattern(pattern, user_input))
    return results
