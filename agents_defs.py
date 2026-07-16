"""
agents_defs.py — Three canonical OpenAI Agents SDK multi-agent patterns.

SDK version targeted: openai-agents 0.18.x

1. HANDOFFS        : triage agent routes the user to a specialist agent.
2. MANAGER/WORKERS : an orchestrator agent delegates subtasks to worker
                      agents exposed as *tools* (each tool runs a sub-agent
                      via Runner.run — the universal "agent-as-tool" pattern).
3. AGENTS-AS-TOOLS : a coordinator agent calls other agents as tools
                      (translator + sentiment) on the same input.

All agents are model-agnostic: `model` is set at runtime by runner.py so we
can inject DEFAULT_MODEL or a fallback per call. Here we leave model unset
and let the runner bind it.
"""
import functools
from agents import Agent, handoff, set_tracing_disabled, function_tool, Runner
from agents.exceptions import AgentsException

set_tracing_disabled(disabled=True)  # keep demo output clean; no trace export


# ---------------------------------------------------------------------------
# Helper: turn an Agent into a callable tool by running it with Runner.
# This is the version-proof "agent as tool" — works on every SDK release.
# ---------------------------------------------------------------------------
def make_agent_tool(worker: Agent, tool_name: str, tool_description: str):
    @function_tool(name_override=tool_name, description_override=tool_description)
    async def _run(input: str) -> str:
        # Run the worker as a nested agent. The runner's model_provider +
        # the worker's own `model` (bound by runner.py) drive the call.
        result = await Runner.run(worker, input)
        return result.final_output
    return _run


# ---------------------------------------------------------------------------
# 1. HANDOFFS PATTERN
# ---------------------------------------------------------------------------
billing_agent = Agent(
    name="BillingSpecialist",
    instructions=(
        "You are a billing support specialist. Help users with invoices, "
        "refunds, subscription changes, and payment failures. Be precise and "
        "cite amounts/plan names. Keep answers under 120 words."
    ),
)

technical_agent = Agent(
    name="TechnicalSpecialist",
    instructions=(
        "You are a technical support engineer. Help users with login issues, "
        "API errors, integrations, and troubleshooting. Provide concrete steps. "
        "Keep answers under 120 words."
    ),
)

sales_agent = Agent(
    name="SalesSpecialist",
    instructions=(
        "You are a sales assistant. Help prospects understand plans, pricing, "
        "and feature fit. Be friendly and consultative. Keep answers under 120 words."
    ),
)

triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        "You are a support triage agent. Read the user message and route it to "
        "exactly ONE specialist using the available handoff tools: "
        "BillingSpecialist for billing/refunds/payments, "
        "TechnicalSpecialist for bugs/login/API/technical, "
        "SalesSpecialist for pricing/plans/sales questions. "
        "Do not answer the question yourself — hand off and stop."
    ),
    handoffs=[
        handoff(billing_agent, tool_name_override="route_to_billing"),
        handoff(technical_agent, tool_name_override="route_to_technical"),
        handoff(sales_agent, tool_name_override="route_to_sales"),
    ],
)


# ---------------------------------------------------------------------------
# 2. MANAGER + WORKER AGENTS (orchestrator delegates via agent-as-tool)
# ---------------------------------------------------------------------------
research_worker = Agent(
    name="ResearchWorker",
    instructions=(
        "You research a single topic and return 3 concise factual bullet points "
        "with no preamble. Topic will be given by the manager."
    ),
)

writer_worker = Agent(
    name="WriterWorker",
    instructions=(
        "You are a copywriter. Given bullet points of research, write a tight "
        "120-word summary with a headline. No preamble."
    ),
)

reviewer_worker = Agent(
    name="ReviewerWorker",
    instructions=(
        "You are an editor. Given a draft, return a 2-line critique: what's "
        "strong and what to fix. No preamble."
    ),
)

research_tool = make_agent_tool(
    research_worker, "research_topic",
    "Research a topic and return factual bullet points.")
writer_tool = make_agent_tool(
    writer_worker, "write_summary",
    "Turn research bullets into a short written summary.")
reviewer_tool = make_agent_tool(
    reviewer_worker, "review_draft",
    "Review a draft and return a short critique.")

manager_agent = Agent(
    name="ManagerAgent",
    instructions=(
        "You are a content manager. Given a topic, you MUST: "
        "1) call research_topic, 2) call write_summary with the research, "
        "3) call review_draft with the summary. Then output the final polished "
        "summary followed by the reviewer's note. Use the tools in order."
    ),
    tools=[research_tool, writer_tool, reviewer_tool],
)


# ---------------------------------------------------------------------------
# 3. AGENTS-AS-TOOLS (direct, explicit agent calls)
# ---------------------------------------------------------------------------
translator_agent = Agent(
    name="TranslatorAgent",
    instructions=(
        "Translate the given text to Spanish. Return ONLY the translation, "
        "no quotes, no explanation."
    ),
)

sentiment_agent = Agent(
    name="SentimentAgent",
    instructions=(
        "Classify the sentiment of the given text as Positive, Negative, or "
        "Neutral and give a one-line reason. Return ONLY that classification."
    ),
)

translate_tool = make_agent_tool(
    translator_agent, "translate_to_spanish",
    "Translate input text into Spanish.")
sentiment_tool = make_agent_tool(
    sentiment_agent, "analyze_sentiment",
    "Analyze the sentiment of input text.")

coordinator_agent = Agent(
    name="CoordinatorAgent",
    instructions=(
        "You coordinate language services. Given text: call translate_to_spanish "
        "and analyze_sentiment, then present BOTH results clearly labeled. "
        "Use both tools every time."
    ),
    tools=[translate_tool, sentiment_tool],
)


# Registry so the runner can resolve by name and swap models at runtime.
AGENT_REGISTRY = {
    "handoffs": triage_agent,
    "manager_workers": manager_agent,
    "agents_as_tools": coordinator_agent,
}
