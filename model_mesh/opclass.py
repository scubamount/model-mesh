"""model-mesh: op-class probe prompts + fidelity gates.

Inherited from nim-proxy's auto_ranker (the load-bearing logic that survives
the rewrite). Probes must be REPRESENTATIVE: a ~50-char probe measures queue
latency, not throughput — measured 2026-08-03, gpt-oss-20b did a toy prompt in
~2s but 27.9s on a real 14k-char retain chunk, which inverted the ranking and
starved hindsight retain into 300s timeouts.
"""

from __future__ import annotations

import json
from typing import Optional

RETAIN_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Extract atomic facts from the user text. Respond with ONLY a JSON "
            'object of the form {"facts": ["fact1", "fact2"]}. No prose, no markdown.'
        ),
    },
    {"role": "user", "content": "The user dives in Monterey Bay and prefers a 7mm wetsuit."},
]

CONSOLIDATION_MESSAGES = [
    {
        "role": "system",
        "content": (
            # This MUST stay the shape hindsight really asks for. It used to
            # request {"observation_id","facts"}, which no consolidation caller
            # ever sends, so probes measured compliance with a contract that
            # existed nowhere else and passed while every real request failed
            # (see check_fidelity). Probe and request are one contract.
            "You are a memory consolidation system. Synthesize the new facts "
            "into observations, merging with existing observations when "
            "appropriate. Respond with ONLY a JSON object of the form "
            '{"creates": [], "updates": [], "deletes": []}. '
            "An empty envelope is valid when there is nothing to merge. "
            "No prose, no markdown."
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {"facts": [{"text": "The user logged a 30m dive.",
                        "context": "conversation between agent and user"}],
             "observations": []}
        ),
    },
]

REFLECT_MESSAGES = [
    {
        "role": "system",
        "content": (
            # Reflect is PROSE, not JSON. Hindsight's reflect agent asks for a
            # grounded natural-language synthesis over recalled facts (and
            # tool-calls its way there); it never asks for a facts envelope.
            # Scoring it against the retain contract rejected every correct
            # answer as "content is not valid JSON (markdown/prose leak)" —
            # see check_fidelity.
            "You answer questions from the supplied memories. Reply in plain "
            "prose, grounded in what you were given. No JSON required."
        ),
    },
    {
        "role": "user",
        "content": (
            "Memories: the operator runs a local memory daemon on port 9177; "
            "its embedder is pinned to the GPU.\n"
            "Question: summarize how the operator's memory stack is wired."
        ),
    },
]

EVOLVE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You follow the provided skill instructions to complete a task. "
            "Respond with ONLY a JSON object of the form "
            '{"facts": ["step1", "step2"]} listing the steps you would take. '
            "No prose, no markdown."
        ),
    },
    {
        "role": "user",
        "content": (
            "Skill: verify a daemon is healthy before routing traffic to it.\n"
            "Task: the daemon just restarted — what do you do?"
        ),
    },
]

PROMPTS = {"retain": RETAIN_MESSAGES, "consolidation": CONSOLIDATION_MESSAGES,
           "reflect": REFLECT_MESSAGES, "evolve": EVOLVE_MESSAGES}

# Pad probes to the op-class's real payload size.
PAD_CHARS = {"retain": 12000, "consolidation": 12000, "reflect": 12000,
             "evolve": 12000}

_FILLER = (
    "The operator debugged the memory daemon; the embedder runs on the GPU and "
    "the proxy routes the retain alias to whichever model currently wins. "
)


def probe_messages(op_class: str) -> list[dict]:
    """Representative probe for an op_class: op-specific contract + realistic size."""
    base = PROMPTS.get(op_class, PROMPTS["retain"])
    messages = [dict(m) for m in base]
    pad = PAD_CHARS.get(op_class, 0)
    if pad:
        user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        body = messages[user_idx]["content"]
        if len(body) < pad:
            reps = (pad - len(body)) // len(_FILLER) + 1
            messages[user_idx]["content"] = body + "\n\n" + (_FILLER * reps)[:pad]
    return messages


def parse_content(response: dict) -> Optional[str]:
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return msg.get("content")


def _tool_calls(response: dict) -> list:
    """The assistant's tool calls, if any.

    A reply that hands back `content: null` plus a populated `tool_calls` list
    is a CORRECT response to a request that supplied tools — it is the model
    doing the thing it was asked to do. Only a reply with neither is degenerate.
    Hindsight's reflect agent is native tool-calling, so every one of its first
    hops looks exactly like this.
    """
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return []
    calls = msg.get("tool_calls")
    return calls if isinstance(calls, list) else []


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def check_fidelity(response: dict, op_class: str) -> tuple[bool, str]:
    """A model may serve an op_class only if it obeys the structured-output
    contract: non-empty content, valid JSON, expected shape. Reasoning models
    that leave `content` empty (output in reasoning_content) fail on purpose.

    The consolidation contract is the CREATES/UPDATES/DELETES envelope that
    hindsight actually sends, not the {"observation_id","facts"} shape this
    function used to demand. That mismatch was the 2026-08-24 wipeout: real
    consolidation traffic returned a perfectly valid
    `{"creates": [], "updates": [], "deletes": []}` and was scored
    "missing/empty `facts` list" every time — 1,544 fidelity-fails, enough to
    push all 26 candidates under min_success_rate so `ranked()` returned []
    and /health read "no healthy candidates".

    It stayed invisible because the PROBE asked for the wrong shape too: the
    synthetic prompt requested {"observation_id","facts"}, models complied,
    and probes passed 786/899 while real traffic failed 795/830. A checker
    validated against its own probe rather than the caller's contract agrees
    with itself forever. The probe prompt in CONSOLIDATION_MESSAGES is now the
    real envelope, so probe and request are measured against one contract.

    An EMPTY envelope is a valid, successful consolidation: "nothing to merge"
    is a real answer. Only a missing/malformed envelope is a fidelity failure.

    REFLECT IS PROSE. Hindsight's reflect agent asks for a grounded
    natural-language synthesis; it sends no JSON schema and receives none.
    Judging it by the retain `facts` contract rejected every CORRECT answer as
    "content is not valid JSON (markdown/prose leak)" — the 2026-09-01 finding:
    a live `auto/reflect` request tried 16 models, 12 fidelity-fail, and 503'd
    after 280s, while `gpt-oss-120b` answered the same alias's underlying model
    directly in 0.3s. The alias was ALSO mis-declared `op_class: "retain"` in
    config, so its samples polluted the retain ranking with prose verdicts and
    it never had a lane of its own. Its contract here is what the caller
    actually needs: non-empty, non-degenerate text.
    """
    content = parse_content(response)
    if not content or not content.strip():
        # A tool call IS the answer when tools were supplied. Hindsight's
        # reflect agent is native tool-calling: its first hops return
        # `content: null` + `tool_calls: [...]`, which this function scored
        # "empty content (reasoning-only or no output)" — 21 fidelity-fails
        # across 9 models on 2026-09-01, every one of them a model correctly
        # calling the recall tool it was handed. Two consecutive such verdicts
        # floor a model out of the op_class, so the whole pool was demoted for
        # doing the right thing. Distinguish "said nothing" from "acted".
        if _tool_calls(response):
            return True, "ok"
        return False, "empty content (reasoning-only or no output)"

    if op_class == "reflect":
        # Deliberately minimal. The only reflect failure a probe can see is
        # "said nothing" — which the empty-content check above already caught,
        # including the reasoning-model split (content empty, output in
        # reasoning_content). Anything stricter re-imports a format opinion the
        # caller does not hold, and that is the bug this branch exists to fix.
        return True, "ok"

    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, ValueError):
        return False, "content is not valid JSON (markdown/prose leak)"

    if op_class == "consolidation":
        if not isinstance(parsed, dict):
            return False, "consolidation: JSON is not an object"
        keys = ("creates", "updates", "deletes")
        present = [k for k in keys if k in parsed]
        if not present:
            return False, ("consolidation: missing creates/updates/deletes "
                           "envelope")
        bad = [k for k in present if not isinstance(parsed[k], list)]
        if bad:
            return False, f"consolidation: {', '.join(bad)} is not a list"
        return True, "ok"

    if isinstance(parsed, list):
        facts = parsed
    elif isinstance(parsed, dict):
        facts = parsed.get("facts")
    else:
        return False, "JSON is neither object nor array"
    if not isinstance(facts, list) or not facts:
        return False, "missing/empty `facts` list"
    return True, "ok"
