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
    {"role": "user", "content": "Scubamount dives in Monterey and prefers a 7mm wetsuit."},
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
            {"facts": [{"text": "Scubamount logged a 30m dive.",
                        "context": "conversation between agent and user"}],
             "observations": []}
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
            "Task: the daemon on port 8002 just restarted — what do you do?"
        ),
    },
]

PROMPTS = {"retain": RETAIN_MESSAGES, "consolidation": CONSOLIDATION_MESSAGES,
           "evolve": EVOLVE_MESSAGES}

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
    """
    content = parse_content(response)
    if not content or not content.strip():
        return False, "empty content (reasoning-only or no output)"
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
