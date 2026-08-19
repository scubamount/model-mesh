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
            "Consolidate the observation. Respond with ONLY a JSON object of the form "
            '{"observation_id": "<echo the id>", "facts": ["fact1"]}. '
            "No prose, no markdown."
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {"observation_id": "obs-4271", "text": "Scubamount logged a 30m dive."}
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
    """
    content = parse_content(response)
    if not content or not content.strip():
        return False, "empty content (reasoning-only or no output)"
    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, ValueError):
        return False, "content is not valid JSON (markdown/prose leak)"

    if isinstance(parsed, list):
        facts = parsed
    elif isinstance(parsed, dict):
        facts = parsed.get("facts")
    else:
        return False, "JSON is neither object nor array"
    if not isinstance(facts, list) or not facts:
        return False, "missing/empty `facts` list"
    if op_class == "consolidation":
        if not isinstance(parsed, dict) or not parsed.get("observation_id"):
            return False, "consolidation: missing echoed `observation_id`"
    return True, "ok"
