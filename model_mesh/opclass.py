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

# Pad probes toward the op-class's REAL payload size.
#
# These are STATIC numbers, deliberately not derived from live traffic at
# runtime. Deriving them would close a feedback loop: traffic shapes the probe,
# the probe decides which models the traffic goes to, and the routing then
# changes what the traffic looks like. A static target keeps the probe an
# independent yardstick.
#
# Measured 2026-10-01 over all non-probe samples in the live DB:
#
#   op_class       n      real p50    real p90   real p95   probe was
#   retain      7311       12564       15432      15494     12152 (0.97x)
#   consolidation 13366    33171       46950      68874    12152 (0.37x)
#   reflect     22826       31758      149542     199130    12152 (0.38x)
#   evolve        726          34        5060       5060   12152 (357x!)
#
# The old flat 12000 was wrong in BOTH directions. A flat pad UNDER-stated
# consolidation and reflect by ~3x, which is the nim-proxy toy-probe defect
# again: those lanes measure queue latency, not the work they will be handed.
# It also OVER-stated evolve by 357x, spending a 12KB payload to probe a lane
# whose real requests are ~34 characters -- pure waste, and it made evolve
# probes look slower than the work they stand in for.
#
# Targets sit at the real p50, the size a request actually has half the time.
# Sizing to p90 instead would make every probe cost 4x a typical request, and
# the probe budget is shared with the 429-sensitive key.
PAD_CHARS = {"retain": 13000, "consolidation": 33000, "reflect": 32000,
             "evolve": 400}

# Op classes whose padded payload is a JSON STREAM (space-separated objects).
#
# Measured 2026-10-01, this is consolidation and NOTHING else. Its real body is
# a `{"facts": [...], "observations": []}` envelope, so the padding continues
# in kind. The other lanes were checked against their real payloads and are
# NOT JSON:
#   - retain  real chunks are prose sentences the model reduces to a facts
#     array; 645 real retain samples average 97 chars. A JSON document there
#     measures a shape that never arrives.
#   - evolve  the modal real payload is 306 chars of prose (238 of 726 samples);
#     616 real evolve samples average 141 chars. JSON padding overstates it
#     and, worse, splices a document onto a prose body -- which is why this
#     list has two members and not four.
#   - reflect prose by contract.
_JSON_OPS = {"consolidation"}

_PROSE_FILLER = (
    "The operator debugged the memory daemon; the embedder runs on the GPU and "
    "the proxy routes the retain alias to whichever model currently wins. ",
    "A prior consolidation pass merged two observations about the same dive "
    "site, so the newer record supersedes the older one. ",
    "The nightly discovery job syncs the catalog and marks anything absent as "
    "end-of-life, which is why the retired pool only ever shrinks. ",
)

_JSON_FACT = {
    "text": "The operator noted a preference worth carrying forward.",
    "context": "conversation between agent and user",
}


def _prose_filler(chars: int) -> str:
    """Enough filler prose to reach `chars`, cycling the source sentences.

    Cycling rather than repeating ONE sentence: a payload that is the same 118
    characters 100 times is a shape no real request has, and a model can key
    on the repetition instead of processing the content.
    """
    out: list[str] = []
    size = 0
    i = 0
    while True:
        sentence = _PROSE_FILLER[i % len(_PROSE_FILLER)]
        if size + len(sentence) > chars:
            break
        out.append(sentence)
        size += len(sentence)
        i += 1
    return "".join(out)


def _json_filler(chars: int) -> str:
    """A JSON fragment sequence of at most `chars`, shaped like real traffic.

    Emits a sequence of complete JSON OBJECTS, space-separated, so the payload
    is a valid JSON stream and every element parses on its own. The final
    object is dropped rather than sliced: an earlier version truncated the
    joined string at exactly `chars`, which cut the last object mid-token and
    produced a body that would not parse -- a probe that ships a broken
    document measures the model's error recovery, not its latency.

    The objects continue consolidation's own `{"text", "context"}` fact shape,
    so the stream is homogeneous with the envelope it follows.
    """
    out: list[str] = []
    size = 0
    i = 0
    while True:
        fact = dict(_JSON_FACT)
        fact["text"] = f"{_JSON_FACT['text']} (observation {i})"
        chunk = json.dumps(fact, separators=(",", ":"))
        need = len(chunk) + (1 if out else 0)
        if size + need > chars:
            break
        out.append(chunk)
        size += need
        i += 1
    return " ".join(out)


def probe_messages(op_class: str) -> list[dict]:
    """Representative probe for an op_class: op-specific contract + realistic size."""
    base = PROMPTS.get(op_class, PROMPTS["retain"])
    messages = [dict(m) for m in base]
    pad = PAD_CHARS.get(op_class, 0)
    if pad:
        user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        body = messages[user_idx]["content"]
        if len(body) < pad:
            room = pad - len(body)
            # Reflect is the one lane that genuinely sends prose, so prose
            # filler is right for it. The JSON lanes get JSON-shaped filler --
            # appending prose after a JSON document would be a shape no real
            # request has.
            filler = (_json_filler(room) if op_class in _JSON_OPS
                      else _prose_filler(room))
            sep = "\n" if op_class in _JSON_OPS else "\n\n"
            messages[user_idx]["content"] = body + sep + filler
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
