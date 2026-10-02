"""Probe realism: a probe must stand in for the work it will be asked to do.

The module docstring in opclass.py records the original defect: a ~50-char
probe measured queue latency, not throughput. gpt-oss-20b did a toy prompt in
~2s but 27.9s on a real 14k-char retain chunk, which inverted the ranking and
starved hindsight retain into 300s timeouts.

The fix here is per-op_class STATIC sizes plus shape-aware filler. Two things
make these tests real rather than decorative:

  - every size is pinned to a MEASURED real-traffic percentile, and the test
    states which percentile, so a future edit that re-flattens the sizes to
    one number fails;
  - the shape assertions parse the payload. A probe that ships a broken or
    out-of-shape document measures a model's error recovery, not its latency,
    which is the same class of mistake as probing the wrong contract.
"""
import json

import pytest

from model_mesh.opclass import (
    PAD_CHARS,
    PROMPTS,
    _json_filler,
    _prose_filler,
    check_fidelity,
    probe_messages,
)

# Real non-probe traffic measured 2026-10-01 from the live DB. These are the
# numbers the static targets are derived from -- if live traffic shifts, these
# are the figures to re-measure, not the targets to silently move.
REAL_P50 = {"retain": 12564, "consolidation": 33171, "reflect": 31758,
            "evolve": 306}
LANES = ("retain", "consolidation", "reflect", "evolve")


def _user_body(op_class):
    msgs = probe_messages(op_class)
    idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    return msgs[idx]["content"]


# --- sizes -----------------------------------------------------------------

@pytest.mark.parametrize("op_class", LANES)
def test_probe_size_tracks_its_own_measured_real_traffic(op_class):
    """Each lane's probe must track the traffic it stands in for, pinned to a
    MEASURED percentile -- not to PAD_CHARS itself.

    The first version of this test compared the probe against PAD_CHARS, so it
    could not fail: shrinking PAD_CHARS shrank the expectation with it. That
    let a mutation putting consolidation and reflect back to 12000 pass green
    while the probes measured a third of the real work. The expectation has to
    come from OUTSIDE the code under test.

    Bounds are +/-30% of the measured real p50. A probe far under the real
    traffic measures queue latency (the nim-proxy defect); far over it
    overstates cost and burns the shared rate-limited probe budget.
    """
    real_p50 = REAL_P50[op_class]
    actual = len(_user_body(op_class))
    assert actual >= real_p50 * 0.7, (
        f"{op_class} probe is {actual} chars but real traffic p50 is "
        f"{real_p50} ({real_p50 / actual:.2f}x) -- this probe understates the "
        f"work it stands in for, which inverts ranking"
    )
    assert actual <= real_p50 * 1.3, (
        f"{op_class} probe is {actual} chars against a real p50 of {real_p50} "
        f"({actual / real_p50:.2f}x) -- overstates cost against a "
        f"rate-limited probe budget"
    )


def test_pad_targets_are_not_flat():
    """The regression guard for the original defect, pinned to measurement.

    Consolidation and reflect run 2-4x larger payloads than evolve, whose real
    traffic is ~300 chars. A flat PAD_CHARS is the nim-proxy defect again, and
    it was wrong in BOTH directions at once: understating two lanes by ~3x and
    overstating evolve by 357x.
    """
    assert len(set(PAD_CHARS.values())) > 1, (
        f"PAD_CHARS is flat at {set(PAD_CHARS.values())}; probes must be sized "
        f"per op_class to the traffic they stand in for"
    )
    for lane in ("consolidation", "reflect"):
        assert PAD_CHARS[lane] > PAD_CHARS["evolve"] * 10, (
            f"{lane} ({PAD_CHARS[lane]}) must be far larger than evolve "
            f"({PAD_CHARS['evolve']}): real traffic is "
            f"{REAL_P50[lane]} vs {REAL_P50['evolve']} chars"
        )


def test_evolve_probe_is_not_wastefully_large():
    """evolve's real payloads are tiny: the modal size is 306 chars and 616
    real samples average 141. A 12KB probe there spends 357x the budget to
    stand in for a request the model will never see."""
    actual = len(_user_body("evolve"))
    assert actual <= 2000, (
        f"evolve probe is {actual} chars against a real p50 of "
        f"{REAL_P50['evolve']}; it wastes the shared probe budget"
    )


def test_every_lane_has_a_pad_target():
    for op_class in PROMPTS:
        assert op_class in PAD_CHARS, f"{op_class} falls back to no padding"


# --- shape -----------------------------------------------------------------

def test_json_lane_filler_is_a_parseable_json_stream():
    """consolidation's real body is an envelope, so its padding continues in
    kind: space-separated objects that EACH parse on their own.

    An earlier version truncated the joined string at exactly the target
    length, cutting the last object mid-token. The probe then shipped a
    document that would not parse, and the model was being measured on error
    recovery rather than latency.
    """
    raw = _json_filler(4000)
    dec = json.JSONDecoder()
    idx, n = 0, 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] == " ":
            idx += 1
        if idx >= len(raw):
            break
        obj, idx = dec.raw_decode(raw, idx)   # raises if the tail is broken
        assert isinstance(obj, dict)
        n += 1
    assert n > 10, f"expected many objects, got {n}"


def test_json_filler_never_exceeds_its_budget():
    for budget in (200, 1000, 5000, 33000):
        assert len(_json_filler(budget)) <= budget, (
            f"_json_filler({budget}) produced {len(_json_filler(budget))}"
        )


def test_prose_filler_never_exceeds_its_budget():
    for budget in (200, 1000, 12000):
        assert len(_prose_filler(budget)) <= budget


def test_prose_filler_cycles_rather_than_repeating_one_sentence():
    """A payload that is the same 118 characters 100 times is a shape no real
    request has, and a model can key on the repetition instead of processing
    the content. At least two distinct sentences must appear.
    """
    from model_mesh.opclass import _PROSE_FILLER
    filler = _prose_filler(4000)
    present = [s for s in _PROSE_FILLER if s in filler]
    assert len(present) >= 2, (
        "filler is a single repeated sentence; that is not a real payload shape"
    )


def test_only_consolidation_gets_json_padding():
    """retain, reflect and evolve send PROSE, and their padding must be prose.

    Measured: 645 real retain samples average 97 chars of prose; the modal
    real evolve payload is 306 chars; reflect is prose by contract. Splicing a
    JSON document onto any of them creates a shape that never arrives -- and
    on evolve it was a literal parse error at the seam.

    The assertion has to look at the FILLER, not at the head of the body. An
    earlier version partitioned on the first newline and checked only the head,
    which is the prose base prompt in every lane -- so it stayed green when
    JSON padding was switched on for all four.
    """
    from model_mesh.opclass import PROMPTS as _P
    for op_class in ("retain", "reflect", "evolve"):
        base = _P[op_class]
        idx = max(i for i, m in enumerate(base) if m["role"] == "user")
        head = base[idx]["content"]
        body = _user_body(op_class)
        assert body.startswith(head), (
            f"{op_class} probe does not start with its own prompt"
        )
        filler = body[len(head):]
        assert filler, f"{op_class} has no filler to inspect"
        dec = json.JSONDecoder()
        # The filler must NOT be a JSON document. Strip the leading separator
        # so the check is about content, not whitespace.
        probe = filler.lstrip("\n ")
        with pytest.raises(ValueError):
            dec.raw_decode(probe)
        # And it must read as prose: real sentences, not punctuation soup.
        assert probe.count(". ") >= 2, (
            f"{op_class} filler is not prose: {probe[:120]!r}"
        )


def test_consolidation_body_stays_valid_across_the_seam():
    """The base envelope and the padded stream must not corrupt each other.

    The first padded object has to survive being read as part of the same
    document the envelope opened.
    """
    body = _user_body("consolidation")
    head, _, tail = body.partition("\n")
    envelope = json.loads(head)
    assert "facts" in envelope and "observations" in envelope, (
        f"consolidation head is not the real envelope: {list(envelope)}"
    )
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(tail):
        while idx < len(tail) and tail[idx] == " ":
            idx += 1
        if idx >= len(tail):
            break
        obj, idx = dec.raw_decode(tail, idx)
        assert "text" in obj, f"padded object is not a fact: {obj}"


# --- probe and request are one contract ------------------------------------

@pytest.mark.parametrize("op_class", LANES)
def test_padded_probe_still_passes_its_own_checker(op_class):
    """Padding must not make a probe unanswerable. A probe the checker
    rejects scores the model for a prompt the caller never sends -- the
    consolidation wipeout, where probes passed while real traffic failed."""
    msgs = probe_messages(op_class)
    if op_class == "reflect":
        reply = {"choices": [{"message": {"content": "A grounded prose answer."}}]}
    elif op_class == "consolidation":
        reply = {"choices": [{"message": {
            "content": '{"creates": [], "updates": [], "deletes": []}'}}]}
    else:
        reply = {"choices": [{"message": {
            "content": '{"facts": ["one", "two"]}'}}]}
    ok, why = check_fidelity(reply, op_class)
    assert ok, why
    # And the system prompt must survive padding untouched.
    assert msgs[0]["content"] == PROMPTS[op_class][0]["content"], (
        "padding altered the system prompt; probe and request must share it"
    )


def test_probe_messages_does_not_mutate_the_prompt_table():
    """probe_messages() is called on every discovery pass and every re-probe.
    If it edited PROMPTS in place, the second call would double-pad."""
    first = _user_body("consolidation")
    second = _user_body("consolidation")
    assert len(first) == len(second), (
        f"probe_messages mutated PROMPTS: {len(first)} then {len(second)}"
    )


def test_padding_never_shrinks_a_body():
    for op_class in LANES:
        base = PROMPTS[op_class]
        idx = max(i for i, m in enumerate(base) if m["role"] == "user")
        assert len(_user_body(op_class)) >= len(base[idx]["content"]), (
            f"{op_class} probe is smaller than its own prompt"
        )
