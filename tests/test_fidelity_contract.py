"""check_fidelity — the contract test that did not exist.

Written 2026-08-24, after a contract mismatch inside this function silently
took auto/consolidation to zero ranked candidates for a full day. The suite
was 131 green through the entire incident and stayed 131 green when the
consolidation contract was INVERTED, because nothing here tested this
function at all. A checker with no test is a checker that can only be
validated by production.

The load-bearing case is `test_consolidation_empty_envelope_is_success`:
hindsight's real reply for "nothing to merge" is
`{"creates": [], "updates": [], "deletes": []}`, and scoring that as a
failure is exactly what floored all 26 candidates.
"""
import json

import pytest

from model_mesh.opclass import (
    CONSOLIDATION_MESSAGES,
    RETAIN_MESSAGES,
    check_fidelity,
)


def resp(content):
    return {"choices": [{"message": {"content": content}}]}


# -- consolidation: the creates/updates/deletes envelope --------------------

def test_consolidation_empty_envelope_is_success():
    """"Nothing to merge" is a real answer, not a fidelity failure.

    This is the exact body that produced 1,544 fidelity-fails and emptied the
    ranking. If this test ever goes red, auto/consolidation is losing its pool
    again.
    """
    ok, why = check_fidelity(
        resp('{"creates": [], "updates": [], "deletes": []}'), "consolidation")
    assert ok, why


def test_consolidation_populated_envelope_is_success():
    body = json.dumps({
        "creates": [{"text": "The user runs model-mesh locally.",
                     "source_fact_ids": ["7eb6ca3c"]}],
        "updates": [], "deletes": [],
    })
    ok, why = check_fidelity(resp(body), "consolidation")
    assert ok, why


def test_consolidation_partial_envelope_is_success():
    """Real models omit keys they have nothing for; that is not a violation."""
    ok, why = check_fidelity(resp('{"creates": []}'), "consolidation")
    assert ok, why


def test_consolidation_rejects_missing_envelope():
    ok, why = check_fidelity(resp('{"facts": ["x"]}'), "consolidation")
    assert not ok
    assert "envelope" in why


def test_consolidation_rejects_non_list_member():
    ok, why = check_fidelity(
        resp('{"creates": "not-a-list", "updates": []}'), "consolidation")
    assert not ok
    assert "not a list" in why


def test_consolidation_rejects_prose():
    ok, why = check_fidelity(resp("Sure! Here are the observations:"),
                             "consolidation")
    assert not ok
    assert "not valid JSON" in why


def test_consolidation_rejects_empty_content():
    """Reasoning-only replies (content empty, output in reasoning_content)."""
    ok, why = check_fidelity(resp(""), "consolidation")
    assert not ok
    assert "empty content" in why


def test_consolidation_rejects_json_array():
    ok, why = check_fidelity(resp('[{"text": "x"}]'), "consolidation")
    assert not ok
    assert "not an object" in why


def test_consolidation_accepts_fenced_envelope():
    ok, why = check_fidelity(
        resp('```json\n{"creates": [], "updates": [], "deletes": []}\n```'),
        "consolidation")
    assert ok, why


# -- the probe must satisfy the same contract it scores --------------------

def test_probe_prompt_requests_the_real_contract():
    """The probe asked for a shape no caller sends, so it passed while every
    real request failed. Probe and request must be one contract."""
    sys_msg = CONSOLIDATION_MESSAGES[0]["content"]
    assert "creates" in sys_msg and "updates" in sys_msg and "deletes" in sys_msg
    assert "observation_id" not in sys_msg


def test_probe_prompt_example_passes_its_own_checker():
    """Whatever shape the probe asks for must be a shape check_fidelity
    accepts. If these two ever drift apart again, this goes red."""
    ok, why = check_fidelity(
        resp('{"creates": [], "updates": [], "deletes": []}'), "consolidation")
    assert ok, why


# -- retain/reflect keep the facts contract --------------------------------

@pytest.mark.parametrize("op_class", ["retain", "evolve"])
def test_facts_contract_still_applies_to_other_op_classes(op_class):
    ok, why = check_fidelity(resp('{"facts": ["a"]}'), op_class)
    assert ok, why
    ok, why = check_fidelity(resp('{"facts": []}'), op_class)
    assert not ok
    assert "facts" in why


def test_retain_accepts_bare_array():
    ok, why = check_fidelity(resp('["a", "b"]'), "retain")
    assert ok, why


def test_retain_rejects_consolidation_envelope():
    """The envelope is consolidation-specific; retain still wants facts."""
    ok, why = check_fidelity(resp('{"creates": [], "updates": []}'), "retain")
    assert not ok


# -- reflect is PROSE, and must not be judged by a JSON contract -----------
#
# Found live 2026-09-01. auto/reflect was declared `op_class: "retain"`, so a
# correct prose synthesis was scored "content is not valid JSON (markdown/prose
# leak)". One real request tried 16 models, 12 fidelity-fail, and 503'd after
# 280s — while the same models answered the same prompt in 0.3s when dialed
# directly. Every hindsight mental-model refresh routes through this alias.

def test_reflect_accepts_prose():
    """The load-bearing case. If this goes red, reflect loses its pool again."""
    ok, why = check_fidelity(
        resp("The daemon listens on 9177 and pins its embedder to the GPU."),
        "reflect")
    assert ok, why


def test_reflect_accepts_markdown():
    ok, why = check_fidelity(
        resp("## Memory stack\n\n- daemon: `:9177`\n- embedder: GPU\n"),
        "reflect")
    assert ok, why


def test_reflect_still_rejects_empty_content():
    """Reasoning-only replies (output in reasoning_content) remain a failure —
    'said nothing' is the only thing a reflect probe can actually detect."""
    ok, why = check_fidelity(resp(""), "reflect")
    assert not ok
    assert "empty content" in why


def test_reflect_alias_has_its_own_op_class():
    """A prose alias declared `retain` votes in hindsight's retain ranking with
    verdicts from a contract retain does not use, and inherits a gate its own
    caller never sends. Both directions are wrong."""
    from model_mesh.config import DEFAULTS
    assert DEFAULTS["aliases"]["auto/reflect"]["op_class"] == "reflect"


def test_reflect_probe_prompt_exists_and_passes_its_own_checker():
    """Same rule as consolidation: probe and request are one contract. Without
    its own entry, `probe_messages` silently fell back to the RETAIN prompt,
    so reflect models were probed on a JSON task they are never asked to do."""
    from model_mesh.opclass import PROMPTS, probe_messages
    assert "reflect" in PROMPTS, "reflect falls back to the retain probe"
    msgs = probe_messages("reflect")
    assert msgs[0] != RETAIN_MESSAGES[0]
    sys_msg = msgs[0]["content"]
    assert "JSON" not in sys_msg or "No JSON required" in sys_msg
    # Whatever the probe asks for must be something check_fidelity accepts.
    ok, why = check_fidelity(resp("A grounded prose answer."), "reflect")
    assert ok, why


# -- a tool call is an ANSWER, not silence ---------------------------------
#
# Second half of the same 2026-09-01 incident, and the more general bug: an
# agentic caller that supplies tools gets `content: null` + `tool_calls: [...]`
# back on every hop but the last. That was scored "empty content
# (reasoning-only or no output)" — 21 fidelity-fails across 9 models, each one
# a model correctly calling the recall tool it was handed. Two consecutive
# verdicts floor a model out of the op_class, so the entire pool was demoted
# for behaving correctly, and mental-model refreshes could never finish.

def _tool_call_resp(content=None):
    return {"choices": [{"message": {
        "content": content,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "recall",
                                     "arguments": '{"query": "projects"}'}}],
    }}]}


@pytest.mark.parametrize("op_class", ["reflect", "retain", "consolidation",
                                      "evolve"])
def test_tool_call_with_null_content_is_success(op_class):
    """Applies to every op_class: the caller decides whether tools were on the
    table, and a model that used them did its job."""
    ok, why = check_fidelity(_tool_call_resp(None), op_class)
    assert ok, why


def test_empty_string_content_with_tool_calls_is_success():
    """Providers differ on null vs "" for the same event."""
    ok, why = check_fidelity(_tool_call_resp(""), "reflect")
    assert ok, why


def test_no_content_and_no_tool_calls_is_still_a_failure():
    """The discriminating case — without it the fix would accept true
    silence, which is the reasoning-only failure this gate exists to catch."""
    ok, why = check_fidelity(resp(""), "reflect")
    assert not ok
    assert "empty content" in why


def test_empty_tool_calls_list_is_not_a_tool_call():
    """`tool_calls: []` is the provider saying "I called nothing"."""
    r = {"choices": [{"message": {"content": None, "tool_calls": []}}]}
    ok, why = check_fidelity(r, "reflect")
    assert not ok
    assert "empty content" in why
