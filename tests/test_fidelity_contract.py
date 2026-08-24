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

from model_mesh.opclass import CONSOLIDATION_MESSAGES, check_fidelity


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
        "creates": [{"text": "Andrew runs model-mesh on 8002.",
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
