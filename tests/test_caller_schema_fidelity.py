"""The caller's declared response schema is the fidelity contract.

Written 2026-10-27 after Hindsight 1.1.0 started sending a second shape down
auto/consolidation: a 1-by-1 dedup verdict `{"action","text","reason"}`. The
op_class heuristic only knew the creates/updates/deletes envelope, so every
correct dedup answer was scored fidelity-fail (307 of 321 consolidation fails
in 24h) and 11 of 13 pool models fell under the success floor. Retain had the
twin defect: `{"facts": []}` ("nothing worth remembering") is valid under
FactExtractionResponse but the heuristic demanded a non-empty list.

Schemas below are the exact provider_json_schema() output of hindsight_api
1.1.0 (_DedupDecision, FactExtractionResponse); responses are verbatim shapes
captured in mesh.log.
"""
import json

from model_mesh.index import Index
from model_mesh.opclass import caller_schema, check_fidelity
from model_mesh.router import Router, RouterConfig

DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"default": "keep", "enum": ["merge", "keep"], "type": "string"},
        "text": {"default": "", "type": "string"},
        "reason": {"default": "", "type": "string"},
    },
    "required": ["action", "text", "reason"],
    "additionalProperties": False,
}
FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "object"}}},
    "required": ["facts"],
}


def body(schema, name="resp"):
    return {"messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "schema": schema,
                                                "strict": False}}}


def resp(content):
    return {"choices": [{"message": {"content": content}}]}


# -- the live defects ---------------------------------------------------------

def test_dedup_merge_verdict_passes_on_consolidation():
    content = '{"action": "merge", "text": "The owner prefers X.", "reason": "same fact"}'
    assert check_fidelity(resp(content), "consolidation", body(DEDUP_SCHEMA)) == (True, "ok")


def test_dedup_keep_verdict_passes_on_consolidation():
    content = '{"action":"keep","text":"","reason":"distinct facet"}'
    assert check_fidelity(resp(content), "consolidation", body(DEDUP_SCHEMA)) == (True, "ok")


def test_retain_empty_facts_passes_when_schema_allows_it():
    assert check_fidelity(resp('{ "facts": [] }'), "retain", body(FACTS_SCHEMA)) == (True, "ok")


# -- the schema still rejects real failures ----------------------------------

def test_schema_rejects_missing_required_key():
    ok, why = check_fidelity(resp('{"action": "merge"}'), "consolidation", body(DEDUP_SCHEMA))
    assert not ok and "missing required" in why


def test_schema_rejects_enum_violation():
    c = '{"action": "maybe", "text": "", "reason": ""}'
    ok, why = check_fidelity(resp(c), "consolidation", body(DEDUP_SCHEMA))
    assert not ok and "enum" in why


def test_schema_rejects_wrong_property_type():
    ok, why = check_fidelity(resp('{"facts": "none"}'), "retain", body(FACTS_SCHEMA))
    assert not ok and "facts" in why


def test_schema_rejects_wrong_top_level_type():
    ok, why = check_fidelity(resp('["a"]'), "retain", body(FACTS_SCHEMA))
    assert not ok and "top-level" in why


def test_schema_path_still_rejects_prose_and_empty():
    assert not check_fidelity(resp("Sure! Here you go"), "consolidation", body(DEDUP_SCHEMA))[0]
    assert not check_fidelity(resp(""), "consolidation", body(DEDUP_SCHEMA))[0]


def test_bool_is_not_a_json_number():
    s = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert not check_fidelity(resp('{"n": true}'), "retain", body(s))[0]
    assert check_fidelity(resp('{"n": 3}'), "retain", body(s))[0]


# -- no schema: the op_class heuristic is unchanged ---------------------------

def test_no_schema_keeps_envelope_heuristic():
    ok, why = check_fidelity(resp('{"action": "merge", "text": "", "reason": ""}'), "consolidation")
    assert not ok and "envelope" in why


def test_no_schema_keeps_nonempty_facts_rule():
    assert not check_fidelity(resp('{"facts": []}'), "retain")[0]


def test_json_object_mode_is_not_a_schema():
    b = {"response_format": {"type": "json_object"}}
    assert caller_schema(b) is None
    assert not check_fidelity(resp('{"facts": []}'), "retain", b)[0]


def test_malformed_response_format_is_ignored():
    for rf in (None, "json", {"type": "json_schema"}, {"type": "json_schema", "json_schema": "x"}):
        assert caller_schema({"response_format": rf}) is None


# -- wiring: the router hands the request body to the checker ----------------

def test_router_records_dedup_answer_as_ok(tmp_path):
    index = Index(tmp_path / "mesh.db")
    dedup = json.dumps({"action": "merge", "text": "t", "reason": "r"})

    def transport(url, b, headers, timeout):
        return 200, {"choices": [{"message": {"content": dedup}}], "model": b["model"]}

    r = Router(index, "http://up.example/v1", "k", RouterConfig(), transport=transport)
    index.sync_catalog("nim", {"m/a"})
    payload, att = r.dial("m/a", body(DEDUP_SCHEMA), "consolidation", source="request")
    assert payload is not None
    assert att.status == "ok", att
