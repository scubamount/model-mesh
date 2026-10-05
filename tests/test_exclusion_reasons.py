"""Exclusion reasons: which gate kept each model out of ranking.

Proposal P1 of the 2026-10-05 holistic pass. The p95 latency ceiling has been
an open question since 2026-10-01 ("we don't want to lose models because they
may be up tomorrow"), argued from single-model snapshots. A live snapshot that
day showed retain ranked at 2 of 11, and every excluded model failed the
SUCCESS floor first, not the ceiling. The question needs counts, not anecdotes.

Load-bearing properties, each pinned below:

  1. exclusion_reason() IS the eligibility predicate: eligible() is
     `exclusion_reason() is None`. A second copy of the gates would let the
     counters describe a predicate the router no longer runs.
  2. Each gate reports its own name, and the FIRST gate wins, so a model is
     counted once, against the gate that actually excluded it.
  3. route() records the reasons ranking produced for that request, and the
     app persists them per request (audit/exclusions.jsonl) so a week of
     evidence accumulates; /mesh/exclusions summarises it.
"""
import time

import pytest

from model_mesh.index import OK, Index
from model_mesh.router import Router, RouterConfig

HANG = (599, {"error": "timeout"})
GOOD = (200, {"choices": [{"message": {"content": '{"facts": ["x"]}'}}]})


@pytest.fixture()
def index(tmp_path):
    return Index(tmp_path / "mesh.db")


def _router(index, **cfg):
    return Router(index, "http://up/v1", "k", RouterConfig(**cfg),
                  transport=lambda u, b, h, t: GOOD)


def _samples(index, model, op, statuses_ms):
    index.ensure_model(model)
    for status, ms in statuses_ms:
        index.record(model, op, "request", status, ms)


# --- 1. one predicate --------------------------------------------------------

def test_eligible_is_exactly_exclusion_reason_is_none(index):
    """Over a spread of states, the two must never disagree."""
    r = _router(index)
    _samples(index, "fine", "retain", [(OK, 1000.0)] * 5)
    _samples(index, "failing", "retain", [("http-503", 1000.0)] * 5)
    _samples(index, "slow", "retain", [(OK, 200_000.0)] * 5)
    index.ensure_model("unknown")
    for m in ("fine", "failing", "slow", "unknown"):
        assert r.eligible(m, "retain") == (r.exclusion_reason(m, "retain") is None), m


def test_every_declared_reason_is_one_the_predicate_can_return():
    """EXCLUSION_REASONS is the published vocabulary, and it matches the
    predicate's return statements exactly. This is a SOURCE check: it proves
    each name is wired to a gate, not that the gate can fire. Reachability is
    a separate question -- see test_budget_floor_cannot_fire_at_default_config."""
    import inspect
    src = inspect.getsource(Router.exclusion_reason)
    for reason in Router.EXCLUSION_REASONS:
        assert f'return "{reason}"' in src, f"{reason} is declared but never returned"
    returned = set()
    for line in src.splitlines():
        line = line.strip()
        if line.startswith('return "'):
            returned.add(line.split('"')[1])
    assert returned == set(Router.EXCLUSION_REASONS), returned ^ set(Router.EXCLUSION_REASONS)


# --- 2. each gate names itself -----------------------------------------------

def test_success_floor_reports_success_floor(index):
    r = _router(index)
    _samples(index, "m", "retain", [("http-503", 1000.0)] * 5)
    assert r.exclusion_reason("m", "retain") == "success_floor"


def test_latency_floor_reports_latency_floor(index):
    """Reliable but slower than the op_class ceiling: the p95 ceiling, and
    the reason the P1 question is about. Must not be mislabelled."""
    r = _router(index)
    ceiling = r.cfg.latency_ceiling_ms("retain")
    _samples(index, "m", "retain", [(OK, ceiling + 5_000.0)] * 5)
    assert r.exclusion_reason("m", "retain") == "latency_floor"


def test_reject_reports_reject(index):
    r = _router(index)
    _samples(index, "m", "retain", [("http-400", 50.0)])
    assert r.exclusion_reason("m", "retain") == "reject"


def test_fidelity_cooldown_reports_fidelity_cooldown(index):
    r = _router(index)
    index.ensure_model("m")
    r._fidelity_fail("m", "retain")
    r._fidelity_fail("m", "retain")
    assert r.exclusion_reason("m", "retain") == "fidelity_cooldown"


def test_breaker_cooldown_reports_breaker_cooldown(index):
    r = _router(index)
    index.ensure_model("m")
    index.breaker_set("m", state="down", consec_fails=3,
                      cooldown_until=time.time() + 60.0, cooldown_s=60.0)
    assert r.exclusion_reason("m", "retain") == "breaker_cooldown"


def test_first_gate_wins_so_a_model_is_counted_once(index):
    """A model failing BOTH the reject gate and the success floor is counted
    against reject -- the earlier gate. Otherwise the tally double-counts and
    overstates the later gates, which is exactly the inflation that would make
    the latency ceiling look costlier than it is."""
    r = _router(index)
    _samples(index, "m", "retain",
             [("http-503", 1000.0)] * 4 + [("http-400", 50.0)])
    assert r.exclusion_reason("m", "retain") == "reject"


def test_an_unknown_model_is_eligible(index):
    r = _router(index)
    index.ensure_model("m")
    assert r.exclusion_reason("m", "retain") is None


# --- 3. route() records, ranked() agrees --------------------------------------

def test_ranked_with_reasons_partitions_the_pool(index):
    """Every candidate is either ranked or carries a reason -- never both,
    never neither."""
    r = _router(index)
    _samples(index, "good", "retain", [(OK, 1000.0)] * 5)
    _samples(index, "bad", "retain", [("http-503", 1000.0)] * 5)
    index.ensure_model("new")
    pool = ["good", "bad", "new"]
    ranked, excluded = r.ranked_with_reasons(pool, "retain")
    assert set(ranked) | set(excluded) == set(pool)
    assert not set(ranked) & set(excluded)
    assert excluded == {"bad": "success_floor"}
    assert ranked == r.ranked(pool, "retain")


def test_route_reports_what_ranking_excluded(index):
    r = _router(index)
    _samples(index, "good", "retain", [(OK, 1000.0)] * 5)
    _samples(index, "bad", "retain", [("http-503", 1000.0)] * 5)
    res = r.route(["good", "bad"], {"messages": []}, "retain")
    assert res.ok
    assert res.excluded == {"bad": "success_floor"}


# --- 4. the app persists and summarises ---------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    import model_mesh.app as A
    idx = Index(tmp_path / "http.db")
    idx.sync_catalog("nim", {"m-good", "m-bad"})
    _samples(idx, "m-good", "retain", [(OK, 1000.0)] * 5)
    _samples(idx, "m-bad", "retain", [("http-503", 1000.0)] * 5)
    router = Router(idx, "http://up/v1", "k", RouterConfig(),
                    transport=lambda u, b, h, t: GOOD)
    monkeypatch.setattr(A, "INDEX", idx)
    monkeypatch.setattr(A, "ROUTER", router)
    monkeypatch.setattr(A, "_state_dir", lambda: tmp_path)
    monkeypatch.setattr(A, "fetch_catalog", lambda *a, **k: {"data": []})
    from fastapi.testclient import TestClient
    with TestClient(A.app) as c:
        yield c, tmp_path


def test_each_routed_request_appends_one_exclusion_record(client):
    import json
    c, home = client
    for _ in range(2):
        r = c.post("/v1/chat/completions", json={
            "model": "auto/retain", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 200, r.text
    lines = (home / "audit" / "exclusions.jsonl").read_text().splitlines()
    assert len(lines) == 2, lines
    rec = json.loads(lines[-1])
    assert rec["alias"] == "auto/retain"
    assert rec["excluded_by"] == {"success_floor": 1}
    assert rec["ok"] is True
    assert rec["pool"] - rec["ranked"] == 1


def test_mesh_status_publishes_reasons(client):
    c, _ = client
    a = c.get("/mesh/status").json()["aliases"]["auto/retain"]
    assert a["excluded"] == {"m-bad": "success_floor"}
    assert a["excluded_by"] == {"success_floor": 1}


def test_mesh_exclusions_splits_by_request_outcome(client):
    """The number that decides a gate: on how many FAILED requests was it
    present? Write two synthetic records and read the summary back."""
    import json
    c, home = client
    path = home / "audit" / "exclusions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    rows = [
        {"ts": now, "alias": "auto/x", "excluded_by": {"latency_floor": 2}, "ok": False},
        {"ts": now, "alias": "auto/x", "excluded_by": {"latency_floor": 1,
                                                       "success_floor": 3}, "ok": True},
        {"ts": now - 10 * 86400, "alias": "auto/x",
         "excluded_by": {"latency_floor": 9}, "ok": False},   # outside window
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = c.get("/mesh/exclusions", params={"hours": 24}).json()["aliases"]["auto/x"]
    assert s["requests"] == 2 and s["failed"] == 1
    assert s["excluded_by"] == {"latency_floor": 3, "success_floor": 3}
    # Model-exclusions on failed requests (2 models) vs failed REQUESTS that
    # carried the gate (1 request) -- two units, kept as two fields.
    assert s["failed_excluded_by"] == {"latency_floor": 2}
    assert s["failed_requests_with"] == {"latency_floor": 1}


def test_mesh_exclusions_survives_malformed_lines(client):
    """One corrupt or foreign line must not 500 a week of evidence."""
    import json
    c, home = client
    path = home / "audit" / "exclusions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    good = {"ts": time.time(), "alias": "auto/x",
            "excluded_by": {"success_floor": 1}, "ok": True}
    path.write_text("not json\n[1, 2]\n{\"no_alias\": 1}\n"
                    + json.dumps(good) + "\n")
    r = c.get("/mesh/exclusions", params={"hours": 24})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["aliases"]["auto/x"]["requests"] == 1
    assert body["skipped_lines"] == 3


def test_audit_file_rolls_over_instead_of_growing_forever(client, monkeypatch):
    import model_mesh.app as A
    c, home = client
    monkeypatch.setattr(A, "_EXCLUSIONS_MAX_BYTES", 200)
    for _ in range(4):
        c.post("/v1/chat/completions", json={
            "model": "auto/retain", "messages": [{"role": "user", "content": "x"}]})
    path = home / "audit" / "exclusions.jsonl"
    assert path.with_suffix(".jsonl.1").is_file(), "never rolled over"
    assert path.stat().st_size <= 200 + 400, "current file exceeded the cap"


def test_an_audit_write_failure_never_fails_the_request(client, monkeypatch):
    import model_mesh.app as A
    c, _ = client

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(A, "_audit", boom)
    r = c.post("/v1/chat/completions", json={
        "model": "auto/retain", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200, r.text


def test_budget_floor_cannot_fire_at_default_config():
    """A FACT this pass found, pinned so it is not rediscovered: under the
    shipped defaults the budget floor is unreachable.

    A model only reaches it after the latency floor passed, so p95 <= ceiling
    <= request_timeout T. Then expected + T = sr*p95 + (1-sr)*T + T <= 2T, and
    the gate fires only when 2T > total_budget_s. Defaults: T = 135s on the
    memory lanes, 2T = 270s < 280s. (Independent review confirmed the algebra.)

    It is NOT dead code -- tests/test_router_expected_dial_cost.py reaches it
    with a 200s budget, and a future budget cut or timeout raise re-arms it.
    If this test starts failing, the gate has become live: that is a routing
    change worth noticing, not a test to delete.
    """
    from model_mesh.config import DEFAULTS
    c = RouterConfig(**DEFAULTS["router"])
    lanes = ["retain", "consolidation", "reflect", "evolve", None]
    assert all(2 * c.request_timeout_for(oc) <= c.total_budget_s for oc in lanes), (
        {oc: c.request_timeout_for(oc) for oc in lanes}, c.total_budget_s)


def test_mesh_exclusions_parses_off_the_event_loop(client, monkeypatch):
    """The audit can be tens of MB. Parsing it on the event loop stalls
    /health and every routed request behind it (independent review,
    2026-10-05). Pin that the summariser runs off the loop's thread.

    Comparing against threading.main_thread() is vacuous here: TestClient
    runs the app's event loop in its own portal thread, so an on-loop call is
    ALSO "not the main thread" and the assertion passed against the bug. The
    loop's thread is captured from inside a coroutine instead.
    """
    import threading
    import model_mesh.app as A
    c, _ = client
    seen = {}
    real_summarise = A._summarise_exclusions
    real_handler_to_thread = A.asyncio.to_thread

    def spy(hours):
        seen["work"] = threading.get_ident()
        return real_summarise(hours)

    async def to_thread_spy(fn, *args, **kwargs):
        seen["loop"] = threading.get_ident()      # we are ON the loop here
        return await real_handler_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(A, "_summarise_exclusions", spy)
    monkeypatch.setattr(A.asyncio, "to_thread", to_thread_spy)
    assert c.get("/mesh/exclusions").status_code == 200
    assert "work" in seen, "summariser never ran"
    assert "loop" in seen, (
        "handler never went through asyncio.to_thread: the summary is "
        "parsed on the event loop")
    assert seen["work"] != seen["loop"], "summary ran on the loop thread"
