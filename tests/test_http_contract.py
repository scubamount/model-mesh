"""App-level HTTP contract tests (TestClient).

The API surface was documented in the README and asserted by no test — the
router/index layers below were covered, but nothing pinned what a client
actually sees: status codes, error-body shapes, x-mesh headers. These tests
exercise the app through FastAPI's TestClient with a stubbed transport, so they
run offline and fast.
"""

from __future__ import annotations

import pytest

from model_mesh.app import app
from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig

OK_BODY = {
    "choices": [{"message": {"content": '{"facts": ["x"]}'}}],
    "model": "m1",
}

PROSE_BODY = {
    "choices": [{"message": {"content": "Sure! Here is a thought about that."}}],
    "model": "bad1",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """The app with its INDEX/ROUTER replaced by offline fakes."""
    import model_mesh.app as A

    idx = Index(tmp_path / "http.db")
    idx.sync_catalog("nim", {"m1"})
    idx.ensure_model("m1")

    def transport(url, body, headers, timeout):
        if body.get("model") == "m1":
            return 200, OK_BODY
        if body.get("model") == "bad1":
            return 200, PROSE_BODY
        return 404, {"error": "not found"}

    router = Router(idx, "http://up/v1", "k", RouterConfig(), transport=transport)
    monkeypatch.setattr(A, "INDEX", idx)
    monkeypatch.setattr(A, "ROUTER", router)
    # /health must not depend on the real network in tests.
    monkeypatch.setattr(A, "fetch_catalog", lambda *a, **k: {"data": []})
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c, idx


def test_chat_success_carries_mesh_headers(client):
    c, _ = client
    r = c.post("/v1/chat/completions", json={
        "model": "auto/retain",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.headers["x-mesh-routed-model"] == "m1"
    assert int(r.headers["x-mesh-attempts"]) >= 1


def test_unknown_alias_is_honest_503_with_evidence(client):
    """A bogus alias 503s upstream; the response body must carry per-attempt
    evidence, never an empty error. And it must leave NO phantom state."""
    c, idx = client
    r = c.post("/v1/chat/completions", json={
        "model": "auto/nonexistent", "messages": [],
    })
    assert r.status_code == 502 or r.status_code == 503
    assert idx.breaker_all() == {}, "phantom breaker for invented id"
    with idx._lock:
        n = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert n == 0


def test_health_deep_503_when_alias_cannot_serve(client):
    """Deep health: every candidate ineligible AND no recent ok anywhere in
    the pool => the mesh actually cannot serve => 503. This is THE property
    that makes /health able to fail — the predecessor's could not."""
    c, idx = client
    # Make m1 measurably failing: below success floor at floor-sample count.
    for _ in range(4):
        idx.record("m1", "retain", "request", "http-500", 1000.0)
    r = c.get("/health")
    assert r.status_code == 503
    assert any("cannot serve" in p for p in r.json()["problems"].values())


def test_health_idle_alias_is_degraded_not_503(client):
    """No traffic at all in the serve window ≠ cannot serve. A low-traffic
    alias (evolve fires hours apart) has no in-window samples most of the
    day; absence of traffic is not evidence of failure. Must be degraded
    (200), never 503 — observed 2026-08-30: auto/evolve 503'd /health at
    16:30 off samples whose newest was 00:00 (idle 16h, not failing 16h)."""
    import time as _t

    c, idx = client
    # Old evidence only: failing samples well outside the serve window.
    with idx._lock:
        for i in range(4):
            idx._conn.execute(
                "INSERT INTO samples (model_id, ts, op_class, source,"
                " latency_ms, status, payload_chars) VALUES (?,?,?,?,?,?,?)",
                ("m1", _t.time() - 20000 + i, "retain", "request",
                 1000.0, "http-500", 10),
            )
        idx._conn.commit()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert any("idle" in v for v in body["degraded"].values())


def test_health_degraded_when_sweep_backstop_serving(client):
    """Floors admit nobody, but the pool produced a recent ok (sweep arm):
    the mesh IS serving, so /health must be 200/degraded, not 503. Whole-pool
    overload is NIM's steady state — observed 2026-08-30: /health 503 on all
    aliases while contract-valid retains returned 200 in the same minute."""
    c, idx = client
    # Recent ok first, then enough failures to fail the success floor
    # (newest-N scoring: 1 ok + 19 fails = 5% < 50% floor => ineligible).
    idx.record("m1", "retain", "request", "ok", 900.0)
    for _ in range(19):
        idx.record("m1", "retain", "request", "http-429", 500.0)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert any("sweep backstop" in v for v in body["degraded"].values())


def test_health_ok_with_a_serving_model(client):
    c, idx = client
    for _ in range(4):
        idx.record("m1", "retain", "request", "ok", 900.0)
    assert c.get("/health").status_code == 200


def test_mesh_status_exposes_rank_inputs(client):
    """/mesh/status must be self-explanatory: ranking + the inputs behind it.
    Added after the blended-score era where consumers saw order but never why."""
    c, _ = client
    body = c.get("/mesh/status").json()
    retain = body["aliases"]["auto/retain"]
    assert "ranking_all" in retain and "rank_inputs" in retain
    assert "scores" in retain and "timeouts" in retain
    assert set(retain["ranking_all"]) == set(retain["rank_inputs"]), (
        "every ranked model must have its bucket/tier reported"
    )


def test_v1_models_lists_aliases_and_upstream(client):
    c, _ = client
    body = c.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    for alias in ("auto/retain", "auto/consolidation", "auto/reflect", "auto/evolve"):
        assert alias in ids


def test_raw_path_rejects_fidelity_fail_body(client):
    """Raw ids must not serve fidelity-fail payloads.

    Observed live 2026-09-05: 4x gpt-oss-20b `generic` fidelity-fails served
    200 after 11-30s burns. dial() records the fail but still returns the
    payload; the raw path gated on `payload is not None` only, while
    route()._dial gates on `att.status == OK`. Same predicate or the raw
    path accepts prose the cascade refuses.
    """
    c, _ = client
    r = c.post("/v1/chat/completions", json={
        "model": "bad1", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 502
    assert "attempt" in r.json()
