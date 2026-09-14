"""/mesh/score endpoint + the timer's single code path (Task 6).

Endpoint tests follow tests/test_http_contract.py's isolation style: offline
transport, TestClient against the real app. The timer and the endpoint MUST
run the same function — a test pinning the endpoint pins the timer's body.
"""
from __future__ import annotations

import pytest

from model_mesh.app import app
from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import model_mesh.app as A

    idx = Index(tmp_path / "score.db")
    idx.sync_catalog("nim", {"m1"})

    def transport(url, body, headers, timeout):
        return 200, {"choices": [{"message": {"content": '{"facts": ["x"]}'}}],
                     "model": "m1"}

    router = Router(idx, "http://up/v1", "k", RouterConfig(), transport=transport)
    monkeypatch.setattr(A, "INDEX", idx)
    monkeypatch.setattr(A, "ROUTER", router)
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c, idx


def test_score_endpoint_disabled_by_default(client, monkeypatch):
    """Ships off (forensics 2026-09-14); an explicit call must say so loudly
    rather than silently doing nothing."""
    import model_mesh.app as A
    monkeypatch.setitem(A.CFG.setdefault("scoring", {}), "enabled", False)
    c, _ = client
    r = c.post("/mesh/score")
    assert r.status_code == 409
    assert "disabled" in r.json()["error"]


def test_score_endpoint_enabled_returns_verdicts(client, monkeypatch):
    import model_mesh.app as A
    monkeypatch.setitem(A.CFG.setdefault("scoring", {}), "enabled", True)
    monkeypatch.setattr(
        A, "run_score_pass",
        lambda cfg, router, candidates_for, top_n=6: {"auto/retain": {"m1": "pass"}},
    )
    c, _ = client
    r = c.post("/mesh/score")
    assert r.status_code == 200
    assert r.json()["scored"]["auto/retain"]["m1"] == "pass"


def test_timer_and_endpoint_share_one_body():
    """The endpoint and the startup loop must run the SAME body, and that body
    must be the scorer; two copies drift, and the drift is silent."""
    import inspect

    import model_mesh.app as A
    assert "_score_once" in inspect.getsource(A.mesh_score)
    assert "_score_once" in inspect.getsource(A._scoring_loop)
    assert "run_score_pass" in inspect.getsource(A._score_once)
    import model_mesh.app as A2
    assert A2.app.router.lifespan_context.__name__ == "_lifespan"
