"""/mesh/status must not lose evidence when the floors exclude the pool.

audit 2026-08-30: `scores` and `rank_inputs` were built by iterating `ranked`.
That is fine while something is admitted, and silently wrong the moment the
eligibility floors correctly exclude everything during a provider episode:
`ranked` is empty, so both dicts came back empty, and a consumer could not tell

    "we have never measured these models"        (unproven — whitelist rot)

from

    "we measured them 140 times, all failing"    (episode — provider outage)

Those need OPPOSITE responses. check-mesh-pool-breadth read the empty dict and
reported "every candidate is unproven, so routing is guesswork", sending an
operator after whitelist rot in the middle of an outage.

Evidence belongs to the model, not to its current rank. `ranking`/`ranking_all`
still report who is admitted.

Driven through the real HTTP route (TestClient), per tests/test_http_contract.py
— the serialized response is what consumers actually read.
"""

from __future__ import annotations

import pytest

from model_mesh.app import app
from model_mesh.index import Index, OK
from model_mesh.router import Router, RouterConfig


def _client(tmp_path, monkeypatch, *, status: str, n: int):
    import model_mesh.app as A

    idx = Index(tmp_path / "status-evidence.db")
    models = {"bad/big-120b", "bad/other-70b"}
    idx.sync_catalog("nim", models)
    for m in models:
        idx.ensure_model(m)
        for _ in range(n):
            idx.record(m, "retain", "request", status, 500.0)

    router = Router(idx, "http://up/v1", "k", RouterConfig(
        min_samples_for_floor=4, min_success_rate=0.5,
    ), transport=lambda *a: (429, {"error": "busy"}))

    monkeypatch.setattr(A, "INDEX", idx)
    monkeypatch.setattr(A, "ROUTER", router)
    monkeypatch.setattr(A, "fetch_catalog", lambda *a, **k: {"data": []})
    monkeypatch.setattr(
        A, "CFG",
        {"provider": {"name": "nim"},
         "aliases": {"auto/retain": {"op_class": "retain",
                                     "include": [], "exclude": []}}},
    )
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture()
def excluded_pool(tmp_path, monkeypatch):
    """Every candidate measured failing => the floors admit nobody."""
    with _client(tmp_path, monkeypatch, status="http-429", n=10) as c:
        yield c.get("/mesh/status").json()["aliases"]["auto/retain"]


def test_scores_survive_a_whole_pool_exclusion(excluded_pool):
    body = excluded_pool
    assert body["ranking_all"] == [], "precondition: the floors admit nobody"
    assert body["scores"], (
        "/mesh/status dropped all evidence when the floors excluded the pool — "
        "'measured failing' is now indistinguishable from 'never measured'"
    )
    for mid, s in body["scores"].items():
        assert s["n"] == 10, f"{mid} lost its sample count"
        assert s["success_rate"] == 0.0, f"{mid} lost its failure evidence"


def test_rank_inputs_survive_a_whole_pool_exclusion(excluded_pool):
    """The sibling dict — fixing one and not the other leaves the twin."""
    assert excluded_pool["rank_inputs"], (
        "/mesh/status dropped the bucket table exactly when routing got "
        "interesting"
    )
    assert {v["bucket"] for v in excluded_pool["rank_inputs"].values()} == {
        "failing"
    }


def test_admitted_models_still_reported(tmp_path, monkeypatch):
    """The normal path is unchanged: evidence AND admission both visible."""
    with _client(tmp_path, monkeypatch, status=OK, n=6) as c:
        body = c.get("/mesh/status").json()["aliases"]["auto/retain"]
    assert body["ranking_all"], "healthy models must still be admitted"
    for s in body["scores"].values():
        assert s["success_rate"] == 1.0
