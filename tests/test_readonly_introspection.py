"""Introspection endpoints must not mutate breaker state.

audit 2026-08-25: Router.eligible() flipped `down`/`auth`->`recovering` on
cooldown expiry, and it was called by ranked() (the /mesh/status path) and by
/health. So every `curl /mesh/status` or `curl /health` — including the
installer's own verify step and any monitoring — wrote to the breaker table,
turning a read API into a write. The recovery-window transition now lives in
dial() (attempt time) and eligible() is read-only; these tests pin that.
"""

from __future__ import annotations

import time

import pytest

from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig


def _seeded_router(tmp_path, state: str) -> tuple[Router, str]:
    idx = Index(tmp_path / "mesh.db")
    idx.sync_catalog("nim", {"m1"})
    idx.breaker_set(
        "m1", state=state, consec_fails=3,
        cooldown_s=120.0, cooldown_until=time.time() - 1.0,  # expired
    )
    r = Router(idx, "http://up/v1", "k", RouterConfig(),
               transport=lambda *a: (200, {"choices": [{"message": {"content": '{"facts":["x"]}'}}]}))
    return r, "m1"


@pytest.mark.parametrize("state", ["down", "auth"])
def test_ranked_does_not_transition_breaker(tmp_path, state):
    r, m = _seeded_router(tmp_path, state)
    assert r.index.breaker_get(m)["state"] == state
    # ranked() is what /mesh/status calls — pure introspection.
    order = r.ranked([m], "retain")
    assert m in order, "expired-cooldown model must still be ranked"
    assert r.index.breaker_get(m)["state"] == state, (
        "/mesh/status mutated breaker state — it must be read-only"
    )


@pytest.mark.parametrize("state", ["down", "auth"])
def test_health_does_not_transition_breaker(tmp_path, state):
    r, m = _seeded_router(tmp_path, state)
    assert r.index.breaker_get(m)["state"] == state
    # /health calls eligible() directly for each alias.
    from model_mesh.config import DEFAULTS
    CFG = dict(DEFAULTS)
    CFG["aliases"] = {"auto/retain": {"op_class": "retain", "include": [], "exclude": []}}
    healthy = [x for x in [m] if r.eligible(x, "retain")]
    assert healthy, "expired-cooldown model must still read as eligible"
    assert r.index.breaker_get(m)["state"] == state, (
        "/health mutated breaker state — it must be read-only"
    )


@pytest.mark.parametrize("state", ["down", "auth"])
def test_dial_still_transitions_on_real_attempt(tmp_path, state):
    r, m = _seeded_router(tmp_path, state)
    payload, _ = r.dial(m, {"messages": []}, "retain", source="request")
    assert payload is not None
    # _transition_for_attempt flips down/auth -> recovering at dial time; a
    # successful attempt then closes to healthy. Either proves the recovery
    # transition fired (a `down`/`auth` model would otherwise never serve).
    assert r.index.breaker_get(m)["state"] in ("recovering", "healthy"), (
        "a real dial must exercise the recovery transition for an "
        "expired-cooldown model"
    )


@pytest.mark.parametrize("state", ["down", "auth"])
def test_route_still_recovers_through_cascade(tmp_path, state):
    r, m = _seeded_router(tmp_path, state)
    res = r.route([m], {"messages": []}, "retain")
    assert res.ok, "expired-cooldown model must be retried by the cascade"
    assert r.index.breaker_get(m)["state"] in ("recovering", "healthy"), (
        "cascade recovery must flip down/auth -> recovering at attempt time"
    )
