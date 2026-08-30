"""The eligibility floors must bind on a breaker-`down` model too.

audit 2026-08-30: live /mesh/status had minimaxai/minimax-m3 at rank 0 on BOTH
auto/retain and auto/reflect with success_rate 0.7% over n=140, ahead of two
`overloaded` (i.e. recently-succeeding) models in the same cascade. It stayed
there across a model rotation — the morning's leader was a different id with
the same shape (1%, n=138), so this is structural, not one bad model.

Cause: `Router.eligible()` returns EARLY for `down` (and `auth`) as soon as the
cooldown expires:

    if b["state"] == "down":
        return time.time() >= b["cooldown_until"]

That early return is above every op_class floor — success-rate, thin-evidence,
reject, fidelity and latency. So the breaker opening on a model makes it MORE
eligible than a model the breaker never touched: the floors that would have
excluded it are never reached. `availability_bucket` then reads its Score
(success_rate 0.007 > 0.0 => not BUCKET_FAILING) and calls it `healthy`, and
because ranking is quality-first a big-tier id sorts to #1 and takes live
memory traffic.

The read-only property from test_readonly_introspection.py must survive: this
is about WHICH answer eligible() gives, not about mutating breaker state.
"""

from __future__ import annotations

import time

from model_mesh.index import Index, OK
from model_mesh.router import Router, RouterConfig


def _seed(tmp_path):
    idx = Index(tmp_path / f"floor-{time.time_ns()}.db")
    for m in ("bad/big-120b", "good/small-8b"):
        idx.ensure_model(m)
    return idx


def _router(idx) -> Router:
    return Router(idx, "http://unused.invalid/v1", lambda: "k",
                  RouterConfig(min_samples_for_floor=4, min_success_rate=0.5))


def _measure(idx, mid, *, oks: int, fails: int, latency_ms: float = 500.0):
    for _ in range(fails):
        idx.record(mid, "retain", "request", "http-429", latency_ms)
    for _ in range(oks):
        idx.record(mid, "retain", "request", OK, latency_ms)


def test_expired_cooldown_does_not_bypass_the_success_floor(tmp_path):
    """A `down` model with expired cooldown still has to clear the floors."""
    idx = _seed(tmp_path)
    # The live shape: 140 samples, ~1% success.
    _measure(idx, "bad/big-120b", oks=1, fails=139)
    idx.breaker_set("bad/big-120b", state="down", consec_fails=145,
                    cooldown_s=120.0, cooldown_until=time.time() - 1.0)
    r = _router(idx)
    assert not r.eligible("bad/big-120b", "retain"), (
        "breaker-`down` + expired cooldown bypassed the success-rate floor — "
        "opening the breaker made a failing model MORE eligible"
    )


def test_failing_leader_does_not_outrank_a_working_model(tmp_path):
    """The end-to-end symptom: rank 0 on live memory traffic."""
    idx = _seed(tmp_path)
    _measure(idx, "bad/big-120b", oks=1, fails=139)
    idx.breaker_set("bad/big-120b", state="down", consec_fails=145,
                    cooldown_s=120.0, cooldown_until=time.time() - 1.0)
    _measure(idx, "good/small-8b", oks=9, fails=1)
    r = _router(idx)
    order = r.ranked(["bad/big-120b", "good/small-8b"], "retain")
    assert order and order[0] == "good/small-8b", (
        f"measured-failing model took rank 0 over a working one: {order}"
    )


def test_auth_expired_cooldown_also_respects_the_floors(tmp_path):
    """`auth` has the identical early-return; fix both or the twin survives."""
    idx = _seed(tmp_path)
    _measure(idx, "bad/big-120b", oks=1, fails=139)
    idx.breaker_set("bad/big-120b", state="auth", consec_fails=3,
                    cooldown_s=120.0, cooldown_until=time.time() - 1.0)
    r = _router(idx)
    assert not r.eligible("bad/big-120b", "retain"), (
        "`auth` + expired cooldown bypassed the floors (sibling of the "
        "`down` path — same early return)"
    )


def test_unmeasured_model_still_returns_after_cooldown(tmp_path):
    """The recovery path must survive: no evidence => still eligible.

    This is the property the early return was protecting. A model with no
    in-window samples scores None, clears every floor, and comes back when its
    cooldown expires — that must keep working, or the breaker becomes an
    execution.
    """
    idx = _seed(tmp_path)
    idx.breaker_set("good/small-8b", state="down", consec_fails=3,
                    cooldown_s=120.0, cooldown_until=time.time() - 1.0)
    r = _router(idx)
    assert r.eligible("good/small-8b", "retain"), (
        "a `down` model with expired cooldown and NO adverse evidence must "
        "return — otherwise nothing ever recovers"
    )


def test_eligible_stays_read_only(tmp_path):
    """Pins test_readonly_introspection's property across this change."""
    idx = _seed(tmp_path)
    _measure(idx, "bad/big-120b", oks=1, fails=139)
    idx.breaker_set("bad/big-120b", state="down", consec_fails=145,
                    cooldown_s=120.0, cooldown_until=time.time() - 1.0)
    r = _router(idx)
    r.eligible("bad/big-120b", "retain")
    r.ranked(["bad/big-120b"], "retain")
    assert idx.breaker_get("bad/big-120b")["state"] == "down", (
        "eligible()/ranked() mutated breaker state — introspection must be "
        "read-only (see tests/test_readonly_introspection.py)"
    )
