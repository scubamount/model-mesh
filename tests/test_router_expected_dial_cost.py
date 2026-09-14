"""Eligibility latency floor: expected dial cost, not worst-case p95.

The floor's job is "can the cascade afford to dial this model, and still
retry?". It asked that with `p95_all_ms > latency_ceiling`, which cannot be
satisfied by a model that ever times out: a timeout sample records the
per-attempt timeout itself, and the ceiling is a FRACTION of that same
timeout, so the comparison is true by construction. With p95 over the newest
20 samples, ~2 timeouts (>=5%) made a model permanently ineligible; the
measured baseline on a shared NIM key is 8-19% timeouts per model, uniform
across vendors. Lanes collapsed to whichever model happened to hold 0
timeouts in-window: reflect 10 -> 1, consolidation 9 -> 1 (live 2026-09-14).

That double-charges one event. A slow response already costs the model a
breaker cooldown (`_on_transient_fail`, 30s -> 300s ladder) which is the
correct, minutes-scale handling of "this endpoint is overloaded right now".
The latency floor then re-charged it for a further 24h, and the breaker
disagreed in the obvious way: gemma read `healthy`/consec=0 while the floor
still rejected it for a timeout 78 minutes old.

The floor must NOT simply ignore timeouts. A perfect alternator (fast
successes interleaved with full-budget timeouts) never reaches
`breaker_threshold` consecutive failures, so the breaker cannot see it --
verified live: gemma's last 20 reflect samples had max run length 1. That is
the 2026-09-13 regression `tests/test_index_p95_all.py` exists to pin.

Expected cost separates the two cases with one number:

    expected = success_rate * p95_ms + (1 - success_rate) * timeout_cost
    reject when 2 * expected > total_budget_s   (no room for a real retry)

    occasional overload  0.85*40s  + 0.15*135s = 54s  -> 108s  ADMIT
    incumbent            1.00*51s              = 51s  -> 102s  ADMIT
    alternator           0.50*61s  + 0.50*135s = 98s  -> 196s  (grows to reject
                                                        as success drops)
    slow-but-reliable    1.00*150s             = 150s -> 300s  REJECT (2026-08-07
                                                        llama-3.3-70b case)
"""
from __future__ import annotations

from model_mesh.index import Index, OK
from model_mesh.router import Router, RouterConfig


def _router(index, **cfg_kw):
    cfg = RouterConfig(**cfg_kw) if cfg_kw else RouterConfig()
    return Router(index, "http://up.example/v1", "k", cfg)


def _seed(index, mid, op, pattern):
    """pattern: list of (status, latency_ms) replayed in order."""
    index.ensure_model(mid)
    for status, ms in pattern:
        index.record(mid, op, "request", status, ms, 100 if status == OK else 0)


TIMEOUT = "http-598"


def test_occasional_overload_stays_eligible(tmp_path):
    """THE DEFECT. 85% success at 40s + 15% timeout is gemma's live reflect
    shape: expected dial 54s against the incumbent's 51s, three seconds apart.
    The old floor rejected it on a worst-case p95 of 135s."""
    ix = Index(tmp_path / "mesh.db")
    pattern = [(OK, 40_000.0)] * 17 + [(TIMEOUT, 135_000.0)] * 3
    _seed(ix, "m", "reflect", pattern)
    r = _router(ix)
    assert r.eligible("m", "reflect") is True


def test_alternator_still_excluded(tmp_path):
    """The 2026-09-13 regression must stay caught.

    Uses the case as it ACTUALLY measured, not a tidied-up 50/50: gemma/reflect
    was 0.32 success with 67% timeouts over 2,043 samples. A model failing two
    dials out of three is excluded by the success-rate floor
    (min_success_rate=0.5), which is the floor that owns "fails too often".

    A literal 50/50 alternator sits exactly ON that boundary and is admitted by
    design — at even odds with a retry available the cascade does better trying
    it than refusing it, and the breaker still reacts if the failures cluster.
    Pinning 50/50 as excluded would be pinning a number nobody measured.
    """
    ix = Index(tmp_path / "mesh.db")
    pattern = []
    for _ in range(6):
        pattern += [(OK, 61_000.0), (TIMEOUT, 135_000.0), (TIMEOUT, 135_000.0)]
    _seed(ix, "m", "reflect", pattern)
    r = _router(ix)
    s = ix.score("m", "reflect")
    assert s is not None
    assert s.success_rate < 0.5, s.success_rate
    assert r.eligible("m", "reflect") is False


def test_slow_but_reliable_still_excluded(tmp_path):
    """The 2026-08-07 case the floor was built for: 100% success, but one
    attempt eats the budget so a retry cannot fit."""
    ix = Index(tmp_path / "mesh.db")
    _seed(ix, "m", "reflect", [(OK, 150_000.0)] * 20)
    r = _router(ix)
    assert r.eligible("m", "reflect") is False


def test_fast_and_reliable_eligible(tmp_path):
    ix = Index(tmp_path / "mesh.db")
    _seed(ix, "m", "reflect", [(OK, 5_000.0)] * 20)
    r = _router(ix)
    assert r.eligible("m", "reflect") is True


def test_lane_recovers_breadth(tmp_path):
    """Live reflect distribution (2026-09-14). The lane served ONE model while
    four more were healthy by every other floor."""
    ix = Index(tmp_path / "mesh.db")
    live = {
        # model, successes@latency, timeouts  (newest-20 shape, measured)
        "ising":  ([(OK, 51_000.0)] * 20),
        "gemma":  ([(OK, 40_000.0)] * 17 + [(TIMEOUT, 135_000.0)] * 3),
        "gptoss": ([(OK, 5_400.0)] * 13 + [(TIMEOUT, 135_000.0)] * 7),
        "laguna": ([(OK, 101_000.0)] * 16 + [(TIMEOUT, 135_000.0)] * 4),
        "glm":    ([(OK, 74_700.0)] * 4 + [(TIMEOUT, 135_000.0)] * 3),
    }
    for mid, pattern in live.items():
        _seed(ix, mid, "reflect", pattern)
    r = _router(ix)
    ranked = r.ranked(list(live), "reflect")
    assert len(ranked) >= 4, f"lane still collapsed: {ranked}"


def test_widening_does_not_change_ordering_rules(tmp_path):
    """Widening the pool must not change HOW models are ordered.

    Ordering is (availability bucket, quality tier, latency) — unchanged by
    this commit, which only touches eligibility. Both models here land in the
    same bucket and tier, so the latency tiebreak decides, and gemma really is
    faster when it works (40s vs 51s). Asserting "the incumbent stays #1" would
    pin the opposite of the documented rule; what matters is that the fix ADDS
    a candidate rather than reordering the ones already there.
    """
    ix = Index(tmp_path / "mesh.db")
    _seed(ix, "ising", "reflect", [(OK, 51_000.0)] * 20)
    _seed(ix, "gemma", "reflect",
          [(OK, 40_000.0)] * 17 + [(TIMEOUT, 135_000.0)] * 3)
    r = _router(ix)
    ranked = r.ranked(["gemma", "ising"], "reflect")
    # the defect was a one-model lane; both must now be dialable
    assert set(ranked) == {"gemma", "ising"}, ranked
    # and the order is the latency tiebreak within an equal bucket/tier
    def _p95(m: str) -> float:
        s = ix.score(m, "reflect")
        assert s is not None
        return s.p95_ms

    assert ranked == sorted(ranked, key=_p95), ranked


def test_budget_scales_the_budget_floor(tmp_path):
    """No new knob: the budget floor derives from total_budget_s.

    Uses a model the LATENCY ceiling admits (fast successes) but that fails
    often enough for expected+retry to overrun a small budget. A 150s model
    cannot be rescued by raising the budget — the latency ceiling is derived
    from request_timeout, not the budget, and that separation is the point:
    "is one success affordable" and "can we afford to dial and retry" are two
    questions with two different sources.
    """
    ix = Index(tmp_path / "mesh.db")
    _seed(ix, "m", "reflect",
          [(OK, 30_000.0)] * 11 + [(TIMEOUT, 135_000.0)] * 9)  # 0.55 success
    assert _router(ix, total_budget_s=200.0).eligible("m", "reflect") is False
    assert _router(ix, total_budget_s=400.0).eligible("m", "reflect") is True
