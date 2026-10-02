"""A fidelity-fail is stochastic; a 7-day bench for it was never measured.

The reject gate (`unrebutted_reject`) keeps its deterministic two-failure /
seven-day rule: the provider parsed the body and refused the shape, so a
retry fails identically. A `fidelity-fail` is a different failure mode — an
HTTP 200 whose content did not obey the contract — and the live index shows it
arrives in BURSTS, not as a stable property:

    nemotron-3-super-120b-a12b / reflect — 308 ok / 330 samples (93.3% healthy),
    23 fidelity-fails in 7d, median inter-failure gap 241s, 8 of 23 gaps under
    120s, one 30.5-minute cluster. The gate armed 3x in 24h.

Under the old rule each arming cost REJECT_RECHECK_S (7 days) of ranked depth
on a lane whose pool is 11 models wide. `reflect ranked=1` was that gate, and
96 of 123 client failures in 24h died after exactly two dials because the main
loop had one candidate to give.

These tests pin the replacement: a short, per-(model, op_class), in-memory
cooldown that decays with time, never persists, and is armed/cleared by the
same samples the gate used to read.

Deliberately NOT tested here: that any specific model ranks a certain way, or
that any latency budget affords N attempts. Those need live provider traffic.
"""
import threading
import time

import pytest

from model_mesh.router import Router, RouterConfig


class FakeIndex:
    """Minimal index stand-in: eligible() only needs the two reject queries."""

    def __init__(self, reject=None, fidelity=0):
        self._reject = reject
        self._fidelity = fidelity
        self.breaker_get_calls = []

    def unrebutted_reject(self, model_id, op_class):
        return self._reject

    def unrebutted_fidelity_fails(self, model_id, op_class, need=2):
        return self._fidelity if self._fidelity >= need else None

    def score(self, model_id, op_class):
        return None

    def breaker_get(self, model_id):
        self.breaker_get_calls.append(model_id)
        return {"state": "healthy", "cooldown_until": 0.0}

    def breaker_set(self, *a, **k):
        pass

    def last_sample_ts(self, model_id, op_class):
        return time.time()

    def last_success_ts(self, model_id, op_class):
        # None = no success on record, so the in-process path is what clears
        # the cooldown unless a test overrides this.
        return None


@pytest.fixture
def cfg():
    return RouterConfig(fidelity_cooldown_base_s=60.0, fidelity_cooldown_max_s=900.0)


@pytest.fixture
def router(cfg):
    r = Router.__new__(Router)
    r.cfg = cfg
    r.index = FakeIndex(fidelity=2)   # the DB already holds two consecutive fails
    # eligible() also consults the quota ladder, so the stub Router needs the
    # real __init__ attributes for every state it touches.
    r._quota = {}
    r._quota_lock = threading.Lock()
    r._provider_pause_until = 0.0
    r._init_cooldown_state()
    return r


MODEL = "nvidia/nemotron-3-super-120b-a12b"


def test_two_consecutive_fidelity_fails_arm_a_cooldown_not_a_seven_day_bench(router):
    """The regression this whole file exists for: eligible() must go False now."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is False


def test_one_fidelity_fail_does_not_bench(router):
    """A single violation is not a verdict — the cascade already absorbed it."""
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is True


def test_cooldown_is_minutes_scale_not_days(router):
    """7 days is what we are removing. Base is 60s, capped at 900s."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    remaining = router._fidelity_cooldown_left(MODEL, "reflect")
    assert 0 < remaining <= router.cfg.fidelity_cooldown_base_s
    assert remaining < 3600


def test_cooldown_expires_and_the_model_returns(router):
    """The user's question, answered as a test: a model must not be lost forever.

    Exclude the model, let the clock pass the window, and it is back. This is
    the property `REJECT_RECHECK_S` failed to give on a stochastic signal.
    """
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is False

    router._advance_clock(router.cfg.fidelity_cooldown_base_s + 1.0)
    assert router.eligible(MODEL, "reflect") is True


def test_a_success_clears_the_cooldown_immediately(router):
    """Recovery must not wait out the window — that is the whole point."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is False

    router._fidelity_succeed(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is True


def test_repeat_fails_inside_the_window_do_not_extend_it(router):
    """A burst must not ratchet the penalty. One arming, one window."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    first = router._fidelity_cooldown_left(MODEL, "reflect")

    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router._fidelity_cooldown_left(MODEL, "reflect") <= first


def test_cooldown_is_scoped_per_op_class(router):
    """A model can obey one contract and not another; one lane's burst must
    not bench the model everywhere."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is False
    assert router.eligible(MODEL, "retain") is True


def test_cooldown_is_scoped_per_model(router):
    other = "nvidia/nemotron-3-ultra-550b-a55b"
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(other, "reflect") is True


def test_deterministic_reject_gate_is_untouched(router):
    """The 7-day rule stays exactly where it belongs: http-400/413/422.

    `test_deterministically_rejecting_model_is_ineligible` (test_pool.py:632)
    and `test_reject_goes_stale_and_admits_a_retry` (test_pool.py:727) cover the
    real index. This pins that the fidelity work did not widen the reject path.
    """
    router.index._reject = {"since": time.time(), "status": "http-400"}
    router._fidelity_fail(MODEL, "reflect")
    assert router.eligible(MODEL, "reflect") is False


def test_restart_drops_the_cooldown_and_the_model_is_tried_again(router):
    """In-memory on purpose: a stale row must not bench a model across a
    restart. Mirrors the _quota precedent."""
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")

    fresh = Router.__new__(Router)
    fresh.cfg = router.cfg
    fresh.index = router.index
    fresh._quota = {}
    fresh._quota_lock = threading.Lock()
    fresh._provider_pause_until = 0.0
    fresh._init_cooldown_state()
    assert fresh.eligible(MODEL, "reflect") is True


def test_cooldown_state_is_lock_guarded(router):
    """`eligible()` is called concurrently by /mesh/status and /health.

    A new shared mutable dict read from a read-only-by-contract method without
    a lock is a data race, so the lock is part of the design, not an extra.
    """
    router._fidelity_fail(MODEL, "reflect")
    router._fidelity_fail(MODEL, "reflect")

    errors = []

    def hammer():
        try:
            for _ in range(300):
                router.eligible(MODEL, "reflect")
        except Exception as exc:            # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert hasattr(router, "_fidelity_lock"), "cooldown dict must be lock-guarded"


def test_no_stored_fidelity_fails_setting_remains():
    """`fidelity_fails_for_floor` is gone from config; the two float fields
    replace it. A leftover int field would silently keep a 7-day window."""
    from model_mesh.config import DEFAULTS

    r = DEFAULTS["router"]
    assert "fidelity_fails_for_floor" not in r
    assert isinstance(r["fidelity_cooldown_base_s"], float)
    assert isinstance(r["fidelity_cooldown_max_s"], float)
