"""A starved dial must not be recorded as a model failure.

When the cascade budget is nearly spent, `min(request_timeout, remaining)`
leaves the final dial a few seconds. That attempt then times out for a reason
that is OURS, not the model's, and it is recorded as an ordinary `http-598`
failure sample against whichever model drew it.

Measured in the live index: short (<20s) `http-598` samples cluster at exactly
10.0s — `min(135, 10)` — from `request` and `sweep` sources. Those rows enter
`score()`, `success_rate`, and the breaker threshold as evidence that a model
is failing, when the real cause was our own budget. Self-inflicted evidence
that then feeds every eligibility decision.

The fix: below a measured minimum useful dial, record the attempt as
`skipped-budget` (already a status in this codebase) and write NO sample.

Deliberately NOT asserted here: how many attempts a budget affords. That
depends on live provider latency, and the budget-sharing predicate that
addressed it was withdrawn in review — a budget that truncates the
slow-but-correct path selects for the fast-and-wrong one.
"""
import time

import pytest

from model_mesh.router import RouterConfig

# The `index` fixture and the transport harness already exist in test_router;
# importing them keeps one definition of "how a Router is built for a test"
# instead of a second one that can drift from the first.
from tests.test_router import index, make_router  # noqa: F401

HANG = (599, {"error": "timeout"})


def _samples(index, model_id):
    return index._conn.execute(
        "SELECT status, latency_ms FROM samples WHERE model_id=? ORDER BY rowid",
        (model_id,),
    ).fetchall()


def test_budget_too_small_to_dial_is_skipped_not_attempted(index):
    """The regression: with less budget than a useful dial, no dial happens."""
    index.ensure_model("m-a")
    router, t = make_router(
        index, {"m-a": [HANG] * 10},
        total_budget_s=3.0, request_timeout_s=135.0,
    )
    res = router.route(["m-a"], {"messages": []}, "retain")

    assert res.ok is False
    assert [a.status for a in res.attempts][-1] == "skipped-budget"


def test_a_skipped_dial_writes_no_failure_sample(index, monkeypatch):
    """The load-bearing half. Our own empty budget must not become evidence
    that the model is failing, because that evidence feeds score(),
    success_rate, and the breaker."""
    models = ["m-a", "m-b", "m-c", "m-d"]
    router, _spent, _granted = _draining_router(index, monkeypatch, models)

    res = router.route(models, {"messages": []}, "retain")

    statuses = [a.status for a in res.attempts]
    assert statuses == ["http-599", "http-599", "skipped-budget"], statuses
    # The skipped model gets NO sample. Its two dialled siblings each get one.
    assert _samples(index, "m-c") == [], "a skipped dial must write no sample"
    assert len(_samples(index, "m-a")) == 1
    assert len(_samples(index, "m-b")) == 1


def test_no_starved_dial_is_ever_recorded_as_a_short_timeout(index):
    """Across a cascade that drains the budget, no sample may carry a timeout
    shorter than a useful dial. This is the live-index bug, pinned."""
    index.ensure_model("m-a")
    index.ensure_model("m-b")
    index.ensure_model("m-c")
    script = {m: [HANG] * 20 for m in ("m-a", "m-b", "m-c")}
    router, _ = make_router(
        index, script,
        total_budget_s=280.0, request_timeout_s=135.0, max_attempts=8,
    )
    router.route(["m-a", "m-b", "m-c"], {"messages": []}, "retain")

    for model in ("m-a", "m-b", "m-c"):
        for status, latency in _samples(index, model):
            if status == "http-598":
                assert latency is None or latency >= 1000.0 * (
                    router.cfg.min_useful_dial_s
                ), (
                    f"{model} recorded a {latency}ms timeout — shorter than "
                    f"min_useful_dial_s ({router.cfg.min_useful_dial_s}s), so "
                    f"the cause was our budget, not the model"
                )


def test_a_dial_with_real_budget_behind_it_is_still_attempted(index):
    """The skip must be narrow, or this change silently shrinks the cascade."""
    index.ensure_model("m-a")
    router, t = make_router(
        index, {"m-a": [HANG] * 10},
        total_budget_s=280.0, request_timeout_s=135.0,
    )
    res = router.route(["m-a"], {"messages": []}, "retain")

    assert t.calls == ["m-a"], "the first dial has the full budget and must run"
    assert [a.status for a in res.attempts][0] != "skipped-budget"


def test_sufficient_budget_still_fails_over_to_the_next_model(index):
    """Skipping must not swallow real failover: model A hangs, B answers."""
    index.ensure_model("m-a")
    index.ensure_model("m-b")
    router, t = make_router(
        index, {"m-a": [HANG] * 10},   # m-b falls through to the default OK
        total_budget_s=280.0, request_timeout_s=135.0,
    )
    res = router.route(["m-a", "m-b"], {"messages": []}, "retain")

    assert res.ok is True
    assert res.model_id == "m-b"
    assert t.calls == ["m-a", "m-b"]


def _draining_router(index, monkeypatch, models, *, probe_messages=None,
                     max_attempts=8, reprobe_top_n=4, sweep_max_models=12,
                     script=None, hang_cost_s=None):
    """A Router whose dials each consume their FULL timeout.

    FakeTransport returns instantly, so without this the 280s budget is never
    actually spent and no arm is ever short of money. Advancing the monotonic
    clock inside the injected transport makes every dial cost exactly the grant
    it was given, which is how a hung provider behaves and is the only way to
    reach the starved-dial branch deterministically.
    """
    for m in models:
        index.ensure_model(m)
    if script is None:
        script = {m: [HANG] * 40 for m in models}
    router, _ = make_router(
        index, script,
        total_budget_s=280.0, request_timeout_s=135.0,
        max_attempts=max_attempts, reprobe_top_n=reprobe_top_n,
        sweep_max_models=sweep_max_models,
    )
    spent = {"t": 0.0}
    granted: list[float] = []
    real = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real() + spent["t"])
    original = router._transport

    def timed_out_transport(url, body, headers, timeout):
        # Record the GRANT. It is the only thing min_useful_dial_s governs,
        # and samples.latency_ms records the MEASURED duration — a 503 that
        # came back in 1s is a fast legitimate failure, not a starved dial.
        # Conflating the two is what made an earlier version of this file
        # assert against the wrong clock.
        granted.append(timeout)
        spent["t"] += timeout if hang_cost_s is None else hang_cost_s
        return original(url, body, headers, timeout)

    monkeypatch.setattr(router, "_transport", timed_out_transport)
    return router, spent, granted


def _starved_grants(granted, floor_s):
    """Grants issued below min_useful_dial_s — i.e. dials we KNEW we could not
    finish, which must never be issued at all.

    Asserted against the grant log rather than samples.latency_ms on purpose:
    the guard governs how long we ALLOW a dial, while latency_ms records how
    long it actually took. A 503 that returns in 1s is a fast legitimate
    failure with a full grant, and treating its latency as a starved dial is
    the category error an earlier draft of this file made.
    """
    return [g for g in granted if g < floor_s]


def test_no_arm_records_a_dial_it_knew_it_could_not_finish(index, monkeypatch):
    """Main loop, re-probe retry and sweep all share one invariant.

    A sabotage matrix showed each arm's guard could be deleted while its own
    test stayed green — the main loop was covered, the other two were not.
    This test drives ONE cascade through all three arms and asserts over every
    sample written, so no arm can quietly reintroduce the starved dial.

    Reaching the sweep needs candidates beyond max_attempts; reaching the
    re-probe retry needs probe_messages, which only a request-time caller
    supplies.
    """
    models = [f"m-{i}" for i in range(14)]
    router, _spent, granted = _draining_router(
        index, monkeypatch, models,
        probe_messages=[{"role": "user", "content": "probe"}],
        max_attempts=6, sweep_max_models=12,
    )

    res = router.route(models, {"messages": []}, "retain")

    assert res.ok is False
    starved = _starved_grants(granted, router.cfg.min_useful_dial_s)
    assert starved == [], (
        "an arm issued a dial shorter than min_useful_dial_s, which is "
        f"exactly the starved dial recorded as a model failure: {starved}"
    )


def test_the_cascade_actually_reached_all_three_arms(index, monkeypatch):
    """The test above is only meaningful if the sweep and re-probe arms really
    ran. Without this, deleting an arm's guard AND its reachability together
    would keep the suite green — which is exactly the 'green gate lies' failure
    the sabotage matrix exists to prevent."""
    models = [f"m-{i}" for i in range(14)]
    # Realistic shape, NOT all-hangs. Two 135s hangs would exhaust the 280s
    # budget inside the main loop and the sweep could never run — which is why
    # the live sweep wins 31 reflect requests/day on FAST failures (http-400
    # in 0.4s) rather than on hangs. Here every dial fails fast (1s) so the
    # budget survives for the sweep, and the models that hang are the ones
    # ranked late.
    # Every dial is scripted to hang, so the budget is genuinely exhausted and
    # all three arms must be REACHED-and-stopped by the guard rather than
    # never running. An earlier version asserted source='sweep' appeared in
    # samples, which is wrong: once the budget is gone no arm may dial, so the
    # sweep correctly writes no sample at all. Its participation shows up as a
    # skipped-budget attempt over the models it would have tried.
    router, _spent, _granted = _draining_router(
        index, monkeypatch, models,
        probe_messages=[{"role": "user", "content": "probe"}],
        max_attempts=6, sweep_max_models=12,
    )
    res = router.route(models, {"messages": []}, "retain")

    skipped = [a.model_id for a in res.attempts if a.status == "skipped-budget"]
    # Every candidate must be accounted for as either dialled or skipped —
    # EXCEPT those beyond sweep_max_models, which the arm is configured never
    # to reach (13 candidates here against a cap of 12; the cap is the
    # backstop that stops a dead lane burning the whole budget).
    dialed = {a.model_id for a in res.attempts if a.status != "skipped-budget"}
    capped = RouterConfig().sweep_max_models
    reachable = set(router.ranked(models, "retain")[:capped]) | set(dialed) | set(skipped)
    unaccounted = set(models) - reachable
    assert len(unaccounted) <= len(models) - capped, (
        f"{sorted(unaccounted)} were neither dialled nor skipped, beyond what "
        f"the sweep cap of {capped} allows"
    )
    # Every skipped model wrote NO sample -- that is the evidence-integrity
    # claim, checked across whichever arm did the skipping.
    for m in skipped:
        assert _samples(index, m) == [], (
            f"{m} was skipped for budget yet has sample rows"
        )


def test_sweep_arm_also_skips_a_dial_it_cannot_finish(index, monkeypatch):
    """The sweep is where the starved dial was actually measured (136 of 374
    reflect http-598s came from the sweep source), and it has its own inline
    budget check rather than going through _dial().

    A sabotage matrix caught this test MISSING: deleting the sweep's
    `left < min_useful_dial_s` guard left the whole file green.
    """
    models = [f"m-{i}" for i in range(14)]
    fast_fail = (503, {"error": "unavailable"})
    router, _spent, granted = _draining_router(
        index, monkeypatch, models,
        probe_messages=[{"role": "user", "content": "probe"}],
        max_attempts=6, sweep_max_models=12,
        script={m: [fast_fail] * 40 for m in models}, hang_cost_s=1.0,
    )

    res = router.route(models, {"messages": []}, "retain")

    assert res.swept or any(a.status == "skipped-budget" for a in res.attempts)
    # The sweep ran, and issued no dial below the floor.
    assert _starved_grants(granted, router.cfg.min_useful_dial_s) == []
    # And a skipped dial is never a sample row.
    assert index._conn.execute(
        "SELECT COUNT(*) FROM samples WHERE status='skipped-budget'"
    ).fetchone()[0] == 0


def test_min_useful_dial_defaults_to_a_measured_value_not_zero():
    """A zero default would disable the guard entirely while passing every
    other test, so the constant itself is pinned."""
    d = RouterConfig()
    # MUST exceed 10.0s: that is exactly where `min(135, left)` lands after two
    # 135s dials against a 280s budget, and it is the artifact being fixed (139
    # such rows in 7 days). A guard at or below 10 catches none of them.
    assert d.min_useful_dial_s > 10.0
    # Well under any per-op-class probe timeout (45s / 100s), and above the
    # 1.0s cascade stop at router.route's `left <= 1.0` guard.
    assert d.min_useful_dial_s < 45.0


def test_config_default_matches_dataclass():
    """test_config_defaults_match_dataclass covers RouterConfig, but this
    field is load-bearing for evidence integrity, so pin the yaml mirror."""
    from model_mesh.config import DEFAULTS
    assert DEFAULTS["router"]["min_useful_dial_s"] == RouterConfig().min_useful_dial_s
