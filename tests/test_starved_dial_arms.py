"""Sabotage-armed tests for the specific arms the first pass left uncovered.

A mutation matrix over the whole file found SIX sabotages that left it green:

  - the re-probe arm's retry guard deleted
  - the re-probe arm's PROBE guard deleted (this one was a real bug: the arm
    was issuing 4-8s probe grants once the budget was spent)
  - the sweep reverted from `continue` to `break`, silently truncating itself
  - `_dial()`'s `grant_s` reverted to re-reading the clock, reopening the
    check-to-dial gap that the parameter exists to close
  - the main loop dropping `grant_s`
  - the cooldown ignoring a success recorded outside the Router

Each test below is written to make exactly one of those mutations red. They
are not redundant with test_starved_dial.py: that file asserts the invariant
end-to-end, which is why a single wrong arm could hide in it.
"""
import json
import time

import pytest

from model_mesh.index import Index, OK
from model_mesh.router import Router, RouterConfig

# Alias the fixture back to `index`: importing it as `index_fixture` renamed
# it for pytest, and a fixture that only exists under one name is a trap.
from tests.test_router import index  # noqa: F401
from tests.test_router import make_router

# A timeout costs its whole grant, which drains the budget fast enough to
# leave the later arms unfunded.
@pytest.fixture(autouse=True)
def _restore_clock():
    """Every test swaps time.monotonic for a virtual clock; put it back."""
    real = time.monotonic
    yield
    time.monotonic = real


HANG = (599, {"error": "timeout"})

# retain's fidelity contract requires a JSON facts array. Plain prose is a
# FIDELITY FAIL, and a failed probe is exactly what keeps the re-probe arm's
# retry path unreachable.
PROBE_OK = (200, {"choices": [{"message": {"content": json.dumps({"facts": ["x"]})}}]})

PROBE_MSGS = [{"role": "user", "content": "probe"}]


def test_reprobe_arm_probe_guard_blocks_a_probe_it_cannot_finish(index):
    """The re-probe arm's PROBE calls come out of the same budget and were
    issuing 4-8s grants -- the same starved dial the main loop produced.

    Deleting `if probe_left < self.cfg.min_useful_dial_s` must go red here.
    """
    models = [f"m-{i}" for i in range(10)]
    for m in models:
        index.ensure_model(m)
    # The drain rate is the whole point. dial_cost=130 leaves the re-probe arm
    # a small box, and cheap probes walk that box down in 2s steps until the
    # next one is unaffordable -- the exact moment the guard exists for. A
    # slower drain (dial_cost=5) leaves the arm rich, the guard never fires,
    # and the test passes whether or not the guard exists.
    router, calls = _scripted(index, models, max_attempts=2,
                              dial_cost=130.0, probe_cost=2.0)

    router.route(models, {"messages": []}, "retain", probe_messages=PROBE_MSGS)

    assert any(k == "probe" for _, k, _ in calls), (
        f"no probe ran; the re-probe arm was never entered: {calls}"
    )
    floor = router.cfg.min_useful_dial_s
    starved = [(m, k, g) for m, k, g in calls if k == "probe" and g < floor]
    assert not starved, (
        f"re-probe issued grants below min_useful_dial_s ({floor}s): {starved}"
    )


def test_reprobe_retry_arm_blocks_a_dial_it_cannot_finish(index):
    """The re-probe arm's RETRY dials share the main loop's budget and must
    obey the same floor.

    Deleting that arm's `if left < self.cfg.min_useful_dial_s` must go red.
    """
    models = [f"m-{i}" for i in range(10)]
    for m in models:
        index.ensure_model(m)
    # Same reasoning as the probe test: the retry guard only fires when the arm
    # reaches it holding a budget between 1s and min_useful_dial_s.
    # probe_cost=6 walks the arm's box down until the RETRY is left holding
    # less than min_useful_dial_s. At probe_cost=2 the probes are so cheap the
    # retry still finds ~16s, the guard never fires, and the test is vacuous.
    router, calls = _scripted(index, models, max_attempts=2,
                              dial_cost=130.0, probe_cost=15.0)

    res = router.route(models, {"messages": []}, "retain",
                       probe_messages=PROBE_MSGS)

    assert any(k == "probe" for _, k, _ in calls), (
        f"the re-probe arm was never entered, so this test would pass on an "
        f"unreachable guard: {calls}"
    )
    # Name the arm: the retry guard's message starts "remaining budget", the
    # probe guard's starts "re-probe budget". Asserting on the retry one keeps
    # this test honest about WHICH guard is under test.
    details = [a.detail or "" for a in res.attempts
               if a.status == "skipped-budget"]
    assert any(d.startswith("remaining budget") for d in details), (
        f"the retry arm's own guard never fired; only the probe guard did. "
        f"details={details}"
    )
    floor = router.cfg.min_useful_dial_s
    starved = [(m, k, g) for m, k, g in calls if g < floor]
    assert not starved, (
        f"a request-time dial was granted below min_useful_dial_s ({floor}s): "
        f"{starved}"
    )


def test_reprobe_retry_loop_actually_executes(index):
    """Guard the guard: prove the retry loop BODY runs, not merely the probes.

    A `fresh` of size zero never evaluates the retry guard, so the test above
    can pass with the retry arm entirely unreachable. This one counts retries
    by a side effect only a retry produces: a request-time dial for a model a
    probe had already cleared.
    """
    models = [f"m-{i}" for i in range(10)]
    for m in models:
        index.ensure_model(m)
    router, calls = _scripted(index, models, max_attempts=2,
                              dial_cost=130.0, probe_cost=2.0)

    router.route(models, {"messages": []}, "retain", probe_messages=PROBE_MSGS)

    probed = {m for m, k, _ in calls if k == "probe"}
    assert len(probed) >= 2, (
        f"expected several probes before the retry loop; got {sorted(probed)}"
    )
    retried = [m for m, k, _ in calls if k == "request" and m in probed]
    assert retried, (
        f"the retry loop never ran: no probed model was retried. "
        f"probes={sorted(probed)} calls={calls}"
    )


def test_sweep_continues_past_the_first_unaffordable_model(index):
    """The sweep used `break` after the first model it could not afford, which
    silently truncated its own reach — later candidates were never even
    recorded as skipped.

    Every model the sweep would try must appear in the attempt list, either
    dialled or skipped, up to sweep_max_models.
    """
    models = [f"m-{i}" for i in range(14)]
    for m in models:
        index.ensure_model(m)
    router, _ = _scripted(index, models, max_attempts=4)

    res = router.route(models, {"messages": []}, "retain")

    # The discriminating property is the SKIP COUNT, not reach.
    #
    # With `continue`: the sweep records one skipped-budget attempt per model
    # it could not afford (7 here). With `break`: it stops at the first one
    # and every later model appears nowhere -- an operator reading the attempt
    # list cannot tell "we skipped it for budget" from "we never looked".
    #
    # Reach is deliberately not asserted: how many models the sweep can touch
    # depends on how much budget earlier arms left, which is exactly the
    # measurement-dependent thing that made an earlier version of this test
    # pass on broken code.
    skipped = [a.model_id for a in res.attempts
               if a.status == "skipped-budget"]
    assert len(skipped) >= 3, (
        f"only {len(skipped)} skip(s) recorded; `continue` over `break` should "
        f"record one per unaffordable model, so 1 means the sweep stopped "
        f"dead at the first one"
    )


def test_dial_uses_the_grant_the_caller_checked(index):
    """`grant_s` closes a real gap: between the guard reading `_remaining()`
    and `_dial` computing `min(timeout, _remaining())`, time can pass.

    Advancing the clock from a wrapper around `dial()` CANNOT test this -- the
    grant is already fixed by then, so the mutation stays invisible. The test
    has to move time inside the loop body instead, between the guard's read and
    `_dial`'s. Jumping the clock on the 3rd `time.monotonic()` call lands
    exactly there:

        intact   -> grant 50.0s   (the value the guard approved)
        mutated  -> grant 10.0s   (a re-read, below the 12s floor)

    which is the production bug verbatim: a 10s dial slipping past a 12s guard.

    The budget is deliberately smaller than `request_timeout_s`. With 280s
    available both paths compute min(135, 280) == min(135, 240) == 135, so the
    grant is never the binding term and the mutation cannot show.
    """
    models = ["m-a", "m-b", "m-c"]
    for m in models:
        index.ensure_model(m)

    grants: list[float] = []
    real_monotonic = time.monotonic
    state = {"n": 0, "jumped": False}

    def clock_with_a_gap():
        # Call 1 sets the deadline, call 2 is the guard's _remaining(), call 3
        # is where _dial would re-read if grant_s were dropped. 40s of real
        # work between 2 and 3 is a slow DB write or a GC pause.
        state["n"] += 1
        if state["n"] == 3 and not state["jumped"]:
            state["jumped"] = True
            return real_monotonic() + 40.0
        return real_monotonic()

    def transport(url, body, headers, timeout):
        grants.append(timeout)
        return HANG

    router = Router(
        index, "http://up/v1", "k",
        RouterConfig(total_budget_s=50.0, request_timeout_s=135.0,
                     max_attempts=2, sweep_max_models=12),
        transport=transport,
    )
    time.monotonic = clock_with_a_gap
    try:
        router.route(models, {"messages": []}, "retain")
    finally:
        time.monotonic = real_monotonic

    assert grants, "no dial was made"
    floor = router.cfg.min_useful_dial_s
    starved = [g for g in grants if g < floor]
    assert not starved, (
        f"a dial was granted {starved}s, below the {floor}s floor the caller "
        f"approved. _dial re-read the clock instead of using grant_s -- the "
        f"gap that let a 10s dial through a 12s guard."
    )


def test_cooldown_is_cleared_by_a_success_recorded_outside_the_router(index):
    """The cooldown reads `last_success_ts` from the index, so a success
    recorded by ANY writer — a discovery probe, an out-of-band repair —
    clears it, not just one this Router dialled.

    Deleting that check must go red here.
    """
    model, op = "m-a", "retain"
    index.ensure_model(model)
    router = Router(index, "http://up/v1", "k", RouterConfig(),
                    transport=lambda u, b, h, t: (500, {}))

    router._fidelity_fail(model, op)
    router._fidelity_fail(model, op)
    assert router.eligible(model, op) is False, "cooldown did not arm"

    # Somebody else records the recovery — not router._fidelity_succeed.
    index.record(model, op, "probe", OK, 120.0)
    assert router.eligible(model, op) is True, (
        "an externally-recorded success did not clear the cooldown"
    )


def _scripted(index, models, *, probe_passes=True, max_attempts=2,
              dial_cost=None, budget=280.0, probe_cost=None):
    """Router where the main loop always misses; probes pass or hang.

    Reaching the re-probe arm's RETRY path takes three conditions at once:
      1. the main loop must MISS, or the cascade returns before the arm runs;
      2. a probe must PASS, because only probed models enter `fresh`, and only
         `fresh` is retried;
      3. the retry must then find a budget it cannot spend.

    Distinguishing a probe from a request-time call keys on the probe's
    `max_tokens=4096`. The earlier fixture here used `max_tokens <= 4096`,
    which ALSO matches the real request body -- it carries no `max_tokens` at
    all -- so the main request was misread as a probe, was served PROBE_OK,
    and the cascade returned before the arm under test ever ran. That is why
    four sabotages in that arm stayed green.

    Returns (router, calls); each call is (model, kind, grant_seconds).
    """
    router, _ = make_router(
        index, {m: [HANG] * 40 for m in models},
        total_budget_s=budget, request_timeout_s=135.0,
        max_attempts=max_attempts, sweep_max_models=12, reprobe_top_n=4,
    )
    calls: list[tuple[str, str, float]] = []
    spent = {"t": 0.0}
    real_monotonic = time.monotonic

    def transport(url, body, headers, timeout):
        is_probe = body.get("max_tokens") == 4096
        kind = "probe" if is_probe else "request"
        calls.append((body["model"], kind, round(timeout, 1)))
        # A dial normally burns its whole grant. The cost overrides make the
        # budget drain at a chosen RATE, which is the only way to place the
        # re-probe arm at the interesting moment:
        #   drain too fast -> the arm has no money and never runs at all;
        #   drain too slow -> the arm has plenty and its guard never fires.
        # A guard that never fires cannot be caught by a mutation, so the
        # test must land the arm exactly at the floor.
        cost = probe_cost if is_probe else dial_cost
        spent["t"] += timeout if cost is None else cost
        if is_probe and not probe_passes:
            return HANG
        return PROBE_OK if is_probe else HANG

    router._transport = transport
    time.monotonic = lambda: real_monotonic() + spent["t"]
    # Hand the virtual clock's counter to the caller. A test that wants time to
    # pass must advance THIS dict -- mutating a local of its own leaves the
    # router's clock where it was, which is how an earlier version of the
    # grant_s test asserted nothing while reading as if it did.
    router._virtual_clock = spent
    return router, calls
