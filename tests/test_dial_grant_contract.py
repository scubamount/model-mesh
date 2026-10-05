"""A dial grant of zero or less must never reach the upstream.

Two defects of one class, found by the 2026-10-05 holistic pass:

1. `dial()` and `probe_verdict()` defaulted their timeout with
   `timeout or self.request_timeout(op_class)`. `0.0 or 135` is 135, so a
   caller that had computed a ZERO grant -- budget gone -- was handed a full
   135-second upstream call. The adversarial review of the failover spec named
   this falsy-zero; it survived because every call site happened to guard
   upstream of it.

2. The re-probe arm's probe call re-read `_remaining()` and `_box_left()` when
   building its timeout, instead of using the `probe_left` its own guard had
   just compared. Time passing between the two reads can produce a NEGATIVE
   grant that goes straight to the transport (reproduced below with a synthetic
   clock jump: -9930.0s; not observed in live traffic). It is the same
   check-to-dial gap `grant_s` closed for request dials, left open on probes.

These are tested at the real call sites, not through a helper.
"""
import time

from model_mesh.router import Router, RouterConfig

from tests.test_router import index  # noqa: F401

HANG = (599, {"error": "timeout"})


def _router(index, grants, **cfg):
    def transport(url, body, headers, timeout):
        grants.append(timeout)
        return HANG
    return Router(index, "http://up/v1", "k", RouterConfig(**cfg),
                  transport=transport)


def test_explicit_zero_timeout_is_not_promoted_to_the_full_request_timeout(index):
    index.ensure_model("m-a")
    grants: list[float] = []
    r = _router(index, grants)

    payload, att = r.dial("m-a", {"messages": []}, "retain",
                          source="request", timeout=0.0)

    assert grants == [], (
        f"a zero grant reached the transport as {grants}; `timeout or default` "
        f"turns 0.0 into the full request timeout"
    )
    assert payload is None and att.status == "skipped-budget", vars(att)


def test_negative_timeout_never_reaches_the_transport(index):
    index.ensure_model("m-a")
    grants: list[float] = []
    r = _router(index, grants)

    r.dial("m-a", {"messages": []}, "retain", source="request", timeout=-5.0)

    assert grants == [], f"negative grant dialled: {grants}"


def test_a_refused_grant_writes_no_sample(index):
    """Running out of budget is the cascade's fault, not the model's. Same
    contract as the starved-dial guard: no evidence recorded against it."""
    index.ensure_model("m-a")
    r = _router(index, [])

    r.dial("m-a", {"messages": []}, "retain", source="request", timeout=0.0)

    assert index.score("m-a", "retain") is None, (
        "a refused zero-grant dial was recorded as a sample against the model"
    )


def test_probe_verdict_zero_timeout_is_busy_not_a_full_dial(index):
    """probe_verdict had the same `timeout or default`. A refused grant must
    classify as `busy` (retry next pass), never as a capability verdict."""
    index.ensure_model("m-a")
    grants: list[float] = []
    r = _router(index, grants)

    verdict, _ = r.probe_verdict("m-a", "retain",
                                 [{"role": "user", "content": "p"}], timeout=0.0)

    assert grants == [], f"probe dialled with {grants}"
    assert verdict == "busy", verdict


def test_none_timeout_still_means_the_op_class_default(index):
    """The fix must not break the default path: None keeps meaning
    'use the configured timeout'."""
    index.ensure_model("m-a")
    grants: list[float] = []
    r = _router(index, grants)

    r.dial("m-a", {"messages": []}, "retain", source="request", timeout=None)

    assert grants == [r.request_timeout("retain")], grants


def test_reprobe_probe_uses_the_grant_its_guard_checked(index):
    """Time passing between the re-probe guard's read and the probe call must
    not change the grant. Before the fix this test reproduced -9930.0s.

    The clock jumps inside probe_timeout(), which the call evaluates AFTER the
    guard has compared probe_left -- exactly the gap under test. With the fix,
    the grant is min(probe_timeout, probe_left), both fixed before the jump.
    """
    models = [f"m-{i}" for i in range(3)]
    for m in models:
        index.ensure_model(m)
    grants: list[tuple[str, float]] = []

    def transport(url, body, headers, timeout):
        kind = "probe" if body.get("max_tokens") == 4096 else "request"
        grants.append((kind, timeout))
        return HANG

    r = Router(index, "http://up/v1", "k",
               RouterConfig(total_budget_s=280.0, request_timeout_s=135.0,
                            max_attempts=1),
               transport=transport)

    real_monotonic = time.monotonic
    offset = {"t": 0.0}
    real_pt = r.probe_timeout

    def probe_timeout_after_a_stall(op_class):
        offset["t"] += 10_000.0
        return real_pt(op_class)

    r.probe_timeout = probe_timeout_after_a_stall
    time.monotonic = lambda: real_monotonic() + offset["t"]
    try:
        r.route(models, {"messages": []}, "retain",
                probe_messages=[{"role": "user", "content": "p"}])
    finally:
        time.monotonic = real_monotonic

    probes = [g for k, g in grants if k == "probe"]
    assert probes, f"the re-probe arm never probed, test is vacuous: {grants}"
    floor = r.cfg.min_useful_dial_s
    assert all(g >= floor for g in probes), (
        f"re-probe probe granted {probes}; it re-read the clock instead of "
        f"using the probe_left its guard compared against {floor}s"
    )


def test_probe_uses_its_op_class_probe_timeout(index):
    """The per-op_class PROBE budget reaches the transport.

    Nothing pinned this before 2026-10-05: collapsing probe_timeout() to the
    scalar 45s left the suite green. It is load-bearing -- 5527f20 sized the
    reflect/consolidation probes to ~32k chars on the strength of their 100s
    probe budget, and a 45s budget would time those out on a normal day.
    """
    from model_mesh.config import DEFAULTS
    index.ensure_model("m-a")
    grants: list[float] = []
    r = _router(index, grants, **DEFAULTS["router"])
    by_oc = DEFAULTS["router"]["probe_timeout_s_by_op_class"]
    assert by_oc, "no per-op_class probe override configured; test is vacuous"
    op_class, want = next(iter(by_oc.items()))
    assert want != DEFAULTS["router"]["probe_timeout_s"], "override equals scalar"

    r.probe_verdict("m-a", op_class, [{"role": "user", "content": "p"}])

    assert grants == [want], (
        f"{op_class} probe dialled with {grants}, want its own {want}s budget"
    )
