"""Failure-inclusive p95 (regression: dead-slow alternator, 2026-09-13).

A model alternating 61s successes and 135s timeouts scored p95_ms ~61s and
passed the latency eligibility floor while every timeout ate the whole
per-attempt budget. score() must expose what its failed attempts cost.
"""
from model_mesh.index import Index, OK


def _alternating(ix, mid="m", oc="retain"):
    for _ in range(4):
        ix.record(mid, oc, "request", OK, 61_000.0, 100)
        ix.record(mid, oc, "request", "http-598", 135_000.0, 0)


def _seed(ix, mid="m", oc="retain"):
    ix.ensure_model(mid)
    return ix


def test_p95_all_includes_timeout_latency(tmp_path):
    ix = _seed(Index(tmp_path / "mesh.db"))
    _alternating(ix)
    s = ix.score("m", "retain")
    assert s is not None
    # successes-only view unchanged by this field's addition...
    assert s.p95_ms < 70_000
    # ...but the failure-inclusive view sees the 135s timeouts
    assert s.p95_all_ms > 120_000


def test_p95_all_equals_p95_when_all_ok(tmp_path):
    ix = _seed(Index(tmp_path / "mesh.db"))
    for _ in range(6):
        ix.record("m", "retain", "request", OK, 5_000.0, 100)
    s = ix.score("m", "retain")
    assert s.p95_all_ms == s.p95_ms == 5_000.0


def test_all_fail_window_keeps_real_failure_latency(tmp_path):
    ix = _seed(Index(tmp_path / "mesh.db"))
    for _ in range(4):
        ix.record("m", "retain", "request", "http-598", 90_000.0, 0)
    s = ix.score("m", "retain")
    assert s.success_rate == 0.0
    assert s.p95_ms == 30_000.0      # existing serialization-ceiling convention
    assert s.p95_all_ms == 90_000.0  # real cost survives in the new field
