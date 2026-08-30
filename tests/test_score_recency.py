"""Recency-capped scoring: newest SCORE_RECENT_N samples decide, not the
24h average.

RED arm (pre-fix): a model that failed 100x this morning and recovered ten
minutes ago still scored ~0 success for hours — the 24h window buried the
recovery. NIM overload flips in minutes; evidence must track that.
"""
import time

from model_mesh.index import SCORE_RECENT_N, Index


def _fill(index, model, op, status, n, t0):
    for i in range(n):
        # record() stamps now; write directly for controlled timestamps.
        with index._lock:
            index._conn.execute(
                "INSERT INTO samples (model_id, op_class, source, status,"
                " latency_ms, payload_chars, ts) VALUES (?,?,?,?,?,?,?)",
                (model, op, "test", status, 500.0, 10, t0 + i),
            )
            index._conn.commit()


def test_recovery_visible_within_recent_n(tmp_path):
    """100 morning failures + SCORE_RECENT_N fresh successes => healthy."""
    index = Index(tmp_path / "mesh.db")
    index.ensure_model("m")
    now = time.time()
    _fill(index, "m", "retain", "http-598", 100, now - 7200)   # morning wreck
    _fill(index, "m", "retain", "ok", SCORE_RECENT_N, now - 60)  # recovered
    s = index.score("m", "retain")
    assert s is not None
    assert s.success_rate == 1.0  # newest N are all ok; morning is history
    assert s.n == SCORE_RECENT_N


def test_collapse_visible_within_recent_n(tmp_path):
    """Symmetric arm: a good day does not hide a fresh collapse."""
    index = Index(tmp_path / "mesh.db")
    index.ensure_model("m")
    now = time.time()
    _fill(index, "m", "retain", "ok", 100, now - 7200)          # good morning
    _fill(index, "m", "retain", "http-429", SCORE_RECENT_N, now - 60)
    s = index.score("m", "retain")
    assert s is not None
    assert s.success_rate == 0.0


def test_staleness_bound_still_applies(tmp_path):
    """Samples older than the window score None regardless of count."""
    index = Index(tmp_path / "mesh.db")
    index.ensure_model("m")
    _fill(index, "m", "retain", "ok", 50, time.time() - 200000)  # >24h old
    assert index.score("m", "retain") is None
