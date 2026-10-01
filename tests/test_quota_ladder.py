"""Per-model 429 quota cooldown ladder — a SEPARATE timer from the breaker.

Adopted from CLIProxyAPI (sdk/cliproxy/auth/conductor_refresh.go constants,
conductor_cooldown.go nextQuotaCooldown / quotaCooldownAfterFailure).

RED arm (pre-ladder behavior): a 429 counted toward breaker_threshold, so
three throttles opened the FAILURE breaker on a model that was only
rate-limited, and there was no per-model memory of the throttle at all.
"""
import time

import pytest

from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig

BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
OK_BODY = {"choices": [{"message": {"content": '{"facts":["x"]}'}}]}


def _mk(tmp_path, transport, **cfg_over):
    index = Index(tmp_path / "mesh.db")
    for m in ("m-a", "m-b"):
        index.ensure_model(m)
    cfg_over.setdefault("request_timeout_s_by_op_class", {})
    # Provider pause is a different mechanism; zero it so these tests see
    # only the per-model ladder.
    cfg_over.setdefault("provider_pause_default_s", 0.0)
    cfg_over.setdefault("provider_pause_max_s", 0.0)
    r = Router(index, "http://up", "k", cfg=RouterConfig(**cfg_over),
               transport=transport)
    return index, r


def _expire(r, model_id):
    r._quota[model_id]["until"] = time.time() - 0.01


def test_429_does_not_count_toward_breaker(tmp_path):
    index, r = _mk(tmp_path, lambda *a: (429, {"error": "throttled"}))
    for _ in range(5):
        r.dial("m-a", BODY, "retain", "request")
        _expire(r, "m-a")
    b = index.breaker_get("m-a")
    assert b["consec_fails"] == 0 and b["state"] != "down"


def test_ladder_doubles_and_caps(tmp_path):
    _, r = _mk(tmp_path, lambda *a: (429, {}),
               quota_backoff_base_s=1.0, quota_backoff_max_s=8.0)
    windows = []
    for _ in range(6):
        r.dial("m-a", BODY, "retain", "request")
        windows.append(round(r.quota_left("m-a")))
        _expire(r, "m-a")
    assert windows == [1, 2, 4, 8, 8, 8]
    assert r._quota["m-a"]["level"] == 3  # stops climbing at the cap


def test_burst_inside_window_does_not_climb(tmp_path):
    _, r = _mk(tmp_path, None)
    r._on_quota_hit("m-a", None)
    first = r._quota["m-a"]["until"]
    for _ in range(5):
        r._on_quota_hit("m-a", None)
    assert r._quota["m-a"]["level"] == 1
    assert r._quota["m-a"]["until"] == first


def test_retry_after_floored_and_level_unchanged(tmp_path):
    _, r = _mk(tmp_path, lambda *a: (429, {"_retry_after_s": 2.0}),
               quota_cooldown_floor_s=10.0)
    r.dial("m-a", BODY, "retain", "request")
    assert 9.0 < r.quota_left("m-a") <= 10.0
    assert r._quota["m-a"]["level"] == 0


def test_window_never_shortened(tmp_path):
    _, r = _mk(tmp_path, None, quota_cooldown_floor_s=1.0)
    r._on_quota_hit("m-a", 60.0)
    r._on_quota_hit("m-a", 1.0)
    assert r.quota_left("m-a") > 50.0


def test_cooled_model_skipped_without_call_or_sample(tmp_path):
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(body["model"])
        return 200, OK_BODY

    index, r = _mk(tmp_path, transport)
    r._on_quota_hit("m-a", 30.0)
    assert not r.eligible("m-a", "retain")
    payload, att = r.dial("m-a", BODY, "retain", "request")
    assert payload is None and att.status == "skipped-quota-cooldown"
    assert calls == []
    assert index.score("m-a", "retain") is None


def test_cooldown_is_per_model(tmp_path):
    seq = {"m-a": [(429, {})]}

    def transport(url, body, headers, timeout):
        s = seq.get(body["model"])
        return s.pop(0) if s else (200, OK_BODY)

    _, r = _mk(tmp_path, transport)
    res = r.route(["m-a", "m-b"], BODY, "retain")
    assert res.ok and res.model_id == "m-b"
    assert r.quota_left("m-a") > 0 and r.quota_left("m-b") == 0
    assert r.ranked(["m-a", "m-b"], "retain") == ["m-b"]


def test_success_resets_ladder(tmp_path):
    seq = [(429, {}), (429, {}), (200, OK_BODY), (429, {})]
    _, r = _mk(tmp_path, lambda *a: seq.pop(0))
    for _ in range(3):
        r.dial("m-a", BODY, "retain", "request")
        if "m-a" in r._quota:
            _expire(r, "m-a")
    assert "m-a" not in r._quota
    r.dial("m-a", BODY, "retain", "request")
    assert r._quota["m-a"]["level"] == 1   # back to the first rung


def test_ladder_and_breaker_are_independent_timers(tmp_path):
    """A 5xx run opens the breaker; a 429 arms only the quota timer. Clearing
    one must not touch the other."""
    seq = [(503, {})] * 3 + [(429, {})]
    index, r = _mk(tmp_path, lambda *a: seq.pop(0), breaker_cooldown_s=0.01)
    for _ in range(3):
        r.dial("m-a", BODY, "retain", "request")
    assert index.breaker_get("m-a")["state"] == "down"
    assert r.quota_left("m-a") == 0
    time.sleep(0.02)
    r.dial("m-a", BODY, "retain", "request")       # recovering -> 429
    b = index.breaker_get("m-a")
    assert b["state"] == "recovering" and b["consec_fails"] == 3
    assert r.quota_left("m-a") > 0


def test_quota_all_snapshot(tmp_path):
    _, r = _mk(tmp_path, None)
    r._on_quota_hit("m-a", None)
    snap = r.quota_all()
    assert snap["m-a"]["level"] == 1 and snap["m-a"]["cooldown_left_s"] > 0


def test_defaults_match_cliproxyapi():
    cfg = RouterConfig()
    assert (cfg.quota_backoff_base_s, cfg.quota_backoff_max_s,
            cfg.quota_cooldown_floor_s) == (1.0, 1800.0, 10.0)
