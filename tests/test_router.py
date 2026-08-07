"""Behavior tests for the model-mesh router + index.

Every test here encodes a lesson from the nim-proxy postmortem (2026-08-03):
EOL models must be detected at request time, fallbacks must be real,
breakers must open AND close, and the re-probe arm must rescue a total miss.
Transport is injected; no network.
"""

from __future__ import annotations

import time

import pytest

from model_mesh.index import Index, OK
from model_mesh.router import Attempt, Router, RouterConfig


class FakeTransport:
    """Scriptable upstream: map model_id -> list of (status, payload) popped
    per call; missing/exhausted -> 200 OK."""

    def __init__(self, script: dict[str, list[tuple[int, dict]]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[str] = []

    def __call__(self, url, body, headers, timeout):
        model = body["model"]
        self.calls.append(model)
        seq = self.script.get(model)
        if seq:
            return seq.pop(0)
        return 200, {"choices": [{"message": {"content": '{"facts": ["x"]}'}}],
                     "model": model}


@pytest.fixture()
def index(tmp_path):
    return Index(tmp_path / "mesh.db")


def make_router(index, script=None, **cfg_kw):
    t = FakeTransport(script)
    cfg = RouterConfig(**cfg_kw) if cfg_kw else RouterConfig()
    r = Router(index, "http://up.example/v1", "k", cfg, transport=t)
    return r, t


PROBE = [{"role": "user", "content": "say OK"}]


# --- catalog / EOL ----------------------------------------------------------

def test_catalog_sync_marks_new_and_eol(index):
    r1 = index.sync_catalog("nim", {"a", "b"})
    assert sorted(r1["new"]) == ["a", "b"]
    r2 = index.sync_catalog("nim", {"b", "c"})
    assert r2["new"] == ["c"] and r2["eol"] == ["a"]
    assert sorted(index.live_models("nim")) == ["b", "c"]
    # EOL is data: 'a' still exists in the models table
    assert index.breaker_get("a")["state"] == "gone"


def test_eol_model_can_return(index):
    index.sync_catalog("nim", {"a"})
    index.sync_catalog("nim", set())          # a EOLs
    r = index.sync_catalog("nim", {"a"})      # provider un-deprecates
    assert r["returned"] == ["a"]
    assert index.live_models("nim") == ["a"]
    assert index.breaker_get("a")["state"] == "healthy"


def test_request_time_410_marks_gone_immediately(index):
    """The maverick lesson: EOL detection must not wait for discovery."""
    index.sync_catalog("nim", {"dead", "alive"})
    router, t = make_router(index, {"dead": [(410, {"detail": "EOL"})]})
    res = router.route(["dead", "alive"], {"messages": []}, "retain")
    assert res.ok and res.model_id == "alive"
    assert index.breaker_get("dead")["state"] == "gone"
    assert "dead" not in index.live_models("nim")
    # and it is never tried again
    res2 = router.route(["dead", "alive"], {"messages": []}, "retain")
    assert res2.attempts[0].model_id == "alive"


# --- cascade ----------------------------------------------------------------

def test_cascade_on_transient(index):
    router, t = make_router(index, {"m1": [(429, {})], "m2": [(503, {})]})
    res = router.route(["m1", "m2", "m3"], {"messages": []}, "retain")
    assert res.ok and res.model_id == "m3"
    assert [a.model_id for a in res.attempts] == ["m1", "m2", "m3"]


def test_auth_never_poisons_breaker(index):
    router, t = make_router(index, {"m1": [(401, {})]})
    res = router.route(["m1", "m2"], {"messages": []}, "retain")
    assert res.ok and res.model_id == "m2"
    b = index.breaker_get("m1")
    assert b["state"] == "auth" and b["consec_fails"] == 0


def test_malformed_upstream_body_is_transient(index):
    router, t = make_router(index, {"m1": [(599, {"error": "malformed"})]})
    res = router.route(["m1", "m2"], {"messages": []}, "retain")
    assert res.ok and res.model_id == "m2"


# --- breaker lifecycle --------------------------------------------------------

def test_breaker_opens_after_threshold(index):
    fails = [(500, {})] * 3
    router, t = make_router(index, {"m1": list(fails)}, breaker_threshold=3)
    for _ in range(3):
        router.route(["m1"], {"messages": []}, "retain")
    assert index.breaker_get("m1")["state"] == "down"
    # while down and cooling, m1 is not even attempted
    res = router.route(["m1", "m2"], {"messages": []}, "retain")
    assert res.attempts[0].model_id == "m2"


def test_breaker_recovers_and_closes(index):
    router, t = make_router(index, {"m1": [(500, {})] * 3},
                            breaker_threshold=3, breaker_cooldown_s=0.01)
    for _ in range(3):
        router.route(["m1"], {"messages": []}, "retain")
    assert index.breaker_get("m1")["state"] == "down"
    time.sleep(0.02)
    res = router.route(["m1"], {"messages": []}, "retain")   # recovery request
    assert res.ok
    assert index.breaker_get("m1")["state"] == "healthy"


def test_breaker_reopen_doubles_cooldown(index):
    router, t = make_router(index, {"m1": [(500, {})] * 4},
                            breaker_threshold=3, breaker_cooldown_s=0.01)
    for _ in range(3):
        router.route(["m1"], {"messages": []}, "retain")
    time.sleep(0.02)
    router.route(["m1"], {"messages": []}, "retain")   # recovery fails
    b = index.breaker_get("m1")
    assert b["state"] == "down" and b["cooldown_s"] >= 0.02


# --- ranking ------------------------------------------------------------------

def test_ranking_prefers_stable_model(index):
    # m_fast: consistently 100ms. m_spiky: median low but wild spikes.
    for _ in range(20):
        index.record("m_fast", "retain", "request", OK, 100.0)
    for i in range(20):
        index.record("m_spiky", "retain", "request", OK,
                      80.0 if i % 2 else 9000.0)
    router, t = make_router(index)
    order = router.ranked(["m_spiky", "m_fast"], "retain")
    assert order[0] == "m_fast"


def test_unknown_models_rank_after_scored_but_stay(index):
    for _ in range(5):
        index.record("known", "retain", "request", OK, 100.0)
    router, t = make_router(index)
    order = router.ranked(["newcomer", "known"], "retain")
    assert order == ["known", "newcomer"]


# --- the re-probe arm -----------------------------------------------------------

def test_total_miss_reprobe_rescues(index):
    """All ranked candidates fail; a live re-probe finds m3 healthy and the
    request succeeds instead of 503ing. This is the arm Andrew asked for."""
    script = {
        "m1": [(500, {})] * 4,   # keep failing through the re-probe too
        "m2": [(500, {})] * 4,
        # m3's first call is the PROBE (succeeds), second is the real request
    }
    router, t = make_router(index, script, max_attempts=2)
    res = router.route(["m1", "m2", "m3"], {"messages": []}, "retain",
                       probe_messages=PROBE)
    assert res.ok and res.model_id == "m3" and res.reprobed


def test_total_miss_without_probe_messages_fails_with_evidence(index):
    script = {"m1": [(500, {})], "m2": [(502, {})]}
    router, t = make_router(index, script, max_attempts=2)
    res = router.route(["m1", "m2"], {"messages": []}, "retain")
    assert not res.ok
    assert {a.model_id for a in res.attempts} == {"m1", "m2"}
    assert all(a.status.startswith("http-") for a in res.attempts)


# --- telemetry -------------------------------------------------------------------

def test_real_traffic_lands_in_index(index):
    router, t = make_router(index)
    router.route(["m1"], {"messages": [{"role": "user", "content": "hi"}]},
                 "retain")
    s = index.score("m1", "retain")
    assert s is not None and s.n == 1 and s.success_rate == 1.0


# --- regressions from the 2026-08-04 live recheck ---------------------------

def test_intermittent_model_skipped_by_success_floor(index):
    """The consecutive-fail breaker CANNOT catch ok/fail alternating: live,
    gpt-oss-20b timed out 6 of 8 retain calls at 120s each without ever hitting
    3 in a row, stayed top-ranked, and every retain paid the timeout first."""
    for i in range(8):
        index.record("flaky", "retain", "request",
                     OK if i % 4 == 0 else "timeout",
                     100.0 if i % 4 == 0 else None)
    router, t = make_router(index)
    assert index.breaker_get("flaky")["state"] == "healthy"   # breaker blind
    assert router.ranked(["flaky", "good"], "retain") == ["good"]  # floor catches


def test_success_floor_is_per_op_class(index):
    """nemotron-49b: 43% on retain (http-400) but 100% on consolidation. It must
    stay eligible for the op_class it actually serves."""
    for i in range(8):
        index.record("m", "retain", "request",
                     OK if i < 3 else "http-400", 200.0)
    for _ in range(5):
        index.record("m", "consolidation", "request", OK, 500.0)
    router, t = make_router(index)
    assert router.ranked(["m"], "retain") == []
    assert router.ranked(["m"], "consolidation") == ["m"]


def test_floor_needs_min_samples(index):
    """One bad sample must not exile a model — that's what the breaker is for."""
    index.record("new", "retain", "request", "timeout", None)
    router, t = make_router(index)
    assert router.ranked(["new"], "retain") == ["new"]


# --- latency floor -----------------------------------------------------------

def test_latency_floor_excludes_a_model_that_eats_the_cascade_budget(index):
    """Success rate alone is not enough. Live 2026-08-07: llama-3.3-70b sat at
    61% success (ABOVE the 0.5 floor) with p95 96.9s and stayed ranked #2 on
    retain, so two attempts blew total_budget_s=240 and the op wedged in
    'processing' until the watchdog reset it. Ranking already knew it was bad
    (score 51.9 vs 66.4); eligibility did not look at latency at all."""
    for i in range(10):                     # 70% success, all of them slow
        index.record("slow", "retain", "request",
                     OK if i % 10 < 7 else "timeout", 96_900.0)
    index.record("fast", "retain", "request", OK, 5_000.0)
    router, t = make_router(index)
    s = index.score("slow", "retain")
    assert s.success_rate > 0.5             # passes the success floor
    assert router.ranked(["slow", "fast"], "retain") == ["fast"]


def test_latency_floor_respects_min_samples(index):
    """A single slow sample must not exile a model, same rule as the success
    floor — cold starts and one-off spikes are not a verdict."""
    index.record("cold", "retain", "request", OK, 300_000.0)
    router, t = make_router(index)
    assert router.ranked(["cold"], "retain") == ["cold"]


def test_latency_floor_is_per_op_class(index):
    """Consolidation legitimately runs longer than retain. A model that is slow
    on one op_class must stay eligible for the other, like the success floor."""
    for _ in range(6):
        index.record("m", "retain", "request", OK, 96_900.0)     # too slow
        index.record("m", "consolidation", "request", OK, 40_000.0)  # fine
    router, t = make_router(index)
    assert router.ranked(["m"], "retain") == []
    assert router.ranked(["m"], "consolidation") == ["m"]


def test_latency_floor_leaves_room_for_a_second_attempt(index):
    """The default must let an attempt run TWICE inside the cascade budget,
    otherwise one slow candidate still consumes the whole failover."""
    cfg = RouterConfig()
    assert 2 * (cfg.max_p95_ms_for_eligibility / 1000.0) < cfg.total_budget_s


def test_http_400_not_retried_and_not_breaker_counted(index):
    router, t = make_router(index, {"m1": [(400, {"detail": "bad payload"})]})
    res = router.route(["m1", "m2"], {"messages": []}, "retain")
    assert res.ok and res.model_id == "m2"
    b = index.breaker_get("m1")
    assert b["state"] == "healthy" and b["consec_fails"] == 0   # not transient
    assert "not retryable" in res.attempts[0].detail


def test_cascade_respects_total_budget(index):
    """3 x 120s = 360s overran hindsight's own 300s retain timeout, so the
    client abandoned mid-cascade and failover never completed."""
    def slow_transport(url, body, headers, timeout):
        time.sleep(0.05)
        return 500, {}
    r = Router(index, "http://up/v1", "k",
               RouterConfig(max_attempts=5, total_budget_s=0.12),
               transport=slow_transport)
    res = r.route(["m1", "m2", "m3", "m4", "m5"], {"messages": []}, "retain")
    assert not res.ok
    assert any(a.status == "skipped-budget" for a in res.attempts)
    assert len(res.attempts) < 5          # stopped early, did not burn all 5


def test_per_attempt_timeout_shrinks_to_remaining_budget(index):
    seen = []

    def capture(url, body, headers, timeout):
        seen.append(timeout)
        return 500, {}
    r = Router(index, "http://up/v1", "k",
               RouterConfig(max_attempts=3, request_timeout_s=120.0,
                            total_budget_s=10.0),
               transport=capture)
    r.route(["m1", "m2", "m3"], {"messages": []}, "retain")
    assert seen and all(t <= 10.0 for t in seen)


# --- credential-loss recovery (brick-class bug) ------------------------------

def test_keyless_start_does_not_permanently_brick_every_model(index):
    """auth used to be TERMINAL, and breaker state persists in SQLite. One
    daemon start with a missing key (launchctl setenv does not survive a machine
    restart) marked every model 'auth' forever: ranked() returned [] even after
    a good key was restored, and restarting never healed it."""
    keyless = Router(index, "http://up/v1", "",
                     RouterConfig(auth_cooldown_s=0.01),
                     transport=lambda u, b, h, t: (401, {"error": "unauthorized"}))
    keyless.route(["m1", "m2", "m3"], {"messages": []}, "retain")
    assert all(index.breaker_get(m)["state"] == "auth" for m in ("m1", "m2", "m3"))

    time.sleep(0.02)                       # auth cooldown expires
    fixed, _ = make_router(index)          # same DB, working transport
    assert fixed.ranked(["m1", "m2", "m3"], "retain") != []
    assert fixed.route(["m1"], {"messages": []}, "retain").ok


def test_auth_is_skipped_while_cooling(index):
    r = Router(index, "http://up/v1", "", RouterConfig(auth_cooldown_s=300.0),
               transport=lambda u, b, h, t: (401, {}))
    r.route(["m1"], {"messages": []}, "retain")
    assert r.ranked(["m1"], "retain") == []      # not retried immediately


def test_reprobe_arm_reconsiders_auth_models(index):
    """A restored credential is exactly the stale-state case the re-probe arm
    exists for, so 'auth' must not be excluded there."""
    index.breaker_set("m1", state="auth", cooldown_until=time.time() + 9999)
    r, t = make_router(index)
    res = r.route(["m1"], {"messages": []}, "retain", probe_messages=PROBE)
    assert res.ok and res.reprobed


def test_api_key_read_at_call_time(index):
    """Cached-at-import key ignored a credential fixed after startup."""
    box = {"k": ""}
    seen = []

    def capture(url, body, headers, timeout):
        seen.append(headers["Authorization"])
        return 200, {"choices": [{"message": {"content": '{"facts":["x"]}'}}]}
    r = Router(index, "http://up/v1", lambda: box["k"], RouterConfig(),
               transport=capture)
    r.route(["m1"], {"messages": []}, "retain")
    box["k"] = "LATER-KEY"                       # fixed without a restart
    r.route(["m1"], {"messages": []}, "retain")
    assert seen == ["Bearer ", "Bearer LATER-KEY"]


def test_resolve_api_key_falls_back_to_file(tmp_path, monkeypatch):
    from model_mesh.config import resolve_api_key
    f = tmp_path / ".env"
    f.write_text('OTHER=1\nNVIDIA_API_KEY="file-key"\n')
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert resolve_api_key("NVIDIA_API_KEY", f) == "file-key"
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    assert resolve_api_key("NVIDIA_API_KEY", f) == "env-key"   # env wins
    assert resolve_api_key("MISSING_VAR", f) == ""
