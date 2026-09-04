"""Provider-wide 429 pause: NIM throttles the shared API key, not one model.

RED arms (pre-fix behavior): a 429 on model A used to leave the cascade free
to immediately dial B, C, D — machine-gunning the shared key through the
throttle window and polluting every sibling's samples with self-inflicted
429s. GREEN arms: pause expires, dialing resumes; Retry-After is honored,
capped, and never shortened.
"""
import time

import pytest

from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig


BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}


def _mk(tmp_path, transport, **cfg_over):
    index = Index(tmp_path / "mesh.db")
    for m in ("m-a", "m-b"):
        index.ensure_model(m)
    # These tests are about the provider-pause predicate (pause window vs this
    # attempt's budget), so the budget must be the one the test states. Every
    # dial below uses op_class "retain", which carries its own 135s override in
    # the real defaults (added 2026-09-03 alongside consolidation/reflect); left
    # in place it silently overrides each test's request_timeout_s and the
    # "window >> budget" arms stop testing the skip branch. Clear the per-
    # op_class map here, once, so the scalar governs — a caller that actually
    # wants an override can still pass one.
    cfg_over.setdefault("request_timeout_s_by_op_class", {})
    cfg = RouterConfig(**cfg_over)
    r = Router(index, "http://up", "k", cfg=cfg, transport=transport)
    return index, r


def test_429_arms_provider_pause(tmp_path):
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(body["model"])
        return 429, {"error": "throttled", "_retry_after_s": 30.0}

    # The scalar budget governs (see _mk: retain's own override is cleared).
    _, r = _mk(tmp_path, transport, request_timeout_s=10.0)
    r.dial("m-a", BODY, "retain", "request")
    assert r._provider_pause_until > time.time() + 25.0
    # Second dial inside the window, window >> budget: skipped WITHOUT an
    # upstream call (a 10s budget cannot survive a 30s pause).
    _, att = r.dial("m-b", BODY, "retain", "request")
    assert att.status == "skipped-provider-pause"
    assert calls == ["m-a"]


def test_429_without_retry_after_uses_default(tmp_path):
    def transport(url, body, headers, timeout):
        return 429, {"error": "throttled"}

    _, r = _mk(tmp_path, transport, provider_pause_default_s=5.0)
    t0 = time.time()
    r.dial("m-a", BODY, "retain", "request")
    assert 3.0 < r._provider_pause_until - t0 <= 5.5


def test_retry_after_capped(tmp_path):
    def transport(url, body, headers, timeout):
        return 429, {"error": "throttled", "_retry_after_s": 86400.0}

    _, r = _mk(tmp_path, transport, provider_pause_max_s=60.0)
    t0 = time.time()
    r.dial("m-a", BODY, "retain", "request")
    assert r._provider_pause_until - t0 <= 61.0


def test_pause_never_shortened(tmp_path):
    """Concurrent-arm race: while a dial is in flight, another thread's 429
    arms a LARGE window; this dial's smaller Retry-After lands after it and
    must not shrink the armed window (max-of-windows)."""
    holder = {}

    def transport(url, body, headers, timeout):
        # Simulate the sibling thread arming a large window mid-flight.
        holder["big"] = time.time() + 55.0
        holder["router"]._provider_pause_until = holder["big"]
        return 429, {"error": "throttled", "_retry_after_s": 1.0}

    _, r = _mk(tmp_path, transport)
    holder["router"] = r
    r.dial("m-a", BODY, "retain", "request")
    assert r._provider_pause_until == holder["big"]


def test_short_pause_waits_then_dials(tmp_path):
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(body["model"])
        return 200, {"choices": [{"message": {"content": '{"facts":["x"]}'}}]}

    _, r = _mk(tmp_path, transport)
    r._provider_pause_until = time.time() + 0.2
    t0 = time.monotonic()
    payload, att = r.dial("m-a", BODY, "retain", "request", timeout=30.0)
    assert att.status == "ok"
    assert time.monotonic() - t0 >= 0.15  # actually waited out the window
    assert calls == ["m-a"]


def test_skipped_pause_records_no_sample(tmp_path):
    """A skip is a routing decision, not evidence about the model."""
    def transport(url, body, headers, timeout):
        return 429, {"error": "throttled", "_retry_after_s": 30.0}

    index, r = _mk(tmp_path, transport, request_timeout_s=10.0)
    r.dial("m-a", BODY, "retain", "request")          # arms pause, records 429
    r.dial("m-b", BODY, "retain", "request")          # skipped (30s > 10s budget)
    assert index.score("m-b", "retain") is None       # no sample for m-b
