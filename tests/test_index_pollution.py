"""Index pollution guard: unknown model ids must not create state.

A client can POST any string as `model`. Before this gate, a bogus id got a
breaker row via mark_gone() and samples via record() — so /mesh/status reported
a phantom breaker forever and samples referenced ids no catalog row explained.
Live-verified 2026-08-25: `auto/nonexistent` sat in the live DB's breaker table
with state='gone', plus two orphaned sample rows and an empty-string id.
"""

from __future__ import annotations

import socket
import urllib.error

import pytest

from model_mesh.index import Index
from model_mesh.router import Router, RouterConfig, _is_local_resolver_failure


@pytest.fixture()
def idx(tmp_path):
    return Index(tmp_path / "pollution.db")


def test_mark_gone_refuses_unknown_id(idx):
    """mark_gone on an id the catalog never listed must write NOTHING."""
    idx.mark_gone("client-invented/model", "http-404")
    assert idx.breaker_all() == {}, "phantom breaker row created"
    assert idx.live_models() == []


def test_record_refuses_unknown_id(idx):
    """Samples for never-cataloged ids are orphans; refuse at the writer."""
    idx.record("client-invented/model", "retain", "request", "ok", 100.0)
    with idx._lock:
        n = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert n == 0, "orphan sample row created"


def test_known_but_eol_model_still_records(idx):
    """The weekly EOL recheck needs history: known-but-gone ids MUST record."""
    idx.sync_catalog("nim", {"real/model"})
    idx.mark_gone("real/model", "http-404")
    idx.record("real/model", "retain", "probe", "ok", 100.0)
    s = idx.score("real/model", "retain")
    assert s is not None and s.n == 1, (
        "EOL'd models must keep recording — their recheck probe needs evidence"
    )


def test_empty_string_is_not_a_model(idx):
    """The live DB carried an empty-string id: `(body.get('model') or '')`
    produced it. It must be refused like any other unknown id."""
    idx.record("", "retain", "request", "http-400", 73.0)
    idx.mark_gone("", "http-404")
    with idx._lock:
        n = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert n == 0
    assert idx.breaker_all() == {}


def test_ensure_model_makes_an_id_writable(idx):
    """ensure_model is the explicit test-seeding path; after it, record works."""
    idx.ensure_model("seeded/model")
    idx.record("seeded/model", "retain", "probe", "ok", 100.0)
    s = idx.score("seeded/model", "retain")
    assert s is not None and s.n == 1


def test_production_route_never_creates_phantom_state(idx):
    """End-to-end: routing to an invented id through a real Router leaves the
    index untouched — no breaker row, no sample, no models row."""
    from model_mesh.opclass import probe_messages
    from model_mesh.router import Router, RouterConfig

    def transport(url, body, headers, timeout):
        return 404, {"error": "not found"}

    r = Router(idx, "http://up/v1", "k", RouterConfig(), transport=transport)
    res = r.route(["auto/nonexistent"], {"messages": []}, "retain",
                  probe_messages=probe_messages("retain"))
    assert not res.ok
    assert idx.breaker_all() == {}
    with idx._lock:
        n_samples = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n_models = idx._conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    assert n_samples == 0 and n_models == 0, "phantom state created for invented id"


# --- local-fault isolation (2026-09-12) --------------------------------------
# A launchd-started daemon came up with a wedged resolver and logged 5,202
# `URLError: nodename nor servname provided` in 1-2ms each. Every one was
# recorded as a FAILURE SAMPLE against whichever model was being dialled, which
# tripped breakers and wiped the scores for three of four aliases. DNS worked
# from a shell throughout and a restart cured it, so the fault was this host's,
# not the models'. The index must never carry a verdict about a model the
# request never reached.

def _dns_error() -> urllib.error.URLError:
    return urllib.error.URLError(
        socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")
    )


@pytest.mark.parametrize(
    "exc, want",
    [
        (_dns_error(), True),
        (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), False),
        (TimeoutError("The read operation timed out"), False),
        # Matched by TYPE+errno, never message text: a genuine upstream error
        # whose body happens to say "not known" must stay the model's problem.
        (urllib.error.URLError("upstream said: model not known"), False),
    ],
)
def test_local_resolver_predicate(exc, want):
    assert _is_local_resolver_failure(exc) is want


def _router(tmp_path, transport):
    idx = Index(tmp_path / "localfault.db")
    idx.sync_catalog("nim", {"real/model"})
    return idx, Router(idx, "http://up", "k", cfg=RouterConfig(), transport=transport)


def _samples(idx) -> int:
    with idx._lock:
        return idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


def _transport_raising(exc):
    """A transport that fails the way urlopen does, THROUGH the real classifier.

    The injected transport must not simply raise: in production `_transport`
    IS `_http_post`, which catches urlopen's exception and turns it into a
    (status, payload) pair. A test that raises past that boundary exercises a
    path no real request takes. So call the genuine `_http_post` body with a
    urlopen stub — the classification under test is the one that ships.
    """
    def transport(url, body, headers, budget):
        import urllib.request as _u
        real_urlopen = _u.urlopen
        _u.urlopen = lambda *a, **k: (_ for _ in ()).throw(exc)
        try:
            return Router._http_post(_stub_self, url, body, headers, budget)
        finally:
            _u.urlopen = real_urlopen
    return transport


class _StubSelf:
    """Minimal self for _http_post: it only reads nothing but needs to exist."""


_stub_self = _StubSelf()


def test_dns_failure_records_nothing_against_the_model(tmp_path):
    """The regression: a local DNS fault must leave the model's record clean."""
    idx, router = _router(tmp_path, _transport_raising(_dns_error()))
    router.dial("real/model", {"messages": [{"role": "user", "content": "hi"}]},
                "retain", "request")

    assert _samples(idx) == 0, "local DNS fault recorded as a model failure"
    assert all(b.get("consec_fails", 0) == 0 for b in idx.breaker_all().values()), \
        "local DNS fault counted toward a model's breaker"


def test_real_network_failure_still_records(tmp_path):
    """The other arm: a genuine unreachable upstream IS the model's problem.

    Without this, 'ignore local faults' could be implemented as 'ignore all
    network errors' and the mesh would stop demoting genuinely dead models.
    """
    exc = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    idx, router = _router(tmp_path, _transport_raising(exc))
    router.dial("real/model", {"messages": [{"role": "user", "content": "hi"}]},
                "retain", "request")

    assert _samples(idx) == 1, "genuine network failure must stay recorded"


# --- local route-fault isolation (2026-09-14) ----------------------------------
# Same incident class as the DNS fault above: ENETUNREACH/EHOSTUNREACH/ENETDOWN/
# ECONNUNREACH mean the packet never left this host (sleep, interface flap, VPN
# topology). They hit EVERY candidate identically, so recording them against a
# model manufactures the exact phantom-breaker damage the 2026-09-12 incident
# caused. Extends _is_local_resolver_failure to a network-failure predicate.

import errno as _errno


def _route_error(en=_errno.ENETUNREACH) -> urllib.error.URLError:
    return urllib.error.URLError(OSError(en, "Network is unreachable"))


@pytest.mark.parametrize(
    "exc, want",
    [
        (_route_error(_errno.ENETUNREACH), True),
        (_route_error(_errno.EHOSTUNREACH), True),
        (_route_error(_errno.ENETDOWN), True),
        *([(_route_error(_errno.ECONNUNREACH), True)]
           if getattr(_errno, "ECONNUNREACH", None) is not None
           else []),  # Linux-only errno; getattr like router._LOCAL_ROUTE_ERRNOS
        (_dns_error(), True),
        # Provider-side and genuine-connection failures stay the model's problem:
        (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), False),
        (urllib.error.URLError(ConnectionResetError(54, "Connection reset by peer")), False),
        (TimeoutError("The read operation timed out"), False),
    ],
)
def test_local_network_predicate(exc, want):
    from model_mesh.router import _is_local_network_failure
    assert _is_local_network_failure(exc) is want


def test_enetunreach_records_nothing_against_the_model(tmp_path):
    """Route errnos follow the DNS precedent: no sample, no breaker count."""
    idx, router = _router(tmp_path, _transport_raising(_route_error()))
    router.dial("real/model", {"messages": [{"role": "user", "content": "hi"}]},
                "retain", "request")
    assert _samples(idx) == 0, "local route fault recorded as a model failure"
    assert all(b.get("consec_fails", 0) == 0 for b in idx.breaker_all().values()), \
        "local route fault counted toward a model's breaker"


# --- breaker transitions are observable in the log (2026-09-14) ----------------
# Forensics found ZERO 'breaker'/'opened'/'cooldown' lines in 24,665 lines of
# mesh.log: transitions wrote SQLite only, so "did the breaker trip?" was
# unanswerable from logs by construction. One WARNING per transition.

def test_breaker_transition_logs_a_warning(tmp_path, caplog):
    import logging
    idx = Index(tmp_path / "breakerlog.db")
    idx.sync_catalog("nim", {"m"})
    def t(url, body, headers, budget):
        return 503, {"error": "boom"}
    idx, router = _router(tmp_path, t)
    with caplog.at_level(logging.WARNING, logger="model_mesh.router"):
        for _ in range(3):
            router.dial("m", {"messages": [{"role": "user", "content": "hi"}]},
                        "retain", "request")
    assert any("breaker" in r.message.lower() or "breaker-transition" in r.getMessage().lower()
               for r in caplog.records), "breaker transition logged nothing"
    b = idx.breaker_get("m")
    assert b["state"] == "down", "3 consecutive transients must open the breaker"

