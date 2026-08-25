"""Index pollution guard: unknown model ids must not create state.

A client can POST any string as `model`. Before this gate, a bogus id got a
breaker row via mark_gone() and samples via record() — so /mesh/status reported
a phantom breaker forever and samples referenced ids no catalog row explained.
Live-verified 2026-08-25: `auto/nonexistent` sat in the live DB's breaker table
with state='gone', plus two orphaned sample rows and an empty-string id.
"""

from __future__ import annotations

import pytest

from model_mesh.index import Index


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
