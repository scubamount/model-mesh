"""Verify stale-evidence refresh, budget ordering, and the discovery margin rule.

Converted from check()-style to real pytest (old suite was invisible to pytest —
same disease fixed in test_probe_economy.py). Every assertion below is one of the
original `checks.append` items, now a named test with a real red/green failure.

Bugs this suite catches (proven red-on-sabotage before going green):
- Stale evidence read as fresh → model never probed again (the 2026-08-20
  auto/evolve outage where 0 scored models meant a full day of missed probes).
- discovery.py's dormant gate consuming evidence older than `score()` accepts
  → models expire faster than their re-probe window.
- Budget + alphabetical iteration → oldest-evidence model deferred forever.
- REFRESH_MARGIN == 1.0 → race returns silently, symptoms read as a provider
  outage rather than a discovery defect.
"""

from __future__ import annotations

import os
import tempfile
import time

from model_mesh.discovery import discover, REFRESH_MARGIN
from model_mesh.index import Index
from tests.test_stale_evidence_reprobe_fixtures import (
    PROVIDER,
    FRESH,
    STALE,
    NEVER,
    ALIASES,
    FakeRouter,
    build_index,
)


# ---- Regression: stale evidence must be re-probed ----

def test_stale_evidence_model_is_repolled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        router = FakeRouter()

        import model_mesh.discovery as disc

        orig = disc.fetch_catalog
        try:
            disc.fetch_catalog = lambda base, key, timeout=30.0: {FRESH, STALE, NEVER}
            discover(
                idx, router, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig

        probed = {m for m, _ in router.probed}
        assert STALE in probed, "stale-evidence model must be re-probed"


def test_never_sampled_model_is_probed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        router = FakeRouter()

        import model_mesh.discovery as disc

        orig = disc.fetch_catalog
        try:
            disc.fetch_catalog = lambda base, key, timeout=30.0: {FRESH, STALE, NEVER}
            discover(
                idx, router, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig

        probed = {m for m, _ in router.probed}
        assert NEVER in probed, "never-sampled model must be probed"


def test_fresh_evidence_model_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        router = FakeRouter()

        import model_mesh.discovery as disc

        orig = disc.fetch_catalog
        try:
            disc.fetch_catalog = lambda base, key, timeout=30.0: {FRESH, STALE, NEVER}
            discover(
                idx, router, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig

        probed = {m for m, _ in router.probed}
        assert FRESH not in probed, "fresh-evidence model must NOT be re-probed"


# ---- Boundary: about-to-expire evidence must be refreshed ----

def test_boundary_fixture_still_scores_today() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(os.path.join(tmp, "boundary.db"))
        near = "boundary/expires-in-minutes"
        idx.sync_catalog(PROVIDER, {near})
        idx.record(near, "retain", "probe", "ok", 3000.0)
        now = time.time()
        idx._conn.execute(
            "UPDATE samples SET ts=? WHERE model_id=?",
            (now - (3600 * 24 * 0.99), near),
        )
        idx._conn.commit()
        assert idx.score(near, "retain") is not None, "boundary model must still score now"


def test_about_to_expire_model_is_repolled_before_expiry(tmp_path) -> None:
    """99% through the window: still scoreable now, gone long before the next
    daily pass. Proves the refresh happens with margin, not just for obviously-
    stale models.

    NOTE: This test deliberately builds a fresh Index rather than depending on
    a shared `discovery_catalog_one_model` fixture, because `discover()` mutates
    the Index pass-by-reference and `discover` is the call under test. Fixtures
    that build an Index and mutate it in setup are harmful once multiple tests
    run against the same module state — see test_stale_evidence_reprobe_fixtures
    for why this code path never uses pytest fixtures.
    """
    import model_mesh.discovery as disc

    idx = Index(tmp_path / "boundary-discover.db")
    near = "boundary/expires-in-minutes"
    idx.sync_catalog(PROVIDER, {near})
    idx.record(near, "retain", "probe", "ok", 3000.0)
    now = time.time()
    idx._conn.execute(
        "UPDATE samples SET ts=? WHERE model_id=?",
        (now - (3600 * 24 * 0.99), near),
    )
    idx._conn.commit()

    router = FakeRouter()
    orig = disc.fetch_catalog
    try:
        disc.fetch_catalog = lambda base, key, timeout=30.0: {near}
        discover(
            idx, router, PROVIDER, "http://x", "k", ALIASES,
            probe_new=True, log=lambda *a, **k: None,
        )
    finally:
        disc.fetch_catalog = orig

    probed = {m for m, _ in router.probed}
    assert near in probed, "about-to-expire model must be re-probed before it expires"


def test_refresh_margin_is_strictly_inside_window() -> None:
    assert 0.0 < REFRESH_MARGIN < 1.0, "REFRESH_MARGIN must be strictly inside the scoring window"


# ---- Fixture validation: proves both directions of the comparison ----

def test_stale_model_scores_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        assert idx.score(STALE, "retain") is None, (
            "stale model must score None — proves the fixture reproduces the bug"
        )


def test_fresh_model_scores_non_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        assert idx.score(FRESH, "retain") is not None, (
            "fresh model must score non-None — proves the fixture is not vacuous"
        )


# ---- Budget ordering: oldest evidence wins the single probe ----

def test_budget_ordering_prefers_oldest_not_first_alpha() -> None:
    """With a budget smaller than the work, the STALEST model must be probed —
    not the alphabetically-first one. Otherwise sorted-past-budget models stay
    permanently unknown for their op_class."""
    import model_mesh.discovery as disc

    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(os.path.join(tmp, "ord.db"))
        early, late = "aaa/alphabetically-first", "zzz/oldest-evidence"
        idx.sync_catalog(PROVIDER, {early, late})
        now = time.time()
        idx.record(early, "retain", "probe", "ok", 1000.0)
        idx.record(late, "retain", "probe", "ok", 1000.0)
        idx._conn.execute(
            "UPDATE samples SET ts=? WHERE model_id=?",
            (now - (3600 * 24 + 3600), early),
        )
        idx._conn.execute(
            "UPDATE samples SET ts=? WHERE model_id=?",
            (now - (3600 * 24 + 100000), late),
        )
        idx._conn.commit()

        router = FakeRouter()
        orig = disc.fetch_catalog
        try:
            disc.fetch_catalog = lambda base, key, timeout=30.0: {early, late}
            disc.discover(
                idx, router, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, max_probes=1, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig

        probed = {m for m, _ in router.probed}
        assert late in probed, "oldest-evidence model must be probed under budget"
        assert early not in probed, "fresh-alphabetical model must NOT be probed under budget"
