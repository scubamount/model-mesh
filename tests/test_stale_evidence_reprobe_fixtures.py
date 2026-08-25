"""Shared fixtures for the stale-evidence reprobe test suite.

Why a separate module (instead of pytest fixtures interleaved in the test file):
- These values describe the TEST CONTEXT, not just setUp state. PROVIDER,
  ALIASES, FRESH/STALE/NEVER are part of the bug definition — the stale model
  is STALE because its evidence is >24h old and scores None; the boundary model
  is "about to expire" because it is 99% through the scoring window. Naming that
  context lets a future reader recognize the fixture shape in a failed assertion
  without re-deriving it from the test body.

Why NOT pytest fixtures:
- build_index is NOT lifecycle-managed state — each test creates its own
  TemporaryDirectory and passes a brand-new Index to discover(). A pytest fixture
  that hands out a shared Index across tests would be a defect: discover() mutates
  the Index pass-by-reference, so tests would bleed into each other (the lesson
  from the old check()-style tests that discovery.py's own internal
  _last_idx/cache state made tests order-dependent). Always build per test.
"""

from __future__ import annotations

import os
import time

from model_mesh.discovery import SCORE_WINDOW_S
from model_mesh.index import Index

# ---- Test context constants ----

PROVIDER = "nim"

ALIASES = {"auto/retain": {"op_class": "retain", "include": [], "exclude": []}}

# Model IDs used in the stale-evidence reprobe suite. The exact names are
# arbitrary here — they only need to sort distinctly. In the real bug these were
# real provider IDs, but the test is about the relative freshness of evidence,
# not the identity.
FRESH = "fresh/model"
STALE = "stale/model"
NEVER = "never/model"

# ---- Helpers ----


class FakeRouter:
    """Records which (model, op_class) pairs discovery decided to probe.

    Verdict is always a pass and no sample is written: these tests assert on the
    SELECTION discovery makes, not on what a probe result does to the index.
    (test_probe_economy.py owns the recording behavior.)
    """

    def __init__(self) -> None:
        self.probed: list[tuple[str, str]] = []

    def probe_verdict(self, mid: str, oc: str, messages, timeout: float | None = None):
        self.probed.append((mid, oc))
        return ("pass", "")


def build_index(tmp) -> Index:
    """Build a fresh Index pre-loaded with FRESH, STALE, NEVER and one probe-
    sampled-then-aged-out sample each.

    FRESH: just sampled, scores now.
    STALE: sampled, then aged PAST the 24h scoring window — scores None (the
           regression case: the old gate read `last_ts > 0` and skipped it
           forever because it still had a last_sample_ts).
    NEVER: never sampled — scores None by absence, but must still be probed.
    """
    idx = Index(os.path.join(tmp, "stale.db"))
    idx.sync_catalog(PROVIDER, {FRESH, STALE, NEVER})
    now = time.time()
    idx.record(FRESH, "retain", "request", "ok", 3000.0)
    idx.record(STALE, "retain", "probe", "ok", 3000.0)
    idx._conn.execute(
        "UPDATE samples SET ts=? WHERE model_id=?",
        (now - (SCORE_WINDOW_S + 3600), STALE),
    )
    idx._conn.commit()
    return idx
