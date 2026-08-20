"""Discovery must re-probe STALE evidence, not only missing evidence.

The bug this pins (observed live 2026-08-09, retain alias):

Index.score() reads a sliding window (SCORE_WINDOW_S). A model whose newest
sample is older than that window scores None, and Router.ranked() sorts unknowns
after every scored model — so expired evidence is indistinguishable from never
having been measured.

Discovery skipped any model with `last_sample_ts(mid, oc) > 0`, i.e. anything
ever sampled. Those two rules compose into a one-way ratchet:

    probe once -> evidence ages out of the window -> scores None ->
    ranks last -> never routed to -> never resampled -> never re-probed

The alias then belongs permanently to whichever model happens to carry live
traffic, because only that model's samples stay fresh. Live effect: gpt-oss-120b
measured 3.8s and passed the retain fidelity gate, but sat at rank 30 on 52h-old
evidence while gemma-4-31b-it (10.1s) served 100% of retain traffic unopposed.

Nothing in the system reported a fault. Every gate was green; the pool was
"40 models" and 39 of them were unreachable.
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_mesh.discovery import discover  # noqa: E402
from model_mesh.index import SCORE_WINDOW_S, Index  # noqa: E402

ALIASES = {"auto/retain": {"op_class": "retain"}}
PROVIDER = "nim"
FRESH = "model/fresh-incumbent"
STALE = "model/stale-challenger"
NEVER = "model/never-sampled"


class FakeRouter:
    """Records what discovery chose to probe. Probing is the behavior under
    test, so the verdict is always `pass` — a probe that never happens cannot be
    rescued by any later stage."""

    def __init__(self):
        self.probed: list[tuple[str, str]] = []

    def probe_verdict(self, model_id, op_class, messages, timeout=None):
        self.probed.append((model_id, op_class))
        return "pass", "ok"


def build_index(tmpdir):
    idx = Index(os.path.join(tmpdir, "t.db"))
    idx.sync_catalog(PROVIDER, {FRESH, STALE, NEVER})
    now = time.time()
    # Incumbent: sampled recently, inside the scoring window.
    idx.record(FRESH, "retain", "request", "ok", 4000.0)
    # Challenger: sampled, but older than the scoring window -> scores None.
    idx.record(STALE, "retain", "probe", "ok", 3800.0)
    idx._conn.execute(
        "UPDATE samples SET ts=? WHERE model_id=?",
        (now - (SCORE_WINDOW_S + 3600), STALE),
    )
    idx._conn.commit()
    # NEVER gets no samples at all.
    return idx


def run(monkeypatched_catalog=True):
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp)
        router = FakeRouter()

        import model_mesh.discovery as disc

        orig = disc.fetch_catalog
        disc.fetch_catalog = lambda base, key, timeout=30.0: {FRESH, STALE, NEVER}
        try:
            discover(
                idx, router, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig

        probed = {m for m, _ in router.probed}

        # 1. THE REGRESSION. Stale evidence must be refreshed. Under the old
        #    `last_sample_ts > 0` test this model is skipped forever.
        checks.append((
            "stale-evidence model is re-probed",
            STALE in probed,
        ))

        # 2. Never-sampled models must still be probed (original behavior).
        checks.append((
            "never-sampled model is probed",
            NEVER in probed,
        ))

        # 3. Fresh evidence must NOT be re-probed, or every pass re-probes
        #    everything and the budget stops bounding cost.
        checks.append((
            "fresh-evidence model is skipped",
            FRESH not in probed,
        ))

        # 3b. THE BOUNDARY, which check 3 above cannot see. FRESH is sampled
        #     "now", so it proves only that obviously-fresh evidence is skipped.
        #     The live defect lived at the far edge of the window: discovery runs
        #     once per 24h and stale_after_s was ALSO 24h, so a model probed
        #     23.96h ago read as fresh, was skipped, and expired 131 seconds
        #     later — auto/evolve carried 0 scored models for a full day
        #     (2026-08-20). Equal periods with independent phase always lose
        #     eventually; this asserts the refresh happens with margin.
        idx3 = Index(os.path.join(tmp, "boundary.db"))
        near = "boundary/expires-in-minutes"
        idx3.sync_catalog(PROVIDER, {near})
        idx3.record(near, "retain", "probe", "ok", 3000.0)
        # 99% of the way through the window: still scoreable RIGHT NOW, but gone
        # long before a once-daily pass comes round again.
        idx3._conn.execute(
            "UPDATE samples SET ts=? WHERE model_id=?",
            (time.time() - (SCORE_WINDOW_S * 0.99), near),
        )
        idx3._conn.commit()

        # The fixture must be non-vacuous in BOTH directions: the model has to
        # still score now (else "unscored tomorrow" proves nothing) and has to
        # be inside the raw window (else it is just the stale case again).
        checks.append((
            "boundary fixture still scores today (not yet expired)",
            idx3.score(near, "retain") is not None,
        ))

        r3 = FakeRouter()
        orig3 = disc.fetch_catalog
        disc.fetch_catalog = lambda base, key, timeout=30.0: {near}
        try:
            discover(
                idx3, r3, PROVIDER, "http://x", "k", ALIASES,
                probe_new=True, log=lambda *a, **k: None,
            )
        finally:
            disc.fetch_catalog = orig3

        checks.append((
            "about-to-expire model is re-probed before it expires",
            near in {m for m, _ in r3.probed},
        ))

        # And the constant that buys the margin must stay a fraction. At >= 1.0
        # the race returns silently and the symptom reads as a provider outage.
        checks.append((
            "REFRESH_MARGIN is strictly inside the scoring window",
            0.0 < disc.REFRESH_MARGIN < 1.0,
        ))

        # 4. The staleness threshold must track the scoring window. If discovery
        #    tolerated evidence older than score() accepts, models would expire
        #    out of ranking faster than they are re-probed.
        idx2 = build_index(os.path.join(tmp, "b"))
        checks.append((
            "stale model scores None (proves the fixture reproduces the bug)",
            idx2.score(STALE, "retain") is None,
        ))
        checks.append((
            "fresh model scores non-None (proves the fixture is not vacuous)",
            idx2.score(FRESH, "retain") is not None,
        ))

        # 5. BUDGET ORDERING. A per-pass probe budget plus alphabetical iteration
        #    is a second ratchet: with more models needing probes than the budget
        #    allows, everything past the alphabetical cutoff is deferred every
        #    pass, forever. Live effect: openai/gpt-oss-* sorted past a budget of
        #    25 and was never probed for retain, so the fastest fidelity-passing
        #    models in the pool stayed permanently unknown.
        checks.extend(_check_budget_ordering(tmp))

    return checks


def _check_budget_ordering(tmp):
    """With a budget smaller than the work, the STALEST model must be probed —
    not the alphabetically-first one."""
    import model_mesh.discovery as disc

    idx = Index(os.path.join(tmp, "ord.db"))
    # "aaa/*" sorts first; "zzz/*" sorts last but carries the oldest evidence.
    early, late = "aaa/alphabetically-first", "zzz/oldest-evidence"
    idx.sync_catalog(PROVIDER, {early, late})
    now = time.time()
    idx.record(early, "retain", "probe", "ok", 1000.0)
    idx.record(late, "retain", "probe", "ok", 1000.0)
    idx._conn.execute(
        "UPDATE samples SET ts=? WHERE model_id=?",
        (now - (SCORE_WINDOW_S + 3600), early),
    )
    idx._conn.execute(
        "UPDATE samples SET ts=? WHERE model_id=?",
        (now - (SCORE_WINDOW_S + 100000), late),
    )
    idx._conn.commit()

    router = FakeRouter()
    orig = disc.fetch_catalog
    disc.fetch_catalog = lambda base, key, timeout=30.0: {early, late}
    try:
        disc.discover(
            idx, router, PROVIDER, "http://x", "k", ALIASES,
            probe_new=True, max_probes=1, log=lambda *a, **k: None,
        )
    finally:
        disc.fetch_catalog = orig

    probed = [m for m, _ in router.probed]
    return [
        (
            "budget spends on the STALEST model, not the alphabetically-first",
            probed == [late],
        ),
    ]


def main():
    os.makedirs(os.path.expanduser("~/.model-mesh"), exist_ok=True)
    checks = run()
    failed = 0
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed += 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
