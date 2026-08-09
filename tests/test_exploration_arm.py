"""Ranking must be able to change its mind.

The third ratchet (after discovery's stale-evidence and budget-ordering ones).
Ranking on its own is pure exploitation:

    evidence comes only from traffic
      -> traffic goes only to rank 0
        -> score() pins anything under CONFIDENT_N below the neutral prior
          -> a challenger never accumulates the evidence that would let it win

Observed live 2026-08-09 on the retain alias: gpt-oss-120b measured p95 2.6s,
gemma-4-31b-it measured p95 28.9s, and gemma held rank 0 indefinitely because it
was the only model with n >= CONFIDENT_N. Discovery cannot close this — it probes
for MISSING or STALE evidence, and the challenger's single fresh sample is
neither. A local optimum that reports itself as a global one.

These checks pin the exploration arm AND its safety rails: exploration must never
resurrect a model the breakers or floors excluded, and must never disturb models
that already have enough evidence to be judged on merit.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_mesh.index import CONFIDENT_N, Index  # noqa: E402
from model_mesh.router import Router, RouterConfig  # noqa: E402

OP = "retain"
INCUMBENT = "vendor/incumbent-well-evidenced"
CHALLENGER = "vendor/challenger-thin"
SECOND = "vendor/second-thin"


def build(tmp, explore_rate):
    idx = Index(os.path.join(tmp, f"r{explore_rate}.db"))
    idx.sync_catalog("nim", {INCUMBENT, CHALLENGER, SECOND})
    # Incumbent: enough samples to be judged on merit.
    for _ in range(CONFIDENT_N + 2):
        idx.record(INCUMBENT, OP, "request", "ok", 28000.0)
    # Challengers: one sample each. Fast, but structurally unable to earn more.
    idx.record(CHALLENGER, OP, "probe", "ok", 2600.0)
    idx.record(SECOND, OP, "probe", "ok", 2400.0)
    r = Router(idx, "http://x", "k", cfg=RouterConfig(explore_rate=explore_rate))
    return idx, r


def main():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        idx, router = build(tmp, 0.10)
        pool = [INCUMBENT, CHALLENGER, SECOND]

        ranked = router.ranked(pool, OP)

        # Fixture sanity: the bug must actually be present. If the challenger
        # already outranked the incumbent there would be nothing to fix.
        checks.append((
            "fixture: incumbent outranks a 11x faster challenger",
            ranked[0] == INCUMBENT,
        ))

        # 1. Over many draws, exploration must sometimes promote a thin model.
        promoted = set()
        for _ in range(400):
            promoted.add(router._explore(ranked, OP)[0])
        checks.append((
            "exploration promotes thin-evidence models",
            {CHALLENGER, SECOND} <= promoted,
        ))

        # 2. ...but the incumbent still leads the large majority of the time.
        firsts = [router._explore(ranked, OP)[0] for _ in range(1000)]
        inc_share = firsts.count(INCUMBENT) / len(firsts)
        checks.append((
            f"incumbent still leads most requests (share={inc_share:.2f})",
            0.80 <= inc_share <= 0.97,
        ))

        # 3. Cascade integrity: promotion reorders, never drops. A shortened
        #    cascade would trade redundancy for exploration.
        for _ in range(200):
            out = router._explore(ranked, OP)
            if sorted(out) != sorted(ranked) or len(out) != len(ranked):
                checks.append(("exploration preserves the full cascade", False))
                break
        else:
            checks.append(("exploration preserves the full cascade", True))

        # 4. explore_rate=0.0 must be exactly the old behavior.
        _, off = build(tmp, 0.0)
        off_ranked = off.ranked(pool, OP)
        checks.append((
            "explore_rate=0 is pure exploitation (unchanged order)",
            all(off._explore(off_ranked, OP) == off_ranked for _ in range(100)),
        ))

        # 5. Exploration must not resurrect an INELIGIBLE model. ranked() already
        #    applies breakers and floors; _explore only reorders what survived,
        #    so a gone model must stay gone no matter how thin its evidence is.
        idx.mark_gone(SECOND, "http-404")
        elig = router.ranked(pool, OP)
        checks.append((
            "fixture: gone model is excluded from ranking",
            SECOND not in elig,
        ))
        seen = set()
        for _ in range(400):
            seen.update(router._explore(elig, OP))
        checks.append((
            "exploration never resurrects an ineligible model",
            SECOND not in seen,
        ))

        # 6. A well-evidenced model must never be promoted BY exploration.
        #    Reordering models that already have merit-based scores is noise.
        idx2 = Index(os.path.join(tmp, "allfat.db"))
        idx2.sync_catalog("nim", {INCUMBENT, CHALLENGER})
        for _ in range(CONFIDENT_N + 2):
            idx2.record(INCUMBENT, OP, "request", "ok", 5000.0)
            idx2.record(CHALLENGER, OP, "request", "ok", 9000.0)
        r2 = Router(idx2, "http://x", "k", cfg=RouterConfig(explore_rate=1.0))
        rk2 = r2.ranked([INCUMBENT, CHALLENGER], OP)
        checks.append((
            "no thin candidates -> order untouched even at explore_rate=1.0",
            all(r2._explore(rk2, OP) == rk2 for _ in range(50)),
        ))

    failed = 0
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed += 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
