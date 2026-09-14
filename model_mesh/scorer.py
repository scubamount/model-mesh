"""model-mesh: scheduled scoring pass — probe traffic as standing evidence.

Eligibility floors and availability buckets are only as good as the samples
behind them, and samples EXPIRE (index.py SCORE_WINDOW_S / SCORE_RECENT_N).
Live traffic alone leaves thin lanes without evidence for longer than the
window, and a starved lane ranks on guesses — 1 eligible of 11 while models
were actually serving (2026-09, the auto/evolve recurrence; backfill was a
manual script nobody ran). This module drives router.probe_verdict() per
alias lane; dial() records probe samples into the index, so /mesh/status,
the floors and the cascade see this evidence exactly like live traffic.
No new recording path.

One bad probe must never end a lane; one bad lane must never end a pass —
a stopped pass silently re-creates the starvation this exists to prevent.
"""
import logging

from .opclass import probe_messages

log = logging.getLogger("model_mesh.scorer")


def run_score_pass(cfg: dict, router, candidates_for, top_n: int = 6) -> dict:
    """Probe top_n candidates for every alias once. Never raises per-probe.

    candidates_for: (index_or_none, provider, alias_cfg, alias) -> list[str],
    the same resolver the serving path uses (discovery.candidates_for with a
    partial for the alias), so scoring evidence lands on the same pool that
    routing reads. Returns {alias: {model_id: verdict}}.
    """
    provider = (cfg.get("provider") or {}).get("name", "nim")
    out: dict = {}
    for alias, acfg in (cfg.get("aliases") or {}).items():
        op_class = acfg.get("op_class", "retain")
        try:
            candidates = candidates_for(None, provider, acfg, alias)[:top_n]
        except Exception as e:
            log.warning("score-pass: candidates failed alias=%s: %s", alias, e)
            continue
        messages = probe_messages(op_class)
        verdicts: dict = {}
        for mid in candidates:
            try:
                verdict, _ = router.probe_verdict(mid, op_class, messages)
            except Exception as e:            # one bad dial must not end the lane
                verdict = "error"
                log.warning("score-pass probe failed alias=%s model=%s: %s",
                            alias, mid, e)
            verdicts[mid] = verdict
        out[alias] = verdicts
        log.info("score-pass alias=%s op_class=%s verdicts=%s",
                 alias, op_class, verdicts)
    return out
