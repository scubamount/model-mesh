"""model-mesh: catalog discovery.

Daily job (launchd) + on-demand via POST /mesh/probe. Diffs the provider's
live /v1/models against the index: new models get probed per op_class and
enter the rotation if they pass fidelity; vanished models get eol_at set.

This is the "check daily for newer models versus older models" arm — the
candidate pool is DISCOVERED, not enumerated. The predecessor's static
candidates.json is exactly what rotted (maverick EOL 2026-07-27, third
candidate a 404 typo that was never once probed successfully).
"""

from __future__ import annotations

import fnmatch
import json
import time
import urllib.request
from typing import Optional

from .index import SCORE_WINDOW_S, Index
from .opclass import probe_messages
from .quality import tier as quality_tier
from .router import Router

# A model that has not served a single successful request in this long is
# treated as retired even while the catalog still lists it. Discovery skips it
# instead of spending a probe, then admits exactly one probe per window so a
# model that genuinely returns comes back on its own.
#
# 7 days matches EOL_RECHECK_S and REJECT_RECHECK_S — the same "strong evidence,
# never permanent" trade the rest of the system makes. Long enough that a
# multi-day NIM outage does not retire a good model, short enough that a
# genuinely dead one stops costing a probe a day.
DORMANT_AFTER_S = 7 * 86400.0

# Refresh evidence BEFORE it expires, not exactly when it does.
#
# `stale_after_s` used to equal SCORE_WINDOW_S (24h) while the discovery pass
# itself runs once every 24h (StartCalendarInterval 06:15 local). Two equal
# periods with independent phase is a guaranteed miss, and it fired: on
# 2026-08-20 the auto/evolve lane's newest probe was 23.96h old — 131 SECONDS
# short of the cutoff — so every candidate read as "evidence still inside the
# scoring window", nothing was probed, and the evidence expired two minutes
# later. Result: auto/evolve carried 0 scored models for a full day while
# retain and consolidation (refreshed for free by live hindsight traffic) each
# carried 6. A lane with no live traffic of its own can ONLY be scored by this
# pass, so missing it by two minutes costs the whole day, every day, until some
# unrelated jitter shifts the phase.
#
# Refreshing at 80% of the window means a once-daily pass always re-probes with
# ~4.8h of margin. It cannot cause extra probes for a busy lane: real traffic
# keeps last_sample_ts inside the window regardless, and probe_top_n still caps
# breadth. Ratio, not a constant, so it tracks any change to SCORE_WINDOW_S —
# the invariant that matters is margin < window, and the assert below enforces
# the direction rather than trusting the arithmetic.
REFRESH_MARGIN = 0.8

# Direction, not arithmetic: a margin of 1.0 or more silently restores the
# 24h-vs-24h race this constant exists to remove, and the symptom (one lane
# unscored for a day) looks like a provider problem rather than a config one.
assert 0.0 < REFRESH_MARGIN < 1.0, (
    f"REFRESH_MARGIN must be a fraction of SCORE_WINDOW_S strictly under 1.0 "
    f"(got {REFRESH_MARGIN}); at >= 1.0 discovery refreshes evidence no earlier "
    f"than it expires and a once-daily pass loses any lane it misses by seconds"
)


def fetch_catalog(base: str, api_key: str, timeout: float = 30.0) -> set[str]:
    """Model IDs listed by the provider. NOT proof of serviceability.

    NIM keeps retired models in this listing while their chat endpoint
    hard-404s instantly (verified 2026-08-24: 6 of 6 sampled
    catalog-listed-but-eol-marked ids returned immediate 404s mid-episode;
    102 listed vs 67 actually serving). The catalog is only the candidate
    universe — request-time evidence and `eol_at` marks are the truth.
    Diffing this set against the index shows a gap during/after churn by
    design; that gap is not a missed-adoption defect.
    """
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return {m["id"] for m in data.get("data", [])}


def eligible_for_alias(model_id: str, alias_cfg: dict) -> bool:
    """Capability filter: exclude what CANNOT serve the op_class, admit the rest.

    Deliberately exclusion-first. An `include` whitelist of model families is the
    same static-list rot this module was built to kill: on 2026-08-08 the default
    include list (gpt-oss/llama-3.3/nemotron-super/qwen3) matched 5 of NIM's 102
    live models, so 95 were unrankable — not because they failed, but because
    nothing ever tried them. That silently excluded NIM's current flagships
    (nemotron-3-super-120b, kimi-k2.6, glm-5.2, mistral-large-2, deepseek-v4-pro)
    and made the pool shrink as NIM's catalog grew.

    `include` is still honored when set (an operator pinning a pool is a valid
    override), but the default is empty: anything not excluded is a candidate,
    and the fidelity gate in discovery decides admission EMPIRICALLY. Guessing
    from the model's name is what rots; measuring does not.
    """
    include = alias_cfg.get("include") or []
    exclude = alias_cfg.get("exclude") or []

    def _match(pats: list[str]) -> bool:
        return any(
            p in model_id or fnmatch.fnmatch(model_id, p) for p in pats
        )

    if include and not _match(include):
        return False
    if exclude and _match(exclude):
        return False
    return True


def discover(
    index: Index,
    router: Router,
    provider: str,
    base: str,
    api_key: str,
    aliases: dict[str, dict],
    probe_new: bool = True,
    max_probes: Optional[int] = None,
    stale_after_s: float = SCORE_WINDOW_S * REFRESH_MARGIN,
    probe_top_n: Optional[int] = None,
    dormant_after_s: float = DORMANT_AFTER_S,
    tier_overrides: Optional[dict] = None,
    fetch=None,
    log=print,
) -> dict:
    """One discovery pass. Returns the sync report augmented with probe results.

    `max_probes` caps how many distinct models are probed per pass so a wide
    catalog cannot turn a daily job into an unbounded burn. Remaining models are
    picked up by later passes.

    `probe_top_n` limits probing to the best few candidates per alias instead of
    every model in the catalog. We do not need to know the health of all 24
    models — only of the handful we would actually route to. A model we would
    never dial is one whose health is worth nothing, and probing it costs a real
    request against a shared free endpoint. Ordering is by quality tier, so the
    models kept warm are the ones worth having.

    `dormant_after_s` skips models that have had zero successes across every
    attempt in this window. NIM retires models without removing them from the
    catalog, so `live` is not the same as `servable`; without this, a model
    that 404s forever is re-probed on every pass forever. The skip is keyed to
    the window, not to "has ever been dormant": once the last attempt ages out
    of the window exactly one recheck probe runs per window after that, and a
    single success clears dormancy outright — never a one-way door.

    `stale_after_s` must track Index.score()'s window and land STRICTLY INSIDE
    it (SCORE_WINDOW_S * REFRESH_MARGIN — see the constant for the incident).
    Evidence EXPIRES: score() only reads samples newer than its window, so a
    model whose last sample is older than that scores None and sorts as
    "unknown" — behind every scored model, alphabetically. Setting this equal to
    the window makes a once-daily pass race the expiry and lose by seconds.
    Skipping any model that was ever sampled therefore
    creates a one-way ratchet: probe once, fall out of the window, never get
    re-probed, never score again. Observed 2026-08-09 with the retain alias, where
    gpt-oss-120b (measured 3.8s and fidelity-passing) sat at rank 30 on 52h-old
    evidence while gemma-4-31b-it (10.1s) served all traffic unopposed, purely
    because gemma was the only model with fresh samples. Re-probing on staleness
    is still self-limiting: real traffic refreshes the winner for free, so only
    models that are NOT being routed to ever cost a probe.
    """
    # Resolved at CALL time, not bound as a default. A default of
    # `fetch=fetch_catalog` captures the function object at import, which
    # silently defeats `monkeypatch.setattr("model_mesh.discovery.fetch_catalog")`
    # — the module attribute is rebound but this default still points at the
    # original, so six offline tests started making real network calls.
    live = (fetch or fetch_catalog)(base, api_key)
    report = index.sync_catalog(provider, live)
    for mid in report["new"]:
        log(f"[mesh-discover] NEW  {mid}")
    for mid in report["eol"]:
        log(f"[mesh-discover] EOL  {mid} (catalog-drop)")
    for mid in report["returned"]:
        log(f"[mesh-discover] BACK {mid} (reappeared after EOL)")

    probed: dict[str, dict[str, str]] = {}
    unusable: list[str] = []
    deferred: list[str] = []
    rejected: list[str] = []
    if probe_new:
        # Probe models whose evidence for an op_class is MISSING or STALE — not
        # merely the ones that are new in this pass. `report["new"]` is empty for
        # everything present at bootstrap, so the original version left 95 of 102
        # live models permanently unranked: never new again, never sampled, never
        # eligible.
        #
        # "Ever sampled" was the wrong test. Evidence expires (SCORE_WINDOW_S), so
        # a model probed once and then never routed to falls out of the scoring
        # window and becomes permanently unknown — a one-way ratchet that hands
        # the alias to whichever model already has traffic, forever. Re-probing on
        # staleness closes it and stays self-limiting, because the model actually
        # serving requests is refreshed for free by that traffic.
        stale_cutoff = time.time() - stale_after_s
        op_classes_needed: dict[str, list[str]] = {}
        dormant_skipped: list[str] = []
        for alias, cfg in aliases.items():
            oc = cfg.get("op_class", "retain")
            pool = candidates_for(index, provider, cfg)
            # Probe only the models we would actually route to. Health is worth
            # knowing about a candidate and worthless about a model the router
            # would never dial, and every probe is a real request against a
            # shared free endpoint. Ordering by quality tier keeps the STRONGEST
            # few warm rather than an arbitrary or alphabetical few.
            #
            # This is the practical shape of the problem: NIM's catalog is ~24
            # usable text models and grows, but the cascade only ever tries
            # max_attempts (8) of them. Probing every model to pick 8 spends the
            # budget proving that models we will not use are fine.
            if probe_top_n is not None:
                pool = sorted(
                    pool, key=lambda m: (-quality_tier(m, tier_overrides), m)
                )[:probe_top_n]
            for mid in pool:
                # last_sample_ts returns 0.0 (not None) when a model has never
                # been sampled — `is not None` would skip every model and probe
                # nothing at all.
                if index.last_sample_ts(mid, oc) > stale_cutoff:
                    continue  # evidence still inside the scoring window
                # Dormant gate. "Dormant" = tried within the dormancy window
                # and had zero successes in it: every attempt we made recently
                # failed, so the catalog listing is not evidence the model
                # serves. Skip — but the skip is keyed to the WINDOW, not to
                # the model having EVER been dormant. Gating on
                # `dormant_since() is not None` was a silent third ratchet
                # (found in audit 2026-08-24): the flag only clears via a
                # fresh success, only a probe can produce one, and the probe
                # was the thing being skipped — so a once-dormant lane was
                # unprobed FOREVER, contradicting this constant's own contract
                # of one recheck per window. Keyed to the window, the weekly
                # recheck emerges for free: each probe writes a sample, which
                # suppresses the next probe until the window elapses again,
                # and one success clears dormancy outright (self-rebutting).
                last_ts = index.last_sample_ts(mid, oc)
                ok_ts = index.last_success_ts(mid, oc)
                now = time.time()
                tried_recently = last_ts > 0.0 and (now - last_ts) < dormant_after_s
                worked_recently = ok_ts > 0.0 and (now - ok_ts) < dormant_after_s
                if tried_recently and not worked_recently:
                    if mid not in dormant_skipped:
                        dormant_skipped.append(mid)
                    continue
                op_classes_needed.setdefault(mid, [])
                if oc not in op_classes_needed[mid]:
                    op_classes_needed[mid].append(oc)
        for mid in dormant_skipped:
            log(f"[mesh-discover] DORMANT {mid} (no success in window — skipping probe)")

        budget = max_probes if max_probes is not None else len(op_classes_needed)
        # Probe the STALEST evidence first, never alphabetically. `sorted()` on
        # model_id composes with the per-pass budget into a second ratchet: with
        # 42 models needing probes and a budget of 25, everything from "n" onward
        # is deferred every pass, forever. Live effect 2026-08-09: gpt-oss-120b
        # and gpt-oss-20b were never probed for retain because "openai/" sorts
        # past the cutoff, so the two fastest fidelity-passing models in the pool
        # stayed unknown while 25 already-failing models were re-probed each pass.
        # Oldest-evidence-first makes the budget a rotation instead of a wall:
        # anything skipped this pass is strictly staler next pass, so it advances.
        def _staleness_key(mid: str) -> tuple[float, str]:
            ts = min(index.last_sample_ts(mid, oc) for oc in op_classes_needed[mid])
            return (ts, mid)

        for mid in sorted(op_classes_needed, key=_staleness_key):
            if budget <= 0:
                log(f"[mesh-discover] probe budget exhausted, {mid} deferred to next pass")
                break
            ocs = op_classes_needed[mid]
            probed[mid] = {}
            for oc in ocs:
                verdict, detail = router.probe_verdict(mid, oc, probe_messages(oc))
                probed[mid][oc] = verdict
                log(f"[mesh-discover] PROBE {mid} [{oc}] -> {verdict}"
                    + (f" ({detail})" if detail else ""))
                # A 404/410 means the catalog lists a model the provider will
                # not actually serve. That is EOL-now, not a bad model: leaving
                # it live means every pass re-probes it forever.
                if verdict == "unusable" and "not servable" in detail:
                    index.mark_gone(mid, "http-404")
                    unusable.append(mid)
                    break
                # `busy` records NOTHING permanent. The model was overloaded;
                # it keeps no evidence and gets re-probed on a later pass, so a
                # popular model is never excluded for being popular.
                if verdict == "busy":
                    deferred.append(mid)
                # `rejected` = 400/413/422: the provider parsed the request and
                # refused it, so this op_class is settled until the evidence
                # goes stale. _call already recorded the sample, so scoring and
                # Router.eligible (via unrebutted_reject) both see it and no
                # further pass re-probes this pair. Keep probing the model's
                # OTHER op_classes: a reject is per-request-shape, not a
                # property of the model.
                if verdict == "rejected":
                    rejected.append(mid)
            budget -= 1
    report["probed"] = probed
    report["unusable"] = unusable
    report["deferred_busy"] = deferred
    report["rejected"] = rejected
    return report


def candidates_for(
    index: Index, provider: str, alias_cfg: dict, cap: Optional[int] = None
) -> list[str]:
    """The alias's candidate pool, computed fresh from the index every call:
    live (non-EOL) models matching the alias's patterns. No static list.

    `cap` is NOT applied here. index.live_models() returns rows in SQLite
    insertion order, so slicing before Router.ranked() would keep an arbitrary
    subset and could discard the best-scoring model outright. With a wide
    catalog that turns the cap into a correctness bug rather than a cost
    control, so ordering happens first and the caller caps the RANKED list.
    Cost is bounded by max_attempts, not by pool size — the router only
    ever dials the top few.
    """
    return [
        m for m in index.live_models(provider) if eligible_for_alias(m, alias_cfg)
    ]
