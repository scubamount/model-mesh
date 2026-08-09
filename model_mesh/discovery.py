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
from .opclass import check_fidelity, probe_messages
from .router import Router


def fetch_catalog(base: str, api_key: str, timeout: float = 30.0) -> set[str]:
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
    stale_after_s: float = SCORE_WINDOW_S,
    log=print,
) -> dict:
    """One discovery pass. Returns the sync report augmented with probe results.

    `max_probes` caps how many distinct models are probed per pass so a wide
    catalog cannot turn a daily job into an unbounded burn. Remaining models are
    picked up by later passes.

    `stale_after_s` must track Index.score()'s window. Evidence EXPIRES: score()
    only reads samples newer than its window, so a model whose last sample is
    older than that scores None and sorts as "unknown" — behind every scored
    model, alphabetically. Skipping any model that was ever sampled therefore
    creates a one-way ratchet: probe once, fall out of the window, never get
    re-probed, never score again. Observed 2026-08-09 with the retain alias, where
    gpt-oss-120b (measured 3.8s and fidelity-passing) sat at rank 30 on 52h-old
    evidence while gemma-4-31b-it (10.1s) served all traffic unopposed, purely
    because gemma was the only model with fresh samples. Re-probing on staleness
    is still self-limiting: real traffic refreshes the winner for free, so only
    models that are NOT being routed to ever cost a probe.
    """
    live = fetch_catalog(base, api_key)
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
        for alias, cfg in aliases.items():
            oc = cfg.get("op_class", "retain")
            for mid in candidates_for(index, provider, cfg):
                # last_sample_ts returns 0.0 (not None) when a model has never
                # been sampled — `is not None` would skip every model and probe
                # nothing at all.
                if index.last_sample_ts(mid, oc) > stale_cutoff:
                    continue  # evidence still inside the scoring window
                op_classes_needed.setdefault(mid, [])
                if oc not in op_classes_needed[mid]:
                    op_classes_needed[mid].append(oc)

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
                # Router._eligible (via unrebutted_reject) both see it and no
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
    Cost is bounded by max_attempts (3), not by pool size — the router only
    ever dials the top few.
    """
    return [
        m for m in index.live_models(provider) if eligible_for_alias(m, alias_cfg)
    ]
