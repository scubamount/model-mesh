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
import urllib.request
from typing import Optional

from .index import Index
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
    log=print,
) -> dict:
    """One discovery pass. Returns the sync report augmented with probe results.

    `max_probes` caps how many distinct models are probed per pass so a wide
    catalog cannot turn a daily job into an unbounded burn. Remaining models are
    picked up by later passes; steady state is ~0 probes because every model
    already carries evidence.
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
        # Probe models with NO evidence for an op_class — not merely the ones
        # that are new in this pass. `report["new"]` is empty for everything
        # present at bootstrap, so the original version left 95 of 102 live
        # models permanently unranked: never new again, never sampled, never
        # eligible. Backfilling by "has this model ever been sampled for this
        # op_class" is self-limiting — each model is probed once per op_class,
        # then real traffic maintains it for free.
        op_classes_needed: dict[str, list[str]] = {}
        for alias, cfg in aliases.items():
            oc = cfg.get("op_class", "retain")
            for mid in candidates_for(index, provider, cfg):
                # last_sample_ts returns 0.0 (not None) when a model has never
                # been sampled — `is not None` would skip every model and probe
                # nothing at all.
                if index.last_sample_ts(mid, oc) > 0:
                    continue  # already has evidence; scoring handles it
                op_classes_needed.setdefault(mid, [])
                if oc not in op_classes_needed[mid]:
                    op_classes_needed[mid].append(oc)

        budget = max_probes if max_probes is not None else len(op_classes_needed)
        for mid in sorted(op_classes_needed):
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
