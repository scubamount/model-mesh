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
    """Pattern filter: include/exclude substring-or-glob lists per alias."""
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
    log=print,
) -> dict:
    """One discovery pass. Returns the sync report augmented with probe results."""
    live = fetch_catalog(base, api_key)
    report = index.sync_catalog(provider, live)
    for mid in report["new"]:
        log(f"[mesh-discover] NEW  {mid}")
    for mid in report["eol"]:
        log(f"[mesh-discover] EOL  {mid} (catalog-drop)")
    for mid in report["returned"]:
        log(f"[mesh-discover] BACK {mid} (reappeared after EOL)")

    probed: dict[str, dict[str, bool]] = {}
    if probe_new:
        # Probe each new model once per op_class it could serve. Fidelity gate
        # decides entry; latency lands in the index as its first sample.
        op_classes_needed: dict[str, list[str]] = {}
        for alias, cfg in aliases.items():
            oc = cfg.get("op_class", "retain")
            for mid in report["new"]:
                if eligible_for_alias(mid, cfg):
                    op_classes_needed.setdefault(mid, [])
                    if oc not in op_classes_needed[mid]:
                        op_classes_needed[mid].append(oc)
        for mid, ocs in op_classes_needed.items():
            probed[mid] = {}
            for oc in ocs:
                ok = router.probe(mid, oc, probe_messages(oc))
                probed[mid][oc] = ok
                log(f"[mesh-discover] PROBE {mid} [{oc}] -> {'pass' if ok else 'FAIL'}")
    report["probed"] = probed
    return report


def candidates_for(
    index: Index, provider: str, alias_cfg: dict, cap: Optional[int] = None
) -> list[str]:
    """The alias's candidate pool, computed fresh from the index every call:
    live (non-EOL) models matching the alias's patterns. No static list."""
    pool = [
        m for m in index.live_models(provider) if eligible_for_alias(m, alias_cfg)
    ]
    return pool[:cap] if cap else pool
