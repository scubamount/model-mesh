"""model-mesh daemon: OpenAI-compatible endpoint + mesh introspection API.

Listens on the configured `listen` address (default 127.0.0.1:8002). Aliases
like auto/retain resolve through the index-ranked cascade; raw model ids pass
through with breaker protection and telemetry but no cascade.

/health is DEEP: it fails (503) unless the upstream catalog is reachable AND
every configured alias has at least one healthy candidate. The predecessor's
health check could not fail while retain was down for a week; this one can.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config, resolve_api_key
from .discovery import candidates_for, discover, fetch_catalog
from .index import Index
from .opclass import probe_messages
from .quality import (
    BUCKET_FAILING,
    BUCKET_HEALTHY,
    BUCKET_OVERLOADED,
    BUCKET_UNKNOWN,
    availability_bucket,
)
from .quality import tier as quality_tier
from .router import Router, RouterConfig

_BUCKET_NAMES = {
    BUCKET_HEALTHY: "healthy",
    BUCKET_OVERLOADED: "overloaded",
    BUCKET_UNKNOWN: "unknown",
    BUCKET_FAILING: "failing",
}

CFG = load_config()
INDEX = Index(CFG["db_path"])


def _state_dir() -> Path:
    """The directory holding mesh.db — the mesh's whole state dir (db, logs,
    audit trail). Derived from db_path so relocating state is a config edit,
    not a code change."""
    return Path(CFG["db_path"]).expanduser().parent

# Key resolver, not a cached string: read at CALL time from env, falling back
# to the state dir's .env file (see config.KEY_FALLBACK_FILE). `launchctl
# setenv` does not survive a machine restart, so a cached-at-import key meant
# the daemon 401'd every call until someone noticed.
def _api_key() -> str:
    return resolve_api_key(CFG["provider"]["api_key_env"])


ROUTER = Router(
    INDEX,
    CFG["provider"]["base_url"],
    _api_key,
    RouterConfig(**CFG.get("router", {})),
)
_DISCOVERY_LOCK = threading.Lock()
# Read off the ROUTER's own config rather than re-reading CFG: /mesh/status must
# report the values that actually rank requests. Two independent reads of the
# same config is how a status page starts describing a system that no longer
# exists.
_TIER_OVERRIDES = ROUTER.cfg.tier_overrides
_OVERLOAD_P95_MS = ROUTER.cfg.overload_p95_ms

app = FastAPI(title="model-mesh")


def _alias_cfg(model: str) -> dict | None:
    return CFG["aliases"].get(model)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = (body.get("model") or "").strip()
    alias = _alias_cfg(model)

    # Router I/O is blocking urllib — run it off the event loop, or one slow
    # upstream call (30-90s on big payloads) freezes every endpoint including
    # /health. Observed live 2026-08-03: /mesh/status returned http=000 while
    # a retain was in flight.
    if alias is None:
        payload, att = await asyncio.to_thread(
            ROUTER.dial, model, body, "generic", "request"
        )
        if payload is not None:
            return JSONResponse(payload)
        return JSONResponse(
            {"error": "upstream failure", "attempt": vars(att)}, status_code=502
        )

    op_class = alias.get("op_class", "retain")
    # Hand the router the RAW candidate pool. Ranking happens ONCE, inside
    # route(): a pre-ranked handoff is what made "no healthy candidates"
    # reachable twice over on 2026-08-24 — first via the max_candidates
    # slice, then because ranked()=[] (floors excluded everything) meant the
    # router received an EMPTY candidate list and could 503 in 11ms with
    # zero dials. The router needs the unfiltered list so its last-resort
    # sweep can walk models the floors rejected; it ranks for dial order
    # itself.
    pool = candidates_for(INDEX, CFG["provider"]["name"], alias)
    result = await asyncio.to_thread(
        ROUTER.route, pool, body, op_class, probe_messages(op_class)
    )
    if result.ok:
        resp = JSONResponse(result.response)
        resp.headers["x-mesh-routed-model"] = result.model_id or ""
        resp.headers["x-mesh-attempts"] = str(len(result.attempts))
        return resp
    return JSONResponse(
        {
            "error": f"all candidates failed for {model}",
            "op_class": op_class,
            "reprobed": result.reprobed,
            "models_tried": [vars(a) for a in result.attempts],
        },
        status_code=503,
    )


@app.get("/v1/models")
async def models():
    data = [
        {"id": a, "object": "model", "owned_by": "model-mesh"}
        for a in CFG["aliases"]
    ]
    data += [
        {"id": m, "object": "model", "owned_by": CFG["provider"]["name"]}
        for m in INDEX.live_models(CFG["provider"]["name"])
    ]
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    """Deep health: catalog reachable AND >=1 healthy candidate per alias."""
    problems: dict[str, str] = {}
    try:
        await asyncio.to_thread(
            fetch_catalog,
            CFG["provider"]["base_url"],
            _api_key(),
            10.0,
        )
    except Exception as e:  # noqa: BLE001
        problems["catalog"] = f"unreachable: {type(e).__name__}"

    for alias, cfg in CFG["aliases"].items():
        pool = candidates_for(INDEX, CFG["provider"]["name"], cfg)
        oc = cfg.get("op_class", "retain")
        healthy = [m for m in pool if ROUTER.eligible(m, oc)]
        if not healthy:
            problems[alias] = "no healthy candidates"

    if problems:
        return JSONResponse({"status": "unhealthy", "problems": problems},
                            status_code=503)
    return {"status": "healthy"}


@app.get("/mesh/status")
async def mesh_status():
    out: dict = {"aliases": {}, "breaker": INDEX.breaker_all()}
    for alias, cfg in CFG["aliases"].items():
        oc = cfg.get("op_class", "retain")
        pool = candidates_for(INDEX, CFG["provider"]["name"], cfg)
        ranked = ROUTER.ranked(pool, oc)
        # Score EVERY model that has evidence, not just the visible top 10.
        # A truncated view is how the 2026-08-08 eviction hid: a model with
        # 36/36 successes fell to rank 13 and simply vanished off the end of
        # this response, so neither /mesh/status nor any check reading it could
        # see that the proven model had lost its cascade slot. Reporting is
        # cheap; a blind spot exactly the width of the bug is not.
        scores = {}
        for m in ranked:
            s = INDEX.score(m, oc)
            if s is not None:
                scores[m] = vars(s)
        # The two ranking inputs, reported for EVERY ranked model. Ordering is
        # (bucket, tier, latency), so without these a reader can see the order
        # but not the reason for it — and the previous ranking failed silently
        # in exactly that way: every score collapsed to ~50.0, the visible
        # ordering became alphabetical, and nothing in this response said so.
        # Anything that decides routing has to be inspectable here.
        rank_inputs = {
            m: {
                "tier": quality_tier(m, _TIER_OVERRIDES),
                "bucket": _BUCKET_NAMES[
                    availability_bucket(INDEX.score(m, oc), _OVERLOAD_P95_MS)
                ],
            }
            for m in ranked
        }
        out["aliases"][alias] = {
            "op_class": oc,
            "pool_size": len(pool),
            "ranking": ranked[:10],
            # Full ordering, so a consumer can tell "ranked 13th" from
            # "not ranked at all" — those have very different causes.
            "ranking_all": ranked,
            # max_candidates was retired 2026-08-24: the pre-router slice made
            # "no healthy candidates" reachable (see chat_completions). Dial
            # depth is RouterConfig.max_attempts + the sweep arm now.
            "max_candidates": None,
            "scores": scores,
            "rank_inputs": rank_inputs,
            # The timeouts THIS op_class actually runs under, read off the live
            # router rather than restated from config. Added 2026-08-24: the
            # per-op_class override that fixed auto/consolidation could not be
            # confirmed in-process at all — /mesh/status exposed no config, so
            # "is the new knob wired?" was unanswerable without restarting the
            # daemon and inferring from behavior. A value that decides routing
            # has to be inspectable here, same rule as rank_inputs above.
            "timeouts": {
                "request_timeout_s": ROUTER.request_timeout(oc),
                "probe_timeout_s": ROUTER.probe_timeout(oc),
            },
        }
    return out


@app.get("/mesh/models")
async def mesh_models():
    live = INDEX.live_models(CFG["provider"]["name"])
    return {"live": live, "count": len(live), "breaker": INDEX.breaker_all()}


@app.post("/mesh/probe")
async def mesh_probe():
    """Force a discovery pass. Rate-limited to one at a time.

    Always appends a JSONL audit record — including on failure. The launchd job
    only captures curl's stdout, so a curl timeout (exit 28) left a 0-byte
    discover.log and NO trace that discovery had failed. A job whose failure is
    invisible is not a check.
    """
    if not _DISCOVERY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "already running"}, status_code=429)
    # Audit trail lives in the SAME state directory as mesh.db — one dir holds
    # everything an operator must back up.
    audit = _state_dir() / "audit" / "discovery.jsonl"
    started = time.time()
    try:
        _disc = CFG.get("discovery") or {}
        report = await asyncio.to_thread(
            discover,
            INDEX, ROUTER, CFG["provider"]["name"],
            CFG["provider"]["base_url"],
            _api_key(),
            CFG["aliases"],
            # Keyword from here on. These were positional, which silently binds
            # by ORDER: inserting a parameter in discover()'s signature would
            # have re-aimed max_probes at a different argument with no error.
            probe_new=True,
            max_probes=_disc.get("max_probes_per_pass"),
            probe_top_n=_disc.get("probe_top_n"),
            tier_overrides=(CFG.get("router") or {}).get("tier_overrides") or {},
        )
        _audit(audit, {"ts": started, "ok": True,
                       "duration_s": round(time.time() - started, 1),
                       **{k: report.get(k) for k in
                          ("new", "eol", "returned", "probed")}})
        return report
    except Exception as e:  # noqa: BLE001
        _audit(audit, {"ts": started, "ok": False,
                       "duration_s": round(time.time() - started, 1),
                       "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        _DISCOVERY_LOCK.release()


def _audit(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


@app.get("/mesh/discovery")
async def mesh_discovery(limit: int = 10):
    """Last N discovery outcomes. This is how you check the daily job actually
    ran and what it found — not by tailing a curl-owned log file."""
    path = _state_dir() / "audit" / "discovery.jsonl"
    if not path.is_file():
        return {"runs": [], "note": "no discovery has been recorded yet"}
    lines = path.read_text().strip().splitlines()[-limit:]
    return {"runs": [json.loads(x) for x in lines]}


def main() -> None:
    """Serve on the address in config. THE entrypoint — the launchd plist calls
    this instead of passing `--host/--port` to uvicorn itself.

    Reason: `listen` shipped in DEFAULTS while the plist hardcoded the address
    on the uvicorn command line, so config said one thing and the running daemon
    did another, and editing config.yaml moved nothing (audit 2026-08-25). One
    source of truth, and it is the config.
    """
    import uvicorn

    listen = CFG["listen"]
    uvicorn.run(
        app,
        host=listen["host"],
        port=int(listen["port"]),
        access_log=False,
    )


if __name__ == "__main__":
    main()
