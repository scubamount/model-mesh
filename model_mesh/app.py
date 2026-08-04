"""model-mesh daemon: OpenAI-compatible endpoint + mesh introspection API.

Listens on 127.0.0.1:8002 (nim-proxy keeps :8001 until cutover). Aliases like
auto/retain resolve through the index-ranked cascade; raw model ids pass
through with breaker protection and telemetry but no cascade.

/health is DEEP: it fails (503) unless the upstream catalog is reachable AND
every configured alias has at least one healthy candidate. The predecessor's
health check could not fail while retain was down for a week; this one can.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config, resolve_api_key
from .discovery import candidates_for, discover, fetch_catalog
from .index import Index
from .opclass import probe_messages
from .router import Router, RouterConfig

CFG = load_config()
INDEX = Index(CFG["db_path"])
# Key resolver, not a cached string: read at CALL time from env, falling back
# to ~/.hermes/.env. `launchctl setenv` does not survive a machine restart, so a
# cached-at-import key meant the daemon 401'd every call until someone noticed.
def _api_key() -> str:
    return resolve_api_key(CFG["provider"]["api_key_env"])


ROUTER = Router(
    INDEX,
    CFG["provider"]["base_url"],
    _api_key,
    RouterConfig(**CFG.get("router", {})),
)
_DISCOVERY_LOCK = threading.Lock()

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
            ROUTER._call, model, body, "generic", "request"
        )
        if payload is not None:
            return JSONResponse(payload)
        return JSONResponse(
            {"error": "upstream failure", "attempt": vars(att)}, status_code=502
        )

    op_class = alias.get("op_class", "retain")
    pool = candidates_for(INDEX, CFG["provider"]["name"], alias,
                          cap=alias.get("max_candidates"))
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
        healthy = [m for m in pool if ROUTER._eligible(m, oc)]
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
        scores = {}
        for m in ranked[:10]:
            s = INDEX.score(m, oc)
            scores[m] = vars(s) if s else None
        out["aliases"][alias] = {
            "op_class": oc, "pool_size": len(pool), "ranking": ranked[:10],
            "scores": scores,
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
    audit = Path(os.path.expanduser("~/.model-mesh/audit")) / "discovery.jsonl"
    started = time.time()
    try:
        report = await asyncio.to_thread(
            discover,
            INDEX, ROUTER, CFG["provider"]["name"],
            CFG["provider"]["base_url"],
            _api_key(),
            CFG["aliases"],
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
    path = Path(os.path.expanduser("~/.model-mesh/audit")) / "discovery.jsonl"
    if not path.is_file():
        return {"runs": [], "note": "no discovery has been recorded yet"}
    lines = path.read_text().strip().splitlines()[-limit:]
    return {"runs": [json.loads(x) for x in lines]}
