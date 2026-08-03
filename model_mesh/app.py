"""model-mesh daemon: OpenAI-compatible endpoint + mesh introspection API.

Listens on 127.0.0.1:8002 (nim-proxy keeps :8001 until cutover). Aliases like
auto/retain resolve through the index-ranked cascade; raw model ids pass
through with breaker protection and telemetry but no cascade.

/health is DEEP: it fails (503) unless the upstream catalog is reachable AND
every configured alias has at least one healthy candidate. The predecessor's
health check could not fail while retain was down for a week; this one can.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config
from .discovery import candidates_for, discover, fetch_catalog
from .index import Index
from .opclass import probe_messages
from .router import Router, RouterConfig

CFG = load_config()
INDEX = Index(CFG["db_path"])
ROUTER = Router(
    INDEX,
    CFG["provider"]["base_url"],
    os.environ.get(CFG["provider"]["api_key_env"], ""),
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

    if alias is None:
        # Raw model id: single call, breaker + telemetry still apply.
        payload, att = ROUTER._call(model, body, "generic", "request")
        if payload is not None:
            return JSONResponse(payload)
        return JSONResponse(
            {"error": "upstream failure", "attempt": vars(att)}, status_code=502
        )

    op_class = alias.get("op_class", "retain")
    pool = candidates_for(INDEX, CFG["provider"]["name"], alias,
                          cap=alias.get("max_candidates"))
    result = ROUTER.route(pool, body, op_class,
                          probe_messages=probe_messages(op_class))
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
        fetch_catalog(
            CFG["provider"]["base_url"],
            os.environ.get(CFG["provider"]["api_key_env"], ""),
            timeout=10.0,
        )
    except Exception as e:  # noqa: BLE001
        problems["catalog"] = f"unreachable: {type(e).__name__}"

    for alias, cfg in CFG["aliases"].items():
        pool = candidates_for(INDEX, CFG["provider"]["name"], cfg)
        healthy = [m for m in pool if ROUTER._eligible(m)]
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
    """Force a discovery pass. Rate-limited to one at a time."""
    if not _DISCOVERY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "already running"}, status_code=429)
    try:
        report = discover(
            INDEX, ROUTER, CFG["provider"]["name"],
            CFG["provider"]["base_url"],
            os.environ.get(CFG["provider"]["api_key_env"], ""),
            CFG["aliases"],
        )
        return report
    finally:
        _DISCOVERY_LOCK.release()
