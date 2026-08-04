# model-mesh

**One local OpenAI-compatible endpoint that never dies.**

A health-aware model router for NVIDIA NIM (and any OpenAI-compatible upstream).
Replaces `scubamount-nim-proxy` + its hourly auto-ranker with request-time
routing built on a persistent model index.

## Why (the failure that motivated this)

The predecessor (`nim-proxy` + hourly ranker) failed in a specific, instructive way:

1. `meta/llama-4-maverick` hit EOL (`410 Gone`) on 2026-07-27. The static
   candidate list silently shrank to one model.
2. The litellm fallback tier was hardcoded to the *same* model the alias
   resolved to — primary and fallback were identical. Zero redundancy.
3. The hourly ranker probed with a ~50-char toy prompt; real retain payloads
   are ~14k chars. The probe ranked gpt-oss-20b "fastest" (~2s) while it took
   27.9s on real work — starving hindsight's retain into 300s timeouts.
4. Every health check that should have caught this could not fail:
   `/health` only checked Postgres, the ranker only checked its own toy probe.

Memory writes silently failed for a week. **Every design decision below traces
back to one of those four failures.**

## Architecture

```
client (hindsight / hermes / opencode)
   │  POST /v1/chat/completions  model=auto/retain
   ▼
┌─────────────────────────────────────────────┐
│ model-mesh daemon (localhost only)          │
│                                             │
│  resolve alias → ranked candidate list      │
│    (from index: stability score, breaker)   │
│  try #1 ──fail──▶ try #2 ──fail──▶ try #3   │
│    │                                        │
│    └── all failed? LIVE RE-PROBE the pool,  │
│        rebuild ranking, one more cascade;   │
│        only then 503 (with models_tried)    │
│                                             │
│  every real request updates the index       │
│  (latency, status) — traffic IS telemetry   │
└─────────────────────────────────────────────┘
   │ upstream: NIM cloud (more providers later)
   ▼
┌─────────────────────────────────────────────┐
│ model index (SQLite, ~/.model-mesh/mesh.db) │
│  catalog snapshots · probe+request history  │
│  stability scores · breaker states · EOL log│
└─────────────────────────────────────────────┘
```

### The model index (the durable part)

SQLite at `~/.model-mesh/mesh.db`. Tables:

- `models` — every model ever seen in a catalog sync: id, provider,
  first_seen, last_seen, eol_at (when it vanished or started returning
  410/404), metadata. Models are never deleted — EOL is data.
- `samples` — every probe AND every real request: model, ts, op_class,
  latency_ms, status (http code or timeout/parse-fail), payload_chars.
  Real traffic is the best probe; we log it for free.
- `breaker` — current circuit state per model (healthy / down / recovering /
  auth / gone), consecutive fails, cooldown_until.

Stability score per (model, op_class), computed over a sliding window
(default 24h), FCM-inspired weights:

```
score = 0.30 * p95_latency_component
      + 0.30 * jitter_component        (stdev / median)
      + 0.20 * spike_rate_component    (samples > 3x median)
      + 0.20 * success_rate_component
```

Ranking = healthy models sorted by score. A model with one lucky fast ping
does not outrank a consistently-good one — variance is measured, not assumed.

### Request-time routing (the part nim-proxy never had)

Per request:

1. Resolve alias (`auto/retain` → ranked candidates for op_class `retain`).
2. Skip models whose breaker is open (down/gone/auth) or cooling down.
3. Try candidate #1. On transient failure (429, 5xx, timeout, malformed
   JSON) → breaker counts it, cascade to #2, then #3 (`max_attempts`, default 3).
4. **All tried and failed → live re-probe**: hit `/v1/models` + 1-token pings
   on the top pool, rebuild ranking from fresh data, run ONE more cascade.
   This is the "if nothing works, ping again and find what's actually up
   right now" arm — it converts a stale-index total-miss into a recovery
   instead of an outage.
5. Still nothing → `503` with `models_tried` + per-model failure reasons.

Status-code discipline (learned from FCM):

- `429 / 5xx / timeout / HTML-instead-of-JSON` → transient, cascade.
- `401 / 403` → auth, mark separately, never poisons the breaker.
- `404 / 410` → **gone**: mark `eol_at` in the index immediately. This is the
  maverick lesson — EOL detection at request time, not next audit.

### Circuit breaker

Per model: `healthy → down` after N consecutive fails (default 3);
`down → recovering` after cooldown (default 120s); `recovering` admits one
real request — success closes the circuit, failure reopens with doubled
cooldown (cap 30 min).

### Discovery (daily) + probe (adaptive)

- **Daily catalog sync** (launchd): `GET /v1/models` upstream, diff against
  index. New models → probed once per op_class, enter ranking if they pass
  fidelity. Vanished models → `eol_at` set, breaker `gone`. Log line per
  change: `NEW nvidia/x-9b`, `EOL meta/maverick`.
- **Fidelity gate** per op_class before a model may serve it (inherited from
  nim-proxy's ranker, kept verbatim in spirit): retain/consolidation demand
  strict-JSON output of the expected shape; a tool-call probe validates
  `tool_calls` for op-classes that need it.
- **Representative probes**: probe payload is padded to the op_class's real
  size (retain ≈ 12k chars). A probe that can't fail like production fails
  is worse than no probe.
- **Background probe cadence** only for models with no recent traffic
  (default: skip if a real sample landed in the last 10 min — traffic is
  telemetry, don't burn quota double-checking it).

### API surface

- `POST /v1/chat/completions` — OpenAI-compatible; `model` = alias or raw id.
- `GET /v1/models` — upstream catalog passthrough + aliases.
- `GET /mesh/status` — ranking, breaker states, last sync, per-alias health.
- `GET /mesh/models` — the index: scores, history summary, EOL list.
- `POST /mesh/probe` — force a re-probe (used by cron; rate-limited).
- `GET /health` — **deep** health: catalog reachable AND ≥1 healthy model per
  configured alias. Returns 503 otherwise. (The predecessor's `/health`
  could not fail while retain was down; this one can.)

### Config

`~/.model-mesh/config.yaml` — providers (base URL + key env var), aliases →
op_class + candidate filters (include/exclude patterns — the pool is
discovered, not enumerated), breaker/probe tunables. Secrets stay in env,
never in the DB or config.

## Compatibility

Drop-in for the nim-proxy contract hindsight already speaks:
`openai/auto/retain`, `auto/consolidation`, `auto/reflect` aliases on an
OpenAI-compatible localhost port. Cutover = change one port in
`060-hindsight-setup.sh` (`:8001` → `:8002`), nothing else.

`auto/evolve` serves DSPy skill evolution (hermes-agent-self-evolution) on its
own `evolve` op_class. Separate op_class, not a reuse of `retain`: scores are
per-op_class, so evolution traffic must not vote in the ranking that picks
hindsight's memory models.

## Non-goals (v1)

- Multi-provider fan-out (NIM only; the provider abstraction exists, the
  catalog sync is per-provider, but only NIM ships enabled).
- TUI/dashboard (FCM's core surface — ours is headless; `mesh/status` is JSON).
- Token accounting / billing.

## Ops

- `scripts/install-launchd.sh` — daemon + daily discovery job.
- Runs on `127.0.0.1:8002` (nim-proxy keeps `:8001` until cutover).
- Logs: `~/.model-mesh/mesh.log` (daemon), audit JSONL `~/.model-mesh/audit/`.
- State: everything under `~/.model-mesh/` — copy the dir, keep the history
  (portable to other machines by design).
