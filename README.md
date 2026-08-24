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
│    (quality tier + availability, breaker)   │
│  try #1 ──fail──▶ try #2 ──fail──▶ try #3   │
│    │                                        │
│    ├── all failed? LIVE RE-PROBE the pool,  │
│    │   rebuild ranking, one more cascade;   │
│    └── still failed? SWEEP the rest of the  │
│        ranked pool (deduped, budget-bound); │
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
│  latency/jitter/spike/success · breakers    │
│  · EOL log                                  │
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

### Ranking: quality orders, availability gates

The mesh's job is uptime for hindsight on free NIM endpoints, where models go
overloaded or vanish unpredictably — that churn is the only reason to probe
anything. Two properties matter, on two different timescales, and blending them
into a single number is what broke:

| | timescale | measurable? | role |
|---|---|---|---|
| **quality** | slow-moving, effectively static | not by any probe here | **orders** candidates |
| **availability** | hour to hour | only by probing | **gates** them |

```
rank_key = (availability_bucket, -quality_tier, p95_ms, model_id)
             healthy < overloaded < unknown < failing
```

Take the best model that is actually up right now. Deterministic — same
evidence, same order, every call. `unknown` outranks `failing`: a model with no
evidence is a maybe, and a maybe beats one measured broken.

**Latency is an overload signal, not a quality measure.** On a shared free
endpoint a slow response is not a property of the model, it is other people
using it — and the model answering in 44s is the one about to start timing out.
Past `overload_p95_ms` (20s) a model drops below everything healthy whatever its
tier, and is promoted back for free when its measured p95 recovers. Nothing
marks or unmarks it.

**Quality tier** (1–5) is derived from the model id: parameter count, else a
size adjective, else mid. That deliberately contradicts this codebase's rule
against guessing capability from names, and the tension resolves on *which*
question. "Can this model do the job?" is measurable, so it is measured by the
fidelity probe and never guessed. "Is this model stronger?" is not measurable by
any probe here — a latency probe cannot tell a 550b from a 1b except by the 1b
winning, which is the exact inversion being fixed. Tiers are derived from the
live catalog (new models tier on arrival, no edit), are a prior rather than a
verdict (a top-tier model that is down loses to a healthy lower tier), and are
overridable per model id via `router.tier_overrides`.

Unknown sizes tier **mid, never bottom** — `glm-5.2` and `minimax-m3` publish no
parameter count and are frontier models; tiering them last would quietly exclude
exactly what we want.

<details>
<summary>Superseded: the blended stability score (removed 2026-08-09)</summary>

Ranking was one float over a 24h window:

```
score = 0.30 * p95_latency + 0.30 * jitter + 0.20 * spike_rate + 0.20 * success_rate
```

Every term measures availability; none measures quality, so a 1b model outranked
a 120b whenever it answered faster — the wrong objective for a memory backbone.

It had also lost all resolution. Measured live on `auto/retain`: 23 scored
models spanning p95 0.4s–44.3s, **every score inside [49.9, 50.0]**, because
confidence shrinkage pinned thin evidence just below a neutral prior and floored
proven models at it. Ordering fell through to the alphabetical tiebreak, and
`gemma-4-31b-it` held rank 0 at p95 44.3s with jitter 4.77 and a falling success
rate — visibly degrading, structurally unbeatable.

A randomized exploration arm (`explore_rate`, 10%) was tried first and made the
ranking corrigible without making it correct: challengers still gained evidence
one sample at a time and stayed capped below the prior until n=8. It is gone.

**Tell worth keeping: alphabetical order in `ranking_all` means nothing is
actually scored.**
</details>

`samples` still records p95/jitter/spike/success per (model, op_class) — those
drive the availability bucket, the eligibility floors and the breakers. What
changed is that they no longer pretend to rank quality.

`Score` carries those measurements and **no aggregate**: there is deliberately
no single number to sort by. The blended float was deleted from the code on
2026-08-09 (not just from the ranking), so `/mesh/status` no longer returns a
`score` field per model — read `p95_ms`, `success_rate`, `n`, and `rank_inputs`
instead. Anything reducing those to one number re-creates both failures above:
it will rank availability while looking like quality, and it will lose
resolution as soon as a prior or a cap is added to stabilize it.

### Request-time routing (the part nim-proxy never had)

Per request:

1. Resolve alias (`auto/retain` → ranked candidates for op_class `retain`).
2. Skip models whose breaker is open (down/gone/auth) or cooling down.
3. Try candidate #1. On transient failure (429, 5xx, timeout, malformed
   JSON) → breaker counts it and the cascade moves to the next candidate.
   A 200 whose body violates the op_class JSON contract is a
   `fidelity-fail`: recorded as a failed sample (never a success), the
   response is not returned to the client, the cascade continues, and two
   unrebutted violations drop the model from that op_class until one
   success intervenes or the weekly recheck elapses. The cascade ends on
   TIME (`total_budget_s`), never on an attempt count.
4. **All tried and failed → live re-probe**: hit `/v1/models` + 1-token pings
   on the pool, rebuild ranking from fresh data, run ONE more cascade.
   This is the "if nothing works, ping again and find what's actually up
   right now" arm — it converts a stale-index total-miss into a recovery
   instead of an outage.
5. **Still nothing → last-resort sweep**: walk every remaining candidate —
   the ranked tail first, then models the eligibility floors excluded
   (when a whole-pool episode makes `ranked()` empty, trying beats
   refusing) — dialing each directly with the real body, no probe
   round-trip; skip models this request already dialed and terminal `gone`
   ghosts; stop when one serves or `total_budget_s` dies. This is what
   makes "no healthy candidates" require a whole-pool failure inside a
   single request: hindsight retain/consolidation must fail only when every
   live NIM model actually failed within that request. (`sweep_on_total_miss`,
   capped at `sweep_max_models` dials; both mirrored in `config.py`.)
6. Even the sweep missed → `503` with `models_tried` + per-model failure
   reasons.

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

**Probe economy.** Probing is not free: every probe is a real request against a
shared free endpoint, and the quota it spends is the quota a newly-released
model needs. Two bounds, both from what probing is actually *for* — detecting
overload and disappearance, the only two things that change unpredictably:

- `discovery.probe_top_n` (default 6) — probe only the best few candidates per
  alias, ordered by quality tier. The cascade tries at most 3 models, so the
  health of the 20th-best is worth nothing while costing a real request.
  Set to `null` to probe the whole pool.
- `DORMANT_AFTER_S` (7 days) — a model with no successful request in a week is
  treated as retired even while the catalog still lists it. NIM lists models it
  will not serve: measured 17 of 18 models retired on request-time 404s were
  still in the catalog, so catalog presence cannot answer "is this real". It is
  a skip, not an EOL — one probe still runs per window, and a single success
  clears dormancy with no separate resurrection path to maintain.

### API surface

- `POST /v1/chat/completions` — OpenAI-compatible; `model` = alias or raw id.
- `GET /v1/models` — upstream catalog passthrough + aliases.
- `GET /mesh/status` — ranking, breaker states, last sync, per-alias health.
  Includes `rank_inputs` (quality tier + availability bucket per ranked model),
  so the ordering is explainable from the response alone — the previous ranking
  failure was invisible precisely because status showed an order with no way to
  see the reason for it.
- `GET /mesh/models` — the live upstream catalog (`live`, `count`) plus current
  `breaker` state per model. Per-model measurements and ranking live on
  `/mesh/status`, not here.
- `POST /mesh/probe` — force a re-probe (used by cron; rate-limited).
- `GET /mesh/discovery` — recent discovery `runs`: timestamp, duration, and the
  `new` / `eol` / `returned` / `probed` sets per pass. This is the churn record
  the whole probe design exists for — it shows models arriving and vanishing
  from the free catalog without any log grepping.
- `GET /health` — **deep** health: catalog reachable AND ≥1 healthy model per
  configured alias. Returns 503 otherwise. (The predecessor's `/health`
  could not fail while retain was down; this one can.)

### Config

`~/.model-mesh/config.yaml` — providers (base URL + key env var), aliases →
op_class + candidate filters (include/exclude patterns — the pool is
discovered, not enumerated), breaker/probe tunables. Secrets stay in env,
never in the DB or config. Defaults ship in `model_mesh/config.py`, so the
daemon boots with no config file at all — and currently does: there is no
`config.yaml` on this machine, every value below is the shipped default.

Ranking/probe knobs worth knowing:

| key | default | meaning |
|---|---|---|
| `router.overload_p95_ms` | `20000` | above this p95, a model is treated as overloaded and demoted below everything healthy |
| `router.tier_overrides` | `{}` | `{model_id: 1..5}` when the parameter-count heuristic misjudges a model |
| `discovery.probe_top_n` | `6` | probe only the best N candidates per alias; `null` probes the whole pool |
| `discovery.max_probes_per_pass` | `25` | hard ceiling on probes in one pass |

## Compatibility

Drop-in for the nim-proxy contract hindsight speaks: `openai/auto/retain`,
`auto/consolidation`, `auto/reflect` aliases on an OpenAI-compatible localhost
port.

**Cutover is done.** Hindsight's profile (`~/.hindsight/profiles/hermes.env`)
points every LLM base URL at `http://127.0.0.1:8002/v1` — retain, reflect and
consolidation, each with a litellm fallback to a pinned `openai/gpt-oss-20b` on
the same port. nim-proxy on `:8001` no longer serves this path.

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

- `scripts/install-launchd.sh` — daemon + daily discovery job. launchd labels:
  `com.scubamount.model-mesh` (daemon) and `com.scubamount.model-mesh-discover`.
- Runs on `127.0.0.1:8002`.
- Logs: `~/.model-mesh/mesh.log` (daemon), audit JSONL `~/.model-mesh/audit/`.
- State: everything under `~/.model-mesh/` — copy the dir, keep the history
  (portable to other machines by design).
- **Backups.** `mesh.db` is *learned* state: sample history, breaker states and
  EOL marks accumulated from real traffic, reconstructible only by re-living
  that time. It is not in git and nothing here regenerates it. It is backed up
  by `scripts/backup-durable-state.sh` in `hermes-agent-patches` (sqlite
  `.backup`, WAL-safe — a plain `cp` of a WAL-mode db in use can capture a torn
  page), which runs as an `/aaa` step and is gated by the
  `durable-state-backed-up` invariant on backup *freshness*, not on the script
  existing. A running daemon is not a backup: uptime reads as safety, which is
  why this had none for months.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
