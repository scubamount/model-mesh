# model-mesh

**One local OpenAI-compatible endpoint that never dies.**

A health-aware model router for NVIDIA NIM (and any OpenAI-compatible upstream).
Replaces a static proxy + hourly auto-ranker with request-time routing built on
a persistent model index.

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
┌──────────────────────────────────────────────┐
│ model-mesh daemon (localhost only)           │
│                                              │
│  raw candidate pool → route() ranks once     │
│  ARM 1: dial ranked[:max_attempts],          │
│         budget-bounded (total_budget_s)      │
│  ARM 2: live re-probe (time-boxed 25%),      │
│         re-dial what answers                 │
│  ARM 3: sweep every remaining candidate      │
│         incl. floors-excluded, deduped       │
│  only then 503 (with models_tried evidence)  │
│                                              │
│  every real request updates the index        │
│  (latency, status) — traffic IS telemetry    │
└──────────────────────────────────────────────┘
   │ upstream: NIM cloud (more providers later)
   ▼
┌──────────────────────────────────────────────┐
│ model index (SQLite mesh.db, state dir)      │
│  catalog snapshots · probe+request history   │
│  latency/jitter/spike/success · breakers     │
│  · EOL log                                   │
└──────────────────────────────────────────────┘
```

### The model index (the durable part)

SQLite `mesh.db` in the mesh's **state directory** — the location ships as
`db_path` in `model_mesh/config.py` and every artifact below (db, daemon log,
discovery log, audit JSONL) lives under that same one directory. Tables:

- `models` — every model ever seen in a catalog sync: id, provider,
  first_seen, last_seen, eol_at (when it vanished or started returning
  410/404). Models are never deleted — EOL is data.
- `samples` — every probe AND every real request: model, ts, op_class,
  latency_ms, status, payload_chars. Real traffic is the best probe; we log
  it for free.
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

Two rules locked in by a measured failure (2026-08-09: all 23 scored retain
models fell inside [49.9, 50.0] under a blended stability float, and ordering
degenerated to the alphabetical tiebreak):

- There is deliberately **no aggregate score** — `Score` carries p95, jitter,
  spike rate and success rate and nothing sortable. Anything that reduces them
  to one number re-creates both failures: it ranks availability while looking
  like quality, and loses resolution the moment a prior or cap stabilizes it.
- **Tell worth keeping: alphabetical order in `ranking_all` means nothing is
  actually scored.**

**Latency is an overload signal, not a quality measure.** On a shared free
endpoint a slow response is not a property of the model, it is other people
using it. Past `overload_p95_ms` (20s) a model drops below everything healthy
whatever its tier, and is promoted back for free when its p95 recovers.
Nothing marks or unmarks it.

**Quality tier** (1–5) is derived from the model id: parameter count, else a
size adjective, else mid. That deliberately contradicts this codebase's rule
against guessing capability from names, and the tension resolves on *which*
question. "Can this model do the job?" is measurable — the fidelity gate
measures it, never guesses. "Is this model stronger?" is not measurable by any
probe here. Tiers derive from the live catalog (new models tier on arrival, no
edit), are a prior rather than a verdict, and are overridable per model id via
`router.tier_overrides`. Unknown sizes tier **mid, never bottom**.

### Eligibility floors (who may serve an op_class at all)

Before ranking, a model must pass per-op_class floors — all measured, none
guessed:

- **Fidelity gate**: a 200 whose body violates the op_class JSON contract is
  recorded as `fidelity-fail` (a FAILED sample, never a success) and never
  returned to the client. Two unrebutted violations drop the model from that
  op_class (`fidelity_fails_for_floor`) until one success intervenes or the
  weekly recheck elapses.
- **Capability reject**: HTTP 400/413/422 means the provider parsed and
  refused the request shape — deterministic, retry pointless. Excluded from
  the op_class after two occurrences; the model stays eligible for others.
- **Success floor**: below `min_success_rate` (0.5) over `min_samples_for_floor`
  (4) in-window samples, a model does not serve that op_class.
- **Latency floor**: measured p95 above `max_p95_ms_for_eligibility` (75s)
  excludes the model — an attempt must be able to run twice inside the budget.

Floors are caution, not verdicts: when they exclude everything, the sweep arm
(arm 3 below) still tries them, because a total miss means the caution already
bought nothing.

### Request-time routing: three arms, one budget

Per request (`total_budget_s` = 280s whole-cascade deadline; per-attempt
timeout shrinks to fit what's left; the budget ends the request, never an
attempt count):

1. `route()` ranks the RAW candidate pool once — app.py hands it over
   unfiltered and uncapped. A pre-router slice or pre-ranking is what made
   "no healthy candidates" reachable with zero dials (fixed 2026-08-24);
   ranking happens exactly once, inside the router.
2. **Arm 1 — cascade**: dial ranked candidates in order (`max_attempts`, 8).
   Transient failures (429/5xx/timeout/malformed) trip the breaker and move
   on; fidelity-fails never end the cascade with a bad answer in hand.
3. **Arm 2 — live re-probe**: if everything missed, ping the pool now (the
   index may be stale; `down` AND `auth` models are re-testable — only
   `gone` is terminal). Re-dial whatever answers. Time-boxed to at most 25%
   of remaining budget so hung probes can't starve arm 3.
4. **Arm 3 — last-resort sweep**: dial every remaining candidate directly —
   ranked tail first, then models the eligibility floors excluded (a total
   miss means trying beats refusing), deduped against every model this
   request already dialed, skipping terminal ghosts, capped at
   `sweep_max_models` (12). This is what makes "no healthy candidates"
   require a whole-pool failure inside one request: hindsight retain must
   fail only when every live NIM model verifiably failed within that
   request.
5. All arms missed → `503` with `models_tried` + per-model failure reasons —
   an honest verdict backed by evidence, not a guess.

Status-code discipline:

| code | meaning | action |
|---|---|---|
| 429 / 5xx / timeout | transient overload | cascade onward |
| 401 / 403 | auth | separate state, expires; never poisons the breaker |
| 400 / 413 / 422 | capability reject | excluded from that op_class |
| 404 / 410 | **gone** | `eol_at` marked immediately — EOL detection at request time |

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
  The upstream listing is the candidate universe, not proof of service:
  NIM keeps retired models listed while their endpoint hard-404s (measured
  2026-08-24: 102 listed vs ~67 serving; 6/6 sampled ghost ids returned
  instant 404s). Request-time evidence and `eol_at` are the truth.
- **Representative probes**: probe payload is padded to the op_class's real
  size (retain ≈ 12k chars). A probe that can't fail like production fails
  is worse than no probe.
- **Background probe cadence** only for models with no recent traffic —
  traffic is telemetry; don't burn quota double-checking it.

**Probe economy.** Probing is not free: every probe spends quota on a shared
free endpoint. Two bounds, both derived from what probing is actually *for* —
detecting overload and disappearance, the only two things that change
unpredictably:

- `discovery.probe_top_n` (default 6) — probe only the best few candidates
  per alias, ordered by quality tier. `null` probes the whole pool.
- `discovery.max_probes_per_pass` (default 25) — hard ceiling per pass.
- `DORMANT_AFTER_S` (7 days) — no success across a full window ⇒ skip
  probing that model (it is almost certainly retired; measured mid-episode
  2026-08-24: all 35 retired ids were still catalog-listed). The skip is
  window-keyed: one recheck probe still runs per window, and any real
  success clears dormancy immediately — no separate resurrection path.

### API surface

- `POST /v1/chat/completions` — OpenAI-compatible; `model` = alias or raw id.
  On total failure: 503 with `models_tried` (per-attempt model/status/latency
  evidence) and `reprobed`.
- `GET /v1/models` — upstream catalog passthrough + aliases.
- `GET /mesh/status` — per alias: pool size, ranking, **`ranking_all`** (full
  ordering — a consumer must be able to tell "ranked 13th" from "not ranked
  at all"), `scores` (p95_ms, success_rate, n — deliberately no aggregate),
  `rank_inputs` (quality tier + availability bucket per ranked model), breaker
  states. The ordering is explainable from the response alone.
- `GET /mesh/models` — the live upstream catalog (`live`, `count`) plus current
  `breaker` state per model. Per-model measurements live on `/mesh/status`.
- `POST /mesh/probe` — force a discovery pass now; returns the same
  new/eol/probed report as the daily job. Discovers and fidelity-tests models;
  it does not score them — scores backfill from real traffic.
- `GET /mesh/discovery` — recent discovery runs: timestamp, duration, and the
  `new`/`eol`/`returned`/`probed` sets per pass. The churn record the whole
  probe design exists for.
- `GET /health` — **deep** health: catalog reachable AND ≥1 healthy model per
  configured alias. Returns 503 otherwise. (The predecessor's `/health`
  could not fail while retain was down; this one can.)

### Config

The optional `config.yaml` (same state directory) is currently absent on this
machine: every default below ships in `model_mesh/config.py` and the daemon
boots with no config file. Aliases carry op_class + include/exclude patterns
only (the pool is discovered, not enumerated); secrets stay in env, never in
the DB or config.

Router knobs worth knowing:

| key | default | meaning |
|---|---|---|
| `router.max_attempts` | `8` | ranked candidates arm 1 dials before arm 2 |
| `router.reprobe_top_n` | `4` | passes of live probing in arm 2 |
| `router.sweep_on_total_miss` | `true` | enable arm 3 |
| `router.sweep_max_models` | `12` | max direct dials in arm 3 |
| `router.total_budget_s` | `280` | whole-cascade deadline (< hindsight's 300s) |
| `router.request_timeout_s` | `90` | price of one hung attempt |
| `router.overload_p95_ms` | `20000` | p95 at/above this = overloaded → demoted |
| `router.tier_overrides` | `{}` | `{model_id: 1..5}` when the size heuristic misjudges |
| `router.min_success_rate` | `0.5` | eligibility floor (with `min_samples_for_floor`=4) |
| `router.fidelity_fails_for_floor` | `2` | unrebutted contract violations → excluded |

## Compatibility

Drop-in for the nim-proxy contract hindsight speaks: `openai/auto/retain`,
`auto/consolidation`, `auto/reflect`, `auto/evolve` aliases on an
OpenAI-compatible localhost port.

**Cutover is done.** Hindsight's profile points every LLM base URL at
`http://127.0.0.1:8002/v1` — retain, reflect, consolidation and evolution
traffic all flow through the mesh, with a litellm fallback tier to a pinned
`openai/gpt-oss-20b` on the same port.

`auto/evolve` serves DSPy-style skill evolution on its own `evolve` op_class —
scores are per-op_class, so evolution traffic must not vote in the ranking that
picks the memory models.

## Non-goals (v1)

- Multi-provider fan-out (NIM only by design; the provider abstraction exists,
  catalog sync is per-provider).
- TUI/dashboard — headless; `/mesh/status` is JSON.
- Token accounting / billing.

## Ops

- `scripts/install-launchd.sh` — installs the daemon + daily discovery job as
  launchd user agents (macOS). **Idempotent and adoptive:** if an install
  already exists it reuses that machine's label prefix and state directory, so
  a reinstall updates your install rather than forking a second daemon onto the
  same port and stranding the old `mesh.db`. A fresh machine gets `local.*` and
  `~/.model-mesh`. Overrides, all optional:

  | var | default | what it sets |
  |---|---|---|
  | `MESH_HOME` | adopted, else `~/.model-mesh` | state dir: db, config, `.env`, logs, audit |
  | `MESH_LABEL_PREFIX` | adopted, else `local` | reverse-DNS prefix for both launchd labels |
  | `MESH_HOST` / `MESH_PORT` | `127.0.0.1` / `8002` | seeds `listen` in a new `config.yaml` |
  | `MESH_PYTHON` | first `python3.12+` on `PATH` | interpreter used to build the venv |
  | `MESH_DISCOVER_HOUR` / `_MIN` | `6` / `15` | daily discovery time |

  Restart after code changes:
  `launchctl kickstart -k gui/$(id -u)/<prefix>.model-mesh`.
- **The listen address lives in `config.yaml`, not in the plist.** The agent runs
  the `model-mesh` console entrypoint, which serves whatever `listen` says;
  `MESH_HOST`/`MESH_PORT` only seed that file on first install and never clobber
  an existing one. Change the address by editing config and restarting.
- **One state directory.** `$MESH_HOME` (default `~/.model-mesh`) holds the db,
  `config.yaml`, the key fallback `.env`, logs and the audit trail — they derive
  from one root, so nothing can relocate half an install.
- **Provider key.** Never written to a plist or a repo file. The daemon reads
  `$NVIDIA_API_KEY` at *call* time and falls back to `$MESH_HOME/.env` (lines of
  `ENV_VAR=value`, mode 0600). `launchctl setenv` does not survive a restart, so
  the file is the durable option; override its path with
  `MODEL_MESH_KEY_FALLBACK_FILE`.
- Tests: `.venv/bin/python -m pytest` (196 tests; includes a sabotage matrix
  proving each routing guarantee fails loudly when its mechanism is removed).
- **Backups.** `mesh.db` is *learned* state: sample history, breaker states and
  EOL marks accumulated from real traffic, reconstructible only by re-living
  that time. It is not in git and nothing here regenerates it — back it up with
  sqlite `.backup` (WAL-safe) on a schedule. A running daemon is not a backup:
  uptime reads as safety, which is why this had none for months.

## License

Apache License 2.0 — see [LICENSE](LICENSE).


