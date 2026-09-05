# model-mesh

**One local OpenAI-compatible endpoint that never dies.**

A health-aware model router for NVIDIA NIM (and any OpenAI-compatible upstream).
Point a client at one localhost URL and ask for `auto/retain`; the mesh picks the
best model that is actually up right now, cascades through the rest when that
one fails, and only refuses once every live model has verifiably failed inside
your request.

Built for **free shared inference endpoints**, where models go overloaded for
minutes, disappear for days, and get retired without warning while still being
listed in the catalog — and where every request shares one rate-limited API
key. Whole-pool degradation is the steady state here, not an incident: the
design goal is that the mesh keeps serving *something* through it, preferring
the best model that is up but never requiring any particular one.

## Quickstart

```sh
git clone <your-fork> model-mesh && cd model-mesh
sh scripts/install-launchd.sh          # macOS: daemon + 6-hourly discovery job
echo 'NVIDIA_API_KEY=<your-api-key>' >> ~/.model-mesh/.env
chmod 600 ~/.model-mesh/.env
curl -s http://127.0.0.1:8002/health
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8002/v1` and use
`auto/retain` (or any configured alias) as the model name. No config file is
required — every default ships in `model_mesh/config.py`.

Not on macOS, or prefer to run it yourself:

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e .
MESH_HOME=~/.model-mesh .venv/bin/model-mesh   # serves config's `listen`
```

## Why (the failure that motivated this)

The predecessor (a static proxy + hourly ranker) failed in a specific,
instructive way:

1. `meta/llama-4-maverick` hit EOL (`410 Gone`) on 2026-07-27. The static
   candidate list silently shrank to one model.
2. The fallback tier was hardcoded to the *same* model the alias resolved to —
   primary and fallback were identical. Zero redundancy.
3. The hourly ranker probed with a ~50-char toy prompt; real payloads are ~14k
   chars. The probe ranked gpt-oss-20b "fastest" (~2s) while it took 27.9s on
   real work — starving the client into 300s timeouts.
4. Every health check that should have caught this could not fail: `/health`
   only checked its database, the ranker only checked its own toy probe.

Memory writes silently failed for a week. **Every design decision below traces
back to one of those four failures.**

## Architecture

```
client (any OpenAI-compatible caller)
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

SQLite `mesh.db` in the mesh's **state directory** (`$MESH_HOME`, default
`~/.model-mesh`). Every artifact lives under that one root — db, `config.yaml`,
the key-fallback `.env`, daemon log, discovery log, audit JSONL — so one env var
relocates the whole install and nothing can move half of it. Tables:

- `models` — every model ever seen in a catalog sync: id, provider,
  first_seen, last_seen, eol_at (when it vanished or started returning
  410/404). Models are never deleted — EOL is data.
- `samples` — every probe AND every real request: model, ts, op_class,
  latency_ms, status, payload_chars. Real traffic is the best probe; we log
  it for free.
- `breaker` — current circuit state per model (healthy / down / recovering /
  auth / gone), consecutive fails, cooldown_until.

### Ranking: quality orders, availability gates

The mesh's job is uptime on free shared endpoints, where models go overloaded or
vanish unpredictably — that churn is the only reason to probe anything. Two
properties matter, on two different timescales, and blending them into a single
number is what broke:

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

**Evidence is recency-capped.** `score()` reads the newest `SCORE_RECENT_N`
(20) samples inside a 24h staleness bound — a count window, not a time
window. Overload flips within minutes on these endpoints; a 24h average
buried a recovery (or a fresh collapse) under yesterday's hundred samples
for hours. A count window self-adapts to traffic rate and has no
empty-window edge case; samples older than 24h still say nothing about now
and score `None` (bucket `unknown`).

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
  (4) recent samples, a model does not serve that op_class.
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
   require a whole-pool failure inside one request: a request may fail only
   when every live model verifiably failed within it.
5. All arms missed → `503` with `models_tried` + per-model failure reasons —
   an honest verdict backed by evidence, not a guess.

Status-code discipline:

| code | meaning | action |
|---|---|---|
| 429 | shared-key throttle | cascade onward + **provider-wide pause** (Retry-After honored, capped 60s; siblings skip, no sample recorded) |
| 5xx / timeout | transient overload | cascade onward |
| 401 / 403 | auth | separate state, expires; never poisons the breaker |
| 400 / 413 / 422 | capability reject | excluded from that op_class |
| 404 / 410 | **gone** | `eol_at` marked immediately — EOL detection at request time |

### Circuit breaker

Per model: `healthy → down` after N consecutive fails (default 3);
`down → recovering` after cooldown (default 30s); `recovering` admits one
real request — success closes the circuit, failure reopens with doubled
cooldown (cap 5 min). The 30s→300s ladder is adopted from
free-coding-models' field-proven breaker against the same NIM endpoints:
overload flips within minutes, and the previous 120s→30min ladder kept a
recovered model benched long after a two-minute episode ended.

### Provider-wide 429 pause

A 429 from NIM is a statement about the **shared API key** (40 req/min),
not about the model that happened to answer it. Cascading onward through
the remaining candidates used to machine-gun the same throttled key — each
sibling burned budget on a guaranteed 429 and polluted its own samples with
a failure we inflicted. Now a 429 arms a router-wide pause window:

- `Retry-After` is honored when present (capped at `provider_pause_max_s`,
  60s), else `provider_pause_default_s` (5s); windows only ever extend
  (max-of-windows), never shorten.
- Inside the window, a dial whose budget cannot outlive the pause returns
  `skipped-provider-pause` **without an upstream call and without recording
  a sample** — a self-inflicted throttle hit is evidence about our pacing,
  not about the model. A dial whose budget covers the pause waits it out
  and proceeds with the remainder.

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
- `GET /health` — **deep** health = *can the mesh serve*: `healthy` when ≥1
  eligible model per alias; `degraded` (still 200) when the floors admit
  nobody but the pool produced an OK inside 30 min — whole-pool overload is
  NIM's steady state and the sweep backstop still serves through it; 503
  only when nothing is admitted AND nothing recently served. (The
  predecessor's `/health` could not fail while retain was down; the version
  after that failed while retain was *serving*. This one tracks servability.)

### Config

Every default ships in `model_mesh/config.py`; the daemon boots with no config
file at all. An optional `config.yaml` in the state directory overrides any
subset (deep-merged, so you only write what you change). Aliases carry op_class +
include/exclude patterns only — the pool is discovered, not enumerated. Secrets
stay in env or the key-fallback file, never in the DB or config.

```yaml
# ~/.model-mesh/config.yaml — everything here is optional
listen:
  host: 127.0.0.1
  port: 8002
provider:
  name: nim
  base_url: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_API_KEY
```

`listen` is the **only** place the serving address is defined — the launchd
agent runs the `model-mesh` entrypoint and passes no host/port, so editing this
file and restarting is how you move the port.

Router knobs worth knowing:

| key | default | meaning |
|---|---|---|
| `router.max_attempts` | `8` | ranked candidates arm 1 dials before arm 2 |
| `router.reprobe_top_n` | `4` | passes of live probing in arm 2 |
| `router.sweep_on_total_miss` | `true` | enable arm 3 |
| `router.sweep_max_models` | `12` | max direct dials in arm 3 |
| `router.total_budget_s` | `280` | whole-cascade deadline; keep under the client's own timeout |
| `router.request_timeout_s` | `90` | price of one hung attempt |
| `router.request_timeout_s_by_op_class` | `{consolidation: 135}` | per-op_class override; some ops are legitimately slower |
| `router.probe_timeout_s` | `45` | probe attempt ceiling |
| `router.probe_timeout_s_by_op_class` | `{consolidation: 100}` | per-op_class probe override |
| `router.overload_p95_ms` | `20000` | p95 at/above this = overloaded → demoted |
| `router.max_p95_ms_for_eligibility` | `75000` | measured p95 above this excludes the model |
| `router.tier_overrides` | `{}` | `{model_id: 1..5}` when the size heuristic misjudges |
| `router.min_success_rate` | `0.5` | eligibility floor (with `min_samples_for_floor`=4) |
| `router.fidelity_fails_for_floor` | `2` | unrebutted contract violations → excluded |
| `router.breaker_threshold` | `3` | consecutive fails before a model opens its breaker |
| `router.breaker_cooldown_s` | `30` | first cooldown; doubles to `breaker_cooldown_max_s` (300) |
| `router.provider_pause_default_s` | `5` | provider-wide pause when a 429 has no `Retry-After` |
| `router.provider_pause_max_s` | `60` | cap on any 429 pause window, however large the header |
| `discovery.probe_top_n` | `6` | probe only the best N candidates per alias; `null` = whole pool |
| `discovery.max_probes_per_pass` | `25` | hard ceiling per discovery pass |

Keep `2 × request_timeout_s < total_budget_s` so one slow model cannot consume
the whole cascade.

## Compatibility

Any OpenAI-compatible client works unmodified — point its base URL at
`http://127.0.0.1:8002/v1` and use an alias as the model name. The default
aliases are `auto/retain`, `auto/consolidation`, `auto/reflect` and
`auto/evolve`; raw model ids also pass through, with breaker protection and
telemetry but no cascade.

Scores are **per-op_class**, which is why the aliases are separate: one workload's
traffic must not vote in the ranking that serves another. `auto/reflect`
deliberately maps to the `retain` op_class (same contract, same evidence pool);
`auto/evolve` gets its own.

## Non-goals (v1)

- Multi-provider fan-out (NIM only by design; the provider abstraction exists,
  catalog sync is per-provider).
- TUI/dashboard — headless; `/mesh/status` is JSON.
- Token accounting / billing.

## Ops

- `scripts/install-launchd.sh` — installs the daemon + 6-hourly discovery job as
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
  | `MESH_DISCOVER_INTERVAL_S` | `21600` (6h) | seconds between discovery passes; must stay well under `SCORE_WINDOW_S * REFRESH_MARGIN` (19.2h) or a traffic-free lane's evidence expires |

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
- Tests: `.venv/bin/python -m pytest` (284 tests; includes a sabotage matrix
  proving each routing guarantee fails loudly when its mechanism is removed).
- **Backups.** `mesh.db` is *learned* state: sample history, breaker states and
  EOL marks accumulated from real traffic, reconstructible only by re-living
  that time. It is not in git and nothing here regenerates it — back it up with
  sqlite `.backup` (WAL-safe) on a schedule. A running daemon is not a backup:
  uptime reads as safety, which is why this had none for months.

## Troubleshooting

**`/health` returns 503.** 503 means *measured inability to serve*: attempts
inside the last 30 min, none succeeded, and no eligible candidate. That is
provider-wide failure or a dead key — check `models_tried` on a real request
and `GET /mesh/status` for breaker states. A brand-new install with an empty
index reports `degraded` ("idle, servability unknown") until evidence
arrives; run `POST /mesh/probe` to seed it.

**`/health` says `degraded`.** Two benign shapes, named in the body:
`sweep backstop serving` = the floors admit nobody right now but something
in the pool produced an OK inside 30 min — whole-pool overload is the steady
state on free endpoints and the sweep arm still serves through it;
`idle, servability unknown` = the alias simply had no traffic in the window
(low-cadence ops like evolve fire hours apart) — absence of traffic is not
failure. Neither needs action.

**Every request 401s.** The daemon does not inherit your shell environment.
Put the key in `$MESH_HOME/.env` (`NVIDIA_API_KEY=...`, mode 0600) rather than
relying on `launchctl setenv`, which does not survive a restart. The key is read
at call time, so no restart is needed after writing the file.

**Ranking looks alphabetical.** That is the tell that nothing is actually
scored — check `rank_inputs` in `/mesh/status`. Models with `bucket: unknown`
have no recent evidence; they sort behind everything measured healthy.

**A model you expect is missing from `ranking_all`.** Either an eligibility
floor excluded it (check `scores` for `success_rate` and `n`) or its breaker is
`gone` — a 404/410 at request time marks `eol_at` permanently. `/mesh/models`
shows breaker state per model.

**Requests are slow but succeed.** Expected on free shared endpoints: p95 above
`overload_p95_ms` (20s) demotes a model below everything healthy, and it is
promoted back for free when its p95 recovers. Nothing marks or unmarks it. If
*everything* is overloaded, the mesh is doing its job — serving the least-bad
model that answers.

**Reinstalling.** `install-launchd.sh` adopts an existing install (label prefix
and state dir) rather than creating a second one. To deliberately run a second
independent instance, set both `MESH_HOME` and `MESH_LABEL_PREFIX`, and a
different `MESH_PORT`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).


