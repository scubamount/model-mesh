"""model-mesh: request-time router.

The part nim-proxy never had. Per request: resolve alias -> ranked candidates
-> cascade with circuit breaking -> live re-probe on total miss -> 503 with
evidence only after everything (including a fresh probe pass) failed.

Failure taxonomy (each class routes differently — collapsing them is how the
predecessor silently lost redundancy):

  quota      429                                   -> per-model quota ladder
             (its OWN timer, never breaker-counted) + provider-wide pause:
             the throttle is on the shared key — see dial()
  transient  5xx / timeout / malformed-JSON        -> breaker counts, cascade
  auth       401 / 403                             -> mark 'auth', skip provider, NO breaker poison
  gone       404 / 410                             -> index.mark_gone NOW, cascade
"""

from __future__ import annotations

import errno
import json
import logging
import socket
import threading
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from .index import FIDELITY_FAIL_STATUS, Index, OK
from .opclass import check_fidelity, parse_content
from .quality import rank_key

logger = logging.getLogger("model_mesh.router")

TRANSIENT_CODES = {429, 500, 502, 503, 504}
AUTH_CODES = {401, 403}
GONE_CODES = {404, 410}
# Deterministic request rejection: the payload/params are wrong for THIS model,
# so retrying it with the same body is guaranteed to fail again. Observed live
# 2026-08-04: nemotron-super-49b returned http-400 on 4 of 7 retain calls (large
# structured payload) while serving consolidation at 100%. Counted as a hard
# failure for the op_class it was rejected on — never retried inside a cascade.
REJECT_CODES = {400, 413, 422}
# Same set as status strings, derived so the two can never drift apart.
REJECT_STATUS_NAMES = {f"http-{c}" for c in REJECT_CODES}

# Eligibility ceiling as a fraction of an op_class's request budget. 75/90 is
# the measured pair the scalar floor came from (2026-08-07): a ceiling AT the
# budget admits a model that times out on ~5% of calls by construction, and
# each of those is scored as a failure against a model that actually works.
_LATENCY_CEILING_FRACTION = 75.0 / 90.0


@dataclass
class RouterConfig:
    breaker_threshold: int = 3          # consecutive fails -> down
    breaker_cooldown_s: float = 30.0    # first cooldown (FCM-proven ladder)
    breaker_cooldown_max_s: float = 300.0
    # Provider-wide 429 pause: NIM throttles the shared key, not one model.
    # Default window when the 429 carries no Retry-After; cap bounds a
    # hostile/buggy header so one response can't bench the whole mesh.
    provider_pause_default_s: float = 5.0
    provider_pause_max_s: float = 60.0
    # Per-model quota cooldown ladder, a SEPARATE timer from the failure
    # breaker. Adopted from CLIProxyAPI (sdk/cliproxy/auth/conductor_refresh.go
    # quotaBackoffBase=1s / quotaBackoffMax=30m / minQuotaCooldownFloor=10s;
    # ladder in conductor_cooldown.go nextQuotaCooldown): a 429 without
    # Retry-After benches the model for base * 2**level, level climbs per
    # 429 that lands after the previous window expired, capped at max; a 429
    # WITH Retry-After benches for max(Retry-After, floor) and leaves the
    # level alone. One success resets the level. A 429 is "slow down", not
    # "broken", so it must not count toward breaker_threshold: mixing the two
    # opened the failure breaker on a model that was only throttled.
    quota_backoff_base_s: float = 1.0
    quota_backoff_max_s: float = 1800.0
    quota_cooldown_floor_s: float = 10.0
    # Attempt COUNT must not be the binding constraint — the budget should be.
    # Measured 2026-08-10: a deterministic 4xx reject costs 0.26s median, so a
    # cascade can afford many of them, while max_attempts=3 gave up after three
    # cheap rejects with ~99% of the budget still unspent. Set high enough that
    # `left <= 1.0` (real time) is what ends a cascade, never an arbitrary count.
    max_attempts: int = 8               # candidates tried before the re-probe arm
    reprobe_top_n: int = 4              # models re-probed on total miss
    # 90s, not 120s: a timeout burns this ENTIRE value (measured median 120.1s
    # per http-598, 4323s total across 36 timeouts; era note 2026-09-14 — those
    # numbers predate per-op_class budgets, which are now 90 default / 135 for
    # retain-consolidation-reflect via request_timeout_s_by_op_class; the ratio
    # reasoning stands: a timeout costs its full budget), so the per-attempt
    # timeout sets the price of one failure. At 120s only two failures fit in a 240s
    # budget and the third attempt was always 'skipped-budget' — max_attempts=3
    # was unreachable in the worst case. Measured success latencies: p95 51.6s,
    # p99 88.4s, max 117.8s, so 90s aborts ~1% of successes, and those cascade
    # to another model instead of being lost.
    request_timeout_s: float = 90.0
    # 45.0 must match config.py DEFAULTS["router"]: app.py builds the live
    # router as RouterConfig(**CFG["router"]), so config.py WINS at runtime and
    # a differing dataclass default here is dead code that misleads readers.
    # test_config_defaults_match_dataclass asserts the two stay in sync.
    probe_timeout_s: float = 45.0
    # Per-op_class overrides for the two timeouts above; see config.py for the
    # measured rationale. Absent op_class falls back to the scalar default.
    # These defaults must MATCH config.py DEFAULTS["router"] like every other
    # field here (test_config_defaults_match_dataclass asserts it): app.py
    # builds the live router as RouterConfig(**CFG["router"]), so config.py
    # wins at runtime and a differing default here would be misleading dead
    # code.
    request_timeout_s_by_op_class: dict = field(
        default_factory=lambda: {"consolidation": 135.0, "reflect": 135.0,
                                 "retain": 135.0})
    probe_timeout_s_by_op_class: dict = field(
        default_factory=lambda: {"consolidation": 100.0, "reflect": 100.0,
                                 "retain": 100.0})
    # Auth failures must EXPIRE. They used to be terminal, and breaker state is
    # persisted in SQLite, so a single daemon start with a missing key (launchctl
    # setenv does not survive a restart of the machine) marked every model 'auth'
    # permanently: ranked() returned [] even after a good key was restored, and
    # no amount of restarting fixed it. A credential problem is the operator's
    # to fix, but it must not brick the index.
    auth_cooldown_s: float = 300.0
    # Sustained-failure floor. The consecutive-fail breaker cannot catch a model
    # that alternates ok/fail: gpt-oss-20b timed out 6 of 8 retain calls
    # (120s each) but never hit 3 in a row, so it stayed top-ranked and every
    # retain paid 120s before cascading. Below this success rate (with at least
    # min_samples in-window) a model is skipped for that op_class. Per-op_class
    # because score() is per-op_class: nemotron-49b is 43% on retain (4x
    # http-400 = deterministic payload rejection) but 100% on consolidation.
    min_success_rate: float = 0.5
    min_samples_for_floor: int = 4
    # Below this much remaining budget, a request-time dial is not attempted
    # at all. `min(request_timeout, _remaining())` otherwise hands the last
    # attempt a few seconds, and that attempt then times out for a reason that
    # is OURS: measured live, short (<20s) http-598 samples cluster at exactly
    # 10.0s from request and sweep sources, and those rows enter score(),
    # success_rate and the breaker as evidence against the model.
    #
    # 12.0s is measured, not guessed. The artifact to fix is `min(135, left)`
    # landing on exactly 10.0s, which it does whenever a 280s budget has two
    # 135s dials behind it: 139 such http-598 rows exist in the last 7 days,
    # all at 10.0s. A floor at or below 10 does not catch them, so a 5s guard
    # would have shipped as a fix that fixes nothing.
    #
    # Kept far below the probe timeouts (45s / 100s) and above the cascade's
    # own `left <= 1.0` stop. Sub-2s http-598s are genuine transport/resolver
    # failures, not starved dials, and are deliberately still recorded.
    min_useful_dial_s: float = 12.0
    # Stochastic fidelity-fail cooldown. An http reject is deterministic, so
    # it keeps REJECT_RECHECK_S: the provider parsed the body and refused the
    # shape, so a retry fails identically. A fidelity-fail is a different
    # failure mode — an HTTP 200 whose content did not obey the contract —
    # and the live index shows it arrives in BURSTS: on reflect,
    # nemotron-3-super-120b-a12b is 308 ok / 330 (93.3% healthy) with 23
    # fidelity-fails in 7d, median inter-failure gap 241s and 8 of 23 gaps
    # under 120s. Two adjacent fails co-occur often, and the old rule spent
    # SEVEN DAYS of ranked depth on each burst.
    #
    # These are minutes-scale and in-memory, like _quota below: a transient
    # window must not survive a restart and bench a model that is serving.
    fidelity_cooldown_base_s: float = 60.0
    fidelity_cooldown_max_s: float = 900.0
    # Absolute-failure gate for the thin-evidence arm of the success floor,
    # used only below min_samples_for_floor. 2 = "failed twice", which no
    # amount of missing samples explains away.
    min_failures_for_thin_floor: int = 2
    # Latency floor, sibling of min_success_rate. A model can pass the success
    # floor and still be unusable: llama-3.3-70b sat at 61% success with p95
    # 96.9s and stayed ranked #2 on retain, so two attempts exhausted the 240s
    # budget and the op wedged until the watchdog reset it (2026-08-07).
    # Default 75s: an attempt must be able to run twice inside total_budget_s
    # (2 x 75 = 150 < 280) so the cascade always has a real second try left.
    # Raise it only if total_budget_s rises too —
    # test_config_defaults_match_dataclass pins these defaults together.
    #
    # This scalar is the FALLBACK, not the rule. It is a restatement of the
    # request budget ("what fits twice"), so when an op_class overrides
    # request_timeout_s the ceiling must move with it or the two disagree.
    # Measured 2026-09-05: reflect was raised to 135s to stop clipping its real
    # distribution, this ceiling stayed at the 90s-era 75s, and eligibility then
    # rejected reflect's four best models (sr 1.00/0.95/0.90/0.70, p95
    # 76-102s) as too slow for a budget they fit inside. ranked() returned [],
    # /health read "degraded", and auto/reflect 503'd for 280s while every one
    # of those models was serving. The timeout raise had silently un-ranked
    # exactly the slow-but-correct models it was meant to keep. Derive, never
    # restate: latency_ceiling_ms() is the single source both sides read.
    max_p95_ms_for_eligibility: float = 75_000.0
    # Last-resort sweep. The main loop dials ranked[:max_attempts]. On total
    # miss the re-probe arm PROBES down the whole list, but pays one probe
    # round-trip per candidate and stops at reprobe_top_n passes — ranks
    # beyond max_attempts are probe-gated, never DIALED directly, and a model
    # that times out a 45s probe can still serve a 90s real request. After
    # both arms miss, the sweep walks the REST of the candidate list in
    # ranked order — skipping already-dialed ids and terminal 'gone' models,
    # dialing straight away with the real body — until something serves or
    # the budget dies. This is what makes "no healthy candidates" nearly
    # unreachable: retain must fail only when literally every live model
    # failed within one request.
    #
    # Dedupe against every attempt of THIS request (main + re-probe retries)
    # so no upstream sees two dials for one client call. Mirrored in
    # config.py DEFAULTS["router"]; test_config_defaults_match_dataclass
    # asserts sync.
    sweep_on_total_miss: bool = True
    sweep_max_models: int = 12
    # Whole-cascade budget. Must stay under the CLIENT's timeout or it gives up
    # mid-cascade and the failover never completes: hindsight's retain timeout
    # is 300s. Per-attempt timeout shrinks to fit what's left, so the cascade
    # always gets to try every candidate.
    #
    # 280s, not 240s: at request_timeout_s=90 this fits 3 full-price timeouts
    # (3 x 90 = 270 < 280) where 240 fit only 2. Headroom to the 300s client
    # timeout stays 20s. Raising this above ~295 would let the cascade outlive
    # the client, which silently discards the write mid-failover — the exact
    # failure this budget exists to prevent. Keep it under the calling
    # client's own timeout; the README's config section states the rule.
    total_budget_s: float = 280.0
    # p95 at or above this means "overloaded", not "slow model". On free shared
    # NIM endpoints latency tracks how many OTHER people are hitting a model
    # right now, so a model answering in 40s is not a worse model — it is the
    # same model, queued, and it is the one about to start timing out. Models at
    # or above this drop below every healthy model regardless of quality tier,
    # and are re-promoted for free the moment their measured p95 recovers.
    #
    # 20s sits well above a warm NIM response (0.4-10s measured across the live
    # pool) and well under both the 75s eligibility ceiling and hindsight's 300s
    # client timeout, so a model gets demoted while it is merely degrading
    # rather than after it has started failing.
    overload_p95_ms: float = 20_000.0
    # Per-model quality-tier overrides, {model_id: 1..5}, for cases where the
    # parameter-count heuristic in quality.tier is wrong. Empty by default: the
    # heuristic is derived from the live catalog, so a correct default needs no
    # maintenance and a static list of pins would rot exactly like the
    # candidates.json this system replaced.
    tier_overrides: dict = field(default_factory=dict)

    def request_timeout_for(self, op_class: Optional[str]) -> float:
        """The request budget for one op_class -- THE lookup. Lives on the
        config because the config owns the data; Router.request_timeout and
        latency_ceiling_ms both delegate here, so the per-op_class override
        cannot be read two ways."""
        return self.request_timeout_s_by_op_class.get(
            op_class, self.request_timeout_s)

    def probe_timeout_for(self, op_class: Optional[str]) -> float:
        """The probe budget for one op_class. See request_timeout_for."""
        return self.probe_timeout_s_by_op_class.get(
            op_class, self.probe_timeout_s)

    def latency_ceiling_ms(self, op_class: Optional[str] = None) -> float:
        """The eligibility latency ceiling for one op_class.

        Derived from that op_class's OWN request budget rather than restated:
        the ceiling's whole meaning is "an attempt must fit twice inside the
        cascade", so it is a function of the budget the attempt actually gets.
        Keeping it as a standalone scalar meant raising a per-op_class timeout
        left the ceiling behind, and eligibility then rejected the very models
        the raise existed to keep (reflect, 2026-09-05 — see
        max_p95_ms_for_eligibility).

        Floors at the scalar default so this can only ever ADMIT models a
        longer budget really can serve, never tighten an op_class below the
        measured-safe 75s. Capped at half of total_budget_s so "derived" still
        guarantees a real second attempt.

        Keeps the measured 75s-of-90s margin rather than setting the ceiling
        AT the budget: a model whose p95 equals its request timeout times out
        on ~5% of calls by construction, and those land as failure samples
        against a model that works. The original pair was 75/90, so the
        headroom fraction is that ratio, applied to whatever budget the
        op_class actually has.
        """
        budget_s = self.request_timeout_for(op_class)
        return min(
            max(self.max_p95_ms_for_eligibility,
                budget_s * 1000.0 * _LATENCY_CEILING_FRACTION),
            self.total_budget_s * 1000.0 / 2.0,
        )


# DNS/resolution errnos that mean "this machine could not resolve the name",
# never "the provider said no". EAI_NONAME is what a macOS launchd agent with a
# wedged resolver returns (observed 2026-09-12, 5202 occurrences in 1-2ms each
# while `dig` and `curl` from a shell on the same host both worked); the others
# are the same class on other platforms / transient resolver states.
_LOCAL_RESOLVER_ERRNOS = frozenset(
    e for e in (
        getattr(socket, "EAI_NONAME", None),
        getattr(socket, "EAI_AGAIN", None),
        getattr(socket, "EAI_FAIL", None),
        getattr(socket, "EAI_NODATA", None),
    )
    if e is not None
)

# Route-family errnos: the packet never left this host. Same incident class as
# the resolver fault above (sleep, interface flap, VPN topology, the laptop
# slept): they hit EVERY candidate identically, so scoring them against a model
# manufactures the phantom-breaker damage of 2026-09-12 at a different layer.
# ECONNUNREACH is Linux-only (BSD/EUIA family lives under EHOSTUNREACH here);
# getattr keeps this importable on macOS. Deliberately NOT here: ECONNREFUSED
# and connection resets — the peer answered and rejected/reset, which IS
# evidence about the provider; read timeouts likewise stay the model's.
_LOCAL_ROUTE_ERRNOS = frozenset(
    e for e in (
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "ECONNUNREACH", None),
    )
    if e is not None
)


def _is_local_resolver_failure(exc: BaseException) -> bool:
    """True when the exception is this host failing to resolve, not the model."""
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, socket.gaierror) and reason.errno in _LOCAL_RESOLVER_ERRNOS


def _is_local_network_failure(exc: BaseException) -> bool:
    """True when THIS host could not reach the provider at all — resolution or
    routing. Checked by TYPE and errno, never message text (same rule as the
    resolver predicate: strings are locale/platform dependent)."""
    if _is_local_resolver_failure(exc):
        return True
    reason = getattr(exc, "reason", exc)
    return (isinstance(reason, OSError)
            and not isinstance(reason, socket.gaierror)
            and getattr(reason, "errno", None) in _LOCAL_ROUTE_ERRNOS)


@dataclass
class Attempt:
    model_id: str
    status: str
    latency_ms: Optional[float]
    detail: str = ""


@dataclass
class RouteResult:
    ok: bool
    model_id: Optional[str] = None
    response: Optional[dict] = None
    attempts: list[Attempt] = field(default_factory=list)
    reprobed: bool = False
    swept: bool = False


class Router:
    def __init__(
        self,
        index: Index,
        upstream_base: str,
        api_key: str | Callable[[], str],
        cfg: Optional[RouterConfig] = None,
        # injectable for tests: (url, body, headers, timeout) -> (status, dict)
        transport: Optional[Callable] = None,
    ):
        self.index = index
        self.upstream_base = upstream_base.rstrip("/")
        # Accept a callable so the key is read at CALL time, not import time.
        # Cached-at-import meant a credential fixed after the daemon started was
        # ignored until a full restart — and every call 401'd in the meantime.
        self._api_key = api_key if callable(api_key) else (lambda: api_key)
        self.cfg = cfg or RouterConfig()
        self._transport = transport or self._http_post
        # Provider-wide 429 pause (epoch seconds). NIM throttles the shared
        # API key, not one model, so a Retry-After applies to every sibling:
        # dialing the next candidate during the window spends budget on a
        # guaranteed 429 and pollutes its samples with our own throttle.
        # free-coding-models pauses the entire provider the same way
        # (v0.5.81 provider-cooldown.js:127-135, max-of-windows).
        self._provider_pause_until = 0.0
        # Per-model quota ladder: {model_id: {"level": int, "until": epoch}}.
        # In-memory on purpose (like the provider pause): a quota window is
        # minutes-scale, and persisting it would let one stale row bench a
        # model across a restart for up to quota_backoff_max_s.
        self._quota: dict[str, dict] = {}
        self._quota_lock = threading.Lock()
        self._init_cooldown_state()

    def _init_cooldown_state(self) -> None:
        """Fidelity cooldowns: {(model_id, op_class): expiry_epoch}.

        In-memory on the same reasoning as `_quota` — a transient window that
        outlived a restart would bench a model that is demonstrably serving.
        The lock is mandatory, not optional: `eligible()` is called
        concurrently from `/mesh/status` and `/health` (app.py builds one
        Router for the process), so this is a shared mutable dict read from a
        read-only-by-contract method.
        """
        self._fidelity: dict[tuple[str, str], float] = {}
        self._fidelity_level: dict[tuple[str, str], int] = {}
        self._fidelity_strike: dict[tuple[str, str], int] = {}
        self._fidelity_armed_at: dict[tuple[str, str], float] = {}
        self._fidelity_lock = threading.Lock()
        # Test seam: the production clock is time.time, but the cooldown's
        # load-bearing property is that it DECAYS, which only a test can
        # observe without waiting minutes.
        self._clock = time.time

    def _advance_clock(self, seconds: float) -> None:
        """Test-only: move the cooldown clock forward."""
        self._clock = lambda: time.time() + seconds

    def _fidelity_cooldown_left(self, model_id: str, op_class: str) -> float:
        """Seconds remaining on this model's fidelity cooldown (0 = serving).

        A success recorded by ANY source clears the cooldown, not just one this
        Router observed: the index is the record of what the model actually
        did, and a discovery probe or a sweep that got a contract-obeying
        answer is exactly as much evidence of recovery as a main-loop dial.
        Relying on the in-process dial path alone would let a cooldown outlive
        the recovery it was blind to, which is how a model gets lost for a
        reason nobody can see in the logs.

        Reads through the lock and prunes the expired entry, so the dict cannot
        grow without bound across a long-lived process.
        """
        key = (model_id, op_class)
        now = self._clock()
        with self._fidelity_lock:
            until = self._fidelity.get(key)
            if until is None:
                return 0.0
            armed_at = self._fidelity_armed_at.get(key)
        if armed_at is not None:
            try:
                last_ok = self.index.last_success_ts(model_id, op_class)
            except Exception:      # never let telemetry break eligibility
                last_ok = None
            if last_ok is not None and last_ok > armed_at:
                self._fidelity_succeed(model_id, op_class)
                return 0.0
        if until <= now:
            self._fidelity_succeed(model_id, op_class)
            return 0.0
        return until - now

    def _fidelity_fail(self, model_id: str, op_class: str) -> None:
        """Arm (or re-arm) the cooldown after a fidelity violation.

        Only the FIRST violation in a window extends it: a burst must not
        ratchet a model into an escalating penalty. A model that fails twice
        while already cooling down is not new information.
        """
        key = (model_id, op_class)
        now = self._clock()
        with self._fidelity_lock:
            # Already cooling: a burst must not ratchet the penalty, and the
            # strike count is irrelevant while a window is live.
            if self._fidelity.get(key, 0.0) > now:
                return
            # TWO consecutive violations are still the trigger, unchanged from
            # the gate this replaces. One empty-content 200 during a provider
            # burst is not a verdict — the cascade already absorbed it, and
            # these models run 93.3% ok. Only the WINDOW was wrong.
            strikes = self._fidelity_strike.get(key, 0) + 1
            self._fidelity_strike[key] = strikes
            if strikes < 2:
                return
            # Failing again AFTER the window expired escalates (60s, 120s,
            # 240s ... capped at fidelity_cooldown_max_s), so a persistently
            # non-compliant model backs off further while a one-off burst pays
            # the base window only. A success resets the ladder entirely.
            level = self._fidelity_level.get(key, 0)
            self._fidelity_level[key] = level + 1
            window = min(self.cfg.fidelity_cooldown_max_s,
                         self.cfg.fidelity_cooldown_base_s * (2 ** level))
            self._fidelity[key] = now + window
            self._fidelity_armed_at[key] = now

    def _fidelity_succeed(self, model_id: str, op_class: str) -> None:
        """Any contract-obeying answer ends the cooldown immediately.

        Waiting out a window while the model is demonstrably answering is the
        failure mode this replaces, and a success also resets the strike
        counter and the escalation ladder.
        """
        key = (model_id, op_class)
        with self._fidelity_lock:
            self._fidelity.pop(key, None)
            self._fidelity_level.pop(key, None)
            self._fidelity_strike.pop(key, None)
            self._fidelity_armed_at.pop(key, None)
    @property
    def api_key(self) -> str:
        return self._api_key() or ""

    # -- transport ----------------------------------------------------------

    def _http_post(
        self, url: str, body: dict, headers: dict, timeout: float
    ) -> tuple[int, dict]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                try:
                    return resp.status, json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # HTML maintenance page with a 200 wrapper — treat as
                    # transient provider failure, never forward to the client.
                    return 599, {"error": "malformed upstream body"}
        except urllib.error.HTTPError as e:
            payload: dict
            try:
                payload = json.loads(e.read().decode())
            except Exception:
                payload = {"error": str(e)}
            if e.code == 429:
                # Surface Retry-After to dial() WITHOUT changing the transport
                # signature (tests inject (status, dict) transports). NIM's
                # 429 is a statement about the shared API key, not one model:
                # free-coding-models pauses the whole provider on it
                # (v0.5.81 ping.js:186-189) and that matches the 40-rpm
                # single-key budget here.
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    if ra is not None:
                        payload["_retry_after_s"] = max(0.0, float(ra))
                except (TypeError, ValueError):
                    pass  # HTTP-date form or garbage: no pause signal
            return e.code, payload
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # A LOCAL resolver failure is not a statement about the model.
            # Observed 2026-09-12: a launchd-started instance came up with a
            # broken resolver and logged 5,202 `URLError: nodename nor servname
            # provided` in 1-2ms each. Every one was recorded as a failure
            # sample against whichever model was being dialled, which tripped
            # its breaker and wiped the scores for three of four aliases —
            # `mesh-pool-breadth` then correctly reported "0 model(s) carry a
            # score". DNS worked fine from a shell the whole time; a restart of
            # the daemon cured it, and neighbouring pids in the same log show
            # 0 errors, so the fault is per-process, not per-model.
            #
            # 599 is already the "not the model's fault" transient code
            # (malformed upstream body). Reuse it: dial() still fails over to
            # the next candidate, but record_sample() below refuses to write a
            # verdict about a model we never actually reached.
            if _is_local_network_failure(e):
                return 599, {"error": f"local network fault, model not reached: {type(e).__name__}: {e}",
                             "_local_fault": True}
            return 598, {"error": f"{type(e).__name__}: {e}"}

    # -- quota ladder (separate from the breaker) ---------------------------

    def quota_left(self, model_id: str) -> float:
        """Seconds left on this model's quota cooldown; 0.0 if none. Read-only."""
        with self._quota_lock:
            q = self._quota.get(model_id)
            return max(0.0, q["until"] - time.time()) if q else 0.0

    def quota_all(self) -> dict[str, dict]:
        """Snapshot for /mesh/status: level + seconds left per tracked model."""
        now = time.time()
        with self._quota_lock:
            return {m: {"level": q["level"],
                        "cooldown_left_s": round(max(0.0, q["until"] - now), 1)}
                    for m, q in self._quota.items()}

    def _on_quota_hit(self, model_id: str, retry_after: Optional[float]) -> float:
        """Arm/extend the quota cooldown for one model; returns the window.

        Mirrors CLIProxyAPI's 429 branch: Retry-After wins (floored, level
        unchanged); else a still-live window is kept as-is (a burst of 429s
        inside one window must not climb the ladder N rungs); else the
        ladder steps. A later 429 only ever extends a window, never shortens.
        """
        now = time.time()
        with self._quota_lock:
            q = self._quota.setdefault(model_id, {"level": 0, "until": 0.0})
            level = q["level"]
            if retry_after is not None:
                window = max(float(retry_after), self.cfg.quota_cooldown_floor_s)
                until = now + window
            elif q["until"] > now:
                until = q["until"]
            else:
                cd = self.cfg.quota_backoff_base_s * (2 ** level)
                if cd >= self.cfg.quota_backoff_max_s:
                    cd = self.cfg.quota_backoff_max_s
                else:
                    level += 1
                until = now + cd
            q["level"] = level
            q["until"] = max(q["until"], until)
            window = q["until"] - now
        logger.warning(
            "quota-cooldown model=%s window=%.1fs level=%d (retry-after=%s)",
            model_id, window, level, retry_after,
        )
        return window

    def _on_quota_clear(self, model_id: str) -> None:
        with self._quota_lock:
            if self._quota.pop(model_id, None) is not None:
                logger.warning("quota-cooldown model=%s cleared (success)", model_id)

    # -- breaker ------------------------------------------------------------

    def request_timeout(self, op_class: Optional[str]) -> float:
        """The request budget for one op_class.

        Single predicate for BOTH timeouts (see probe_timeout below): every
        call site routes through these two, so an op_class cannot end up with
        a per-class probe budget and a retain-tuned request budget.
        """
        return self.cfg.request_timeout_for(op_class)

    def probe_timeout(self, op_class: Optional[str]) -> float:
        """The probe budget for one op_class. See request_timeout."""
        return self.cfg.probe_timeout_for(op_class)

    def eligible(self, model_id: str, op_class: Optional[str] = None) -> bool:
        b = self.index.breaker_get(model_id)
        if b["state"] == "gone":
            return False
        # Quota cooldown: separate timer, checked before the breaker. Read-only.
        if self.quota_left(model_id) > 0:
            return False
        if b["state"] in ("auth", "down"):
            # Retry-after-cooldown, not terminal: see auth_cooldown_s. An
            # unexpired cooldown is the whole answer — the model is serving
            # its timeout and nothing below can readmit it.
            #
            # READ-ONLY here: this method is called by ranked() (the
            # /mesh/status path) and /health — a status poll must not mutate
            # the breaker table. The actual flip to `recovering` happens in
            # dial() at attempt time (see _transition_for_attempt).
            if time.time() < b["cooldown_until"]:
                return False
            # Cooldown expired => the model gets to be CONSIDERED again, which
            # is not the same as being admitted. This used to `return True`,
            # and that early return sat ABOVE every floor below, so opening the
            # breaker on a model made it MORE eligible than one the breaker had
            # never touched: the success-rate, thin-evidence, reject, fidelity
            # and latency floors were all skipped for exactly the models with
            # the worst evidence. Observed 2026-08-30 — minimax-m3 held rank 0
            # on live auto/retain AND auto/reflect at success_rate 0.7% (n=140)
            # ahead of two models that were still answering, and survived a
            # model rotation (the previous leader had the same shape at 1%,
            # n=138), because `success_rate > 0` also keeps it out of
            # BUCKET_FAILING and quality-first ranking then sorts the big id
            # to the top. Falling through re-imposes the floors; a model with
            # no adverse evidence still scores None, clears them all, and
            # returns on schedule, so recovery is unchanged.
        # Sustained-failure floor: catches the intermittent model the
        # consecutive-fail breaker structurally cannot (ok/fail alternating
        # never reaches `breaker_threshold` in a row).
        if op_class is not None:
            # Capability rejection floor. A 400/413/422 is the provider saying
            # it parsed the request and refuses it, so every retry fails
            # identically. The success-rate floor below cannot catch this: it
            # only engages at min_samples_for_floor samples, and a model that
            # deterministically rejects never earns a 4th sample (one attempt
            # per cascade, and failing does not cause resampling). Observed
            # 2026-08-08 — nemotron-mini-4b-instruct (4096-token context, and
            # we ask for 4096 completion tokens) sat at n=1, success_rate=0.0,
            # eligible=True and burned a cascade slot on every memory op.
            if self.index.unrebutted_reject(model_id, op_class) is not None:
                return False
            # Fidelity cooldown, deliberately NOT the reject gate above.
            # An http reject is deterministic and keeps REJECT_RECHECK_S. A
            # fidelity-fail is a 200 that broke the contract, and it bursts:
            # on reflect the gate armed 3x in 24h on a model that was 93.3%
            # healthy overall, so a seven-day window was spending ranked depth
            # on transient provider behaviour. Short, in-memory, decayed by the
            # clock, cleared by any success — and cleared on restart, because
            # a stale row must never bench a model that is serving.
            if self._fidelity_cooldown_left(model_id, op_class) > 0.0:
                return False
            s = self.index.score(model_id, op_class)
            if (s is not None
                    and s.n >= self.cfg.min_samples_for_floor
                    and s.success_rate < self.cfg.min_success_rate):
                return False
            # Thin-evidence failure floor. The floor above waits for
            # min_samples_for_floor (4) samples, which is the right caution
            # for a model that has merely been unlucky. It is the wrong
            # caution for one that has already failed more often than it has
            # succeeded: at n=3 / success_rate=0.333 the sample guard
            # suppresses the floor, and because ranking is quality-first a
            # tier-5 id then sorts to #1 and takes live memory traffic while
            # measurably failing. Observed 2026-08-16 — openai/gpt-oss-120b,
            # n=3, success_rate 0.333, bucket "healthy", ranked #1 on both
            # auto/retain and auto/reflect.
            #
            # This arm exists for INTERMITTENT failure, which is exactly what
            # the consecutive-fail breaker cannot see (ok/fail alternating
            # never reaches breaker_threshold in a row). It must not preempt
            # the breaker on the consecutive path: gating at 2 failures with
            # breaker_threshold=3 made a model ineligible after its 2nd
            # straight failure, so the 3rd request never ran, the breaker
            # never opened, and no cooldown or recovery was ever scheduled.
            # Requiring a success in the window keeps this to the alternating
            # case and leaves an all-failure run to the breaker.
            #
            # Not an eviction: probes bypass eligible(), so a model that
            # recovers earns samples back and returns on its own.
            if s is not None and s.n < self.cfg.min_samples_for_floor:
                failures = round(s.n * (1.0 - s.success_rate))
                successes = s.n - failures
                if (successes >= 1
                        and failures >= self.cfg.min_failures_for_thin_floor
                        and s.success_rate < self.cfg.min_success_rate):
                    return False
            # Latency floor. Success rate alone is not enough: a model can sit
            # above the success floor and still be unusable because a single
            # attempt eats the whole cascade budget. Observed 2026-08-07 —
            # llama-3.3-70b at 61% success (above the 0.5 floor) with p95 96.9s
            # stayed ranked #2 on retain, so two attempts blew total_budget_s=240
            # and the op wedged until the watchdog reset it.
            # Ranking already knew (score 51.9 vs 66.4); eligibility did not.
            #
            # Ceiling comes from latency_ceiling_ms(op_class), not the raw
            # scalar: an op_class with a longer request budget must admit the
            # slower models that budget can actually serve (reflect, 2026-09-05).
            #
            # Reads p95_ms (successes only), NOT p95_all_ms. This asks one
            # question — "when this model works, is a single success affordable?"
            # Feeding it p95_all (2026-09-13) broke it: a timeout sample records
            # the per-attempt timeout, and the ceiling is a FRACTION of that same
            # timeout, so any model that ever timed out failed by construction.
            # With p95 over the newest SCORE_RECENT_N=20, ~2 timeouts (>=5%)
            # meant permanent exclusion against a measured 8-19% baseline timeout
            # rate per model — uniform across vendors, i.e. provider load, not
            # model fault. Lanes collapsed to whichever model held 0 timeouts
            # in-window: reflect 10 -> 1, consolidation 9 -> 1 (2026-09-14).
            #
            # It also charged one event twice: a slow response already costs the
            # model a breaker cooldown (_on_transient_fail, 30s -> 300s), the
            # correct minutes-scale handling of "overloaded right now". The two
            # disagreed openly — gemma read healthy/consec=0 while this floor
            # rejected it over a timeout 78 minutes old.
            if (s is not None
                    and s.n >= self.cfg.min_samples_for_floor
                    and s.p95_ms > self.cfg.latency_ceiling_ms(op_class)):
                return False
            # Budget floor, sibling of the one above and a DIFFERENT question:
            # "counting the overload it actually suffers, can the cascade afford
            # to dial this model AND still retry?" A model can be fast when it
            # works yet fail so often that its expected cost plus one more
            # full-timeout attempt overruns the budget.
            #
            # The 2026-09-13 dead-slow alternator is NOT this test's job: it
            # measured 0.32 success / 67% timeouts and is excluded by
            # min_success_rate above. A literal 50/50 model sits exactly on that
            # boundary and is admitted deliberately — at even odds with a retry
            # available the cascade does better trying it than refusing it, and
            # the breaker still reacts if the failures cluster.
            if s is not None and s.n >= self.cfg.min_samples_for_floor:
                # Through request_timeout(), the single predicate every call
                # site shares -- an inline restatement here could drift from it.
                timeout_ms = 1000.0 * self.request_timeout(op_class)
                expected_ms = (s.success_rate * s.p95_ms
                               + (1.0 - s.success_rate) * timeout_ms)
                if expected_ms + timeout_ms > self.cfg.total_budget_s * 1000.0:
                    return False
        return True  # healthy | recovering

    def _on_success(self, model_id: str) -> None:
        prev = self.index.breaker_get(model_id)["state"]
        self.index.breaker_set(
            model_id, state="healthy", consec_fails=0,
            cooldown_until=0.0, cooldown_s=0.0,
        )
        if prev != "healthy":
            # Transitions log (2026-09-14): SQLite-only writes made "did the
            # breaker open / close?" unanswerable from logs (0 hits in 24,665
            # lines) — the 2026-08-03 postmortem rule is breakers must open
            # AND close, and an operator must be able to WATCH that happen.
            logger.warning("breaker-transition model=%s %s->healthy", model_id, prev)

    def _on_transient_fail(self, model_id: str) -> None:
        b = self.index.breaker_get(model_id)
        consec = b["consec_fails"] + 1
        if b["state"] == "recovering":
            # failed its one recovery request -> reopen with doubled cooldown
            cd = min(max(b["cooldown_s"], self.cfg.breaker_cooldown_s) * 2,
                     self.cfg.breaker_cooldown_max_s)
            self.index.breaker_set(
                model_id, state="down", consec_fails=consec,
                cooldown_s=cd, cooldown_until=time.time() + cd,
            )
            logger.warning(
                "breaker-transition model=%s recovering->down consec=%d "
                "cooldown_s=%.0f (failed recovery probe; doubled cooldown)",
                model_id, consec, cd)
        elif consec >= self.cfg.breaker_threshold:
            cd = self.cfg.breaker_cooldown_s
            self.index.breaker_set(
                model_id, state="down", consec_fails=consec,
                cooldown_s=cd, cooldown_until=time.time() + cd,
            )
            logger.warning(
                "breaker-transition model=%s %s->down consec=%d cooldown_s=%.0f "
                "(threshold=%d)", model_id, b["state"], consec, cd,
                self.cfg.breaker_threshold)
        else:
            self.index.breaker_set(model_id, consec_fails=consec)

    # -- single upstream call ----------------------------------------------

    def _transition_for_attempt(self, model_id: str) -> None:
        """Recover `down`/`auth` models whose cooldown has expired, at attempt
        time only.

        This is the ONE place a recovery-window transition happens. It used to
        live inside eligible(), which ranked() (/mesh/status) and /health also
        call — so every status poll or health check mutated the breaker table
        (audit 2026-08-25). Ranking must still ADMIT an expired-cooldown model
        (eligible() returns True for it), but the state flip now waits for the
        real dial so introspection endpoints stay read-only.
        """
        b = self.index.breaker_get(model_id)
        if b["state"] in ("down", "auth") and time.time() >= b["cooldown_until"]:
            self.index.breaker_set(model_id, state="recovering")

    def dial(
        self, model_id: str, body: dict, op_class: str, source: str,
        timeout: Optional[float] = None, request_id: Optional[str] = None,
    ) -> tuple[Optional[dict], Attempt]:
        # Flip a `down`/`auth` model whose cooldown has expired to
        # `recovering` HERE — at attempt time, not at rank/status time.
        # eligible() is read-only (called by ranked() and /health), so the
        # recovery-window transition must live on the one path every real
        # request shares. audit 2026-08-25: /mesh/status and /health used to
        # call eligible() and thereby flipped breaker state on every poll — a
        # read API that wrote. See _transition_for_attempt.
        self._transition_for_attempt(model_id)
        # Provider-wide throttle window: NIM said "stop asking" with a
        # Retry-After. If the window outlives this attempt's budget, don't
        # spend an upstream call on a guaranteed 429 — and don't record a
        # sample, because a self-inflicted throttle hit is evidence about our
        # request pacing, not about the model. If the window ends inside the
        # budget, wait it out and dial with what remains.
        #
        # Explicit None check, not `timeout or default`: a caller that computed
        # a ZERO grant must not be handed the full request timeout (`0.0 or 135`
        # is 135). And a non-positive grant is refused outright rather than
        # passed to the transport -- urllib reads 0 as non-blocking and a
        # negative timeout raises -- and it records no sample, for the same
        # reason a starved dial does not: running out of budget is ours.
        budget = self.request_timeout(op_class) if timeout is None else timeout
        if budget <= 0:
            return None, Attempt(
                model_id, "skipped-budget", None,
                f"non-positive dial grant {budget:.1f}s",
            )
        pause_left = self._provider_pause_until - time.time()
        if pause_left > 0:
            if pause_left >= budget - 1.0:
                return None, Attempt(
                    model_id, "skipped-provider-pause", None,
                    f"provider 429 Retry-After window: {pause_left:.1f}s left",
                )
            time.sleep(pause_left)
            budget -= pause_left
        # Per-model quota cooldown: this model said 429 and its window is
        # live. Same contract as the provider pause skip: no upstream call,
        # no sample. Never waited out — a sibling may be free right now.
        quota_left = self.quota_left(model_id)
        if quota_left > 0:
            return None, Attempt(
                model_id, "skipped-quota-cooldown", None,
                f"model 429 quota cooldown: {quota_left:.1f}s left",
            )
        url = self.upstream_base + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        upstream_body = dict(body)
        upstream_body["model"] = model_id
        payload_chars = sum(
            len(str(m.get("content", ""))) for m in body.get("messages", [])
        )
        t0 = time.monotonic()
        status_code, payload = self._transport(
            url, upstream_body, headers, budget
        )
        ms = (time.monotonic() - t0) * 1000.0

        if status_code == 429:
            # Arm the provider-wide pause. Retry-After is authoritative when
            # present; without it, a short fixed window still stops the
            # cascade from machine-gunning the shared key through its
            # remaining candidates. max(): never shorten an armed window.
            ra = payload.pop("_retry_after_s", None) if isinstance(payload, dict) else None
            self._on_quota_hit(model_id, ra)
            window = ra if ra is not None else self.cfg.provider_pause_default_s
            window = min(float(window), self.cfg.provider_pause_max_s)
            self._provider_pause_until = max(
                self._provider_pause_until, time.time() + window
            )
            logger.warning(
                "provider-pause armed: 429 on %s, window=%.1fs (retry-after=%s)",
                model_id, window, ra,
            )

        if status_code == 200:
            # Fidelity gate, enforced at the ONE place every response passes.
            # It used to run only inside probe_verdict, and the 200 below was
            # recorded as `ok` BEFORE any check — so a prose-leaking model
            # accrued success_rate=1.0, topped the ranking, and served real
            # memory traffic output hindsight cannot parse (audit 2026-08-24).
            # The response still RETURNS here: this cascade already paid for
            # it, and refusing it would spend another full upstream call. What
            # changes is the evidence — the sample reads fidelity-fail, so the
            # success-rate floor demotes the model, and two consecutive
            # violations arm a short per-(model, op_class) cooldown that drops
            # it from ranked order until it expires or any success lands
            # (_fidelity_fail / _fidelity_cooldown_left).
            ok, why = check_fidelity(payload, op_class)
            if not ok:
                self.index.record(model_id, op_class, source,
                                  FIDELITY_FAIL_STATUS, ms, payload_chars,
                                  request_id=request_id)
                # Log the REQUEST that failed alongside the verdict, truncated.
                # Added 2026-08-24: 1,544 fidelity-fails all read
                # "missing/empty `facts` list" and none could be reproduced —
                # hand-built payloads passed at every size from 2KB to 54KB on
                # the same models. The reason was unreachable because the log
                # recorded the verdict and the payload SIZE but never the
                # payload, so the one thing that distinguished a failing call
                # from a passing one was the one thing not captured. A failure
                # message that cannot reproduce its own failure is not
                # diagnostic.
                try:
                    _msgs = body.get("messages") or []
                    _last = _msgs[-1].get("content", "") if _msgs else ""
                    _sys = _msgs[0].get("content", "") if _msgs else ""
                    _got = parse_content(payload)
                except Exception:      # never let logging break routing
                    _last = _sys = _got = "<unavailable>"
                logger.warning(
                    "fidelity-fail model=%s op_class=%s source=%s "
                    "payload_chars=%d why=%s system=%.200r user_tail=%.300r "
                    "response=%.300r",
                    model_id, op_class, source, payload_chars, why,
                    _sys, _last[-300:], _got,
                )
                # Arm the short cooldown, NOT the reject gate. A second
                # consecutive failure inside a live window deliberately does
                # not extend it — see _fidelity_fail.
                self._fidelity_fail(model_id, op_class)
                return payload, Attempt(model_id, FIDELITY_FAIL_STATUS, ms, why)
            self.index.record(model_id, op_class, source, OK, ms, payload_chars,
                              request_id=request_id)
            # A contract-obeying answer clears any live fidelity cooldown, so a
            # recovered model rejoins the cascade immediately rather than
            # waiting out a window it has already disproved.
            self._fidelity_succeed(model_id, op_class)
            self._on_success(model_id)
            self._on_quota_clear(model_id)
            return payload, Attempt(model_id, OK, ms)

        status = f"http-{status_code}"
        if status_code in GONE_CODES:
            self.index.mark_gone(model_id, status)
            self.index.record(model_id, op_class, source, status, ms, payload_chars,
                              request_id=request_id)
            return None, Attempt(model_id, status, ms, "gone: EOL'd at request time")
        if status_code in AUTH_CODES:
            self.index.breaker_set(
                model_id, state="auth", consec_fails=0,
                cooldown_s=self.cfg.auth_cooldown_s,
                cooldown_until=time.time() + self.cfg.auth_cooldown_s,
            )
            self.index.record(model_id, op_class, source, status, ms, payload_chars,
                              request_id=request_id)
            return None, Attempt(model_id, status, ms, "auth: check API key")
        if status_code in REJECT_CODES:
            # Recorded (so the success-rate floor sees it and stops picking this
            # model for this op_class) but NOT breaker-counted: the model is
            # healthy, it just refuses this shape of request.
            self.index.record(model_id, op_class, source, status, ms, payload_chars,
                              request_id=request_id)
            detail = str(payload.get("error", payload.get("detail", "")))[:200]
            # Log it: a 4xx is a CAPABILITY signal, not noise. Diagnosing the
            # 2026-08-07 nemotron rejections meant writing a repro script purely
            # because this body was recorded to mesh.db but never surfaced —
            # sqlite stores the status code, not the upstream's reason.
            logger.warning(
                "reject model=%s op_class=%s status=%s payload_chars=%d detail=%s",
                model_id, op_class, status, payload_chars, detail or "(empty)",
            )
            return None, Attempt(
                model_id, status, ms, f"rejected (not retryable): {detail}"
            )
        # Local fault: the request never reached the provider, so there is no
        # verdict to record about this model. Recording one is how a wedged
        # resolver on THIS host wiped the alias scores on 2026-09-12 — every
        # candidate accumulated failure samples and tripped its breaker while
        # the models themselves were fine. Fail over to the next candidate
        # (the caller still gets an answer if any path works) but write
        # nothing: no sample, no breaker count.
        if payload.get("_local_fault"):
            detail = str(payload.get("error", ""))[:200]
            logger.error(
                "local-fault model=%s op_class=%s ms=%.0f detail=%s "
                "(NOT recorded against the model; this host could not reach "
                "the provider — check this host's DNS/routes, then restart the "
                "daemon)",
                model_id, op_class, ms, detail or "(empty)",
            )
            return None, Attempt(model_id, status, ms, detail)
        # transient (incl. 598 network / 599 malformed body). A 429 is still
        # recorded (score evidence) but the quota ladder above owns its
        # cooldown: it never counts toward the failure breaker.
        self.index.record(model_id, op_class, source, status, ms, payload_chars,
                              request_id=request_id)
        if status_code != 429:
            self._on_transient_fail(model_id)
        detail = str(payload.get("error", payload.get("detail", "")))[:200]
        logger.warning(
            "transient model=%s op_class=%s status=%s ms=%.0f detail=%s",
            model_id, op_class, status, ms, detail or "(empty)",
        )
        return None, Attempt(model_id, status, ms, detail)

    # -- ranking ------------------------------------------------------------

    def ranked(self, candidates: list[str], op_class: str) -> list[str]:
        """Eligible candidates: best model that is actually up, first.

        Ordering is (availability bucket, quality tier, latency) — see
        quality.rank_key. Availability dominates because the mesh's job is
        uptime; quality breaks ties within a bucket because a better model is
        worth having when several are equally up; latency breaks ties within a
        tier because among equals the quick one is preferable.

        Replaces a single blended float. That score weighted p95, jitter, spike
        rate and success rate — all availability, no quality — so a 1b model
        outranked a 120b whenever it answered faster, which is backwards for a
        memory backbone. It had also lost all resolution: measured 2026-08-09,
        all 23 scored retain models fell in [49.9, 50.0] and ranking degenerated
        to the alphabetical tiebreak, leaving gemma-4-31b-it at rank 0 on p95
        44.3s with jitter 4.77 while 0.4s models sat behind it.

        Unknowns rank above FAILING but below anything healthy: a model with no
        evidence is a maybe, and a maybe beats a model measured to be broken.
        """
        eligible = []
        for m in candidates:
            if not self.eligible(m, op_class):
                continue
            eligible.append((m, self.index.score(m, op_class)))
        eligible.sort(
            key=lambda t: rank_key(
                t[0], t[1], self.cfg.overload_p95_ms, self.cfg.tier_overrides
            )
        )
        return [m for m, _ in eligible]

    # -- probe (used by the re-probe arm and discovery) ----------------------

    def probe(
        self, model_id: str, op_class: str, messages: list[dict],
        timeout: Optional[float] = None,
    ) -> bool:
        """Boolean convenience over probe_verdict for the re-probe arm, which
        only cares "did it answer AND obey". Never use where busy-vs-unusable
        matters — that distinction is the whole point of the verdict form."""
        return self.probe_verdict(model_id, op_class, messages, timeout)[0] == "pass"

    def probe_verdict(
        self, model_id: str, op_class: str, messages: list[dict],
        timeout: Optional[float] = None,
    ) -> tuple[str, str]:
        """Probe once, returning (verdict, detail) instead of a bare bool.

        Three outcomes that a boolean collapses into one, wrongly:

          pass       served and obeyed the op_class contract.
          unusable   PERMANENT: 404/410 (listed in the catalog but not actually
                     servable) or a real fidelity failure (empty/non-JSON).
          busy       TEMPORARY: 429/5xx/timeout. The model is overloaded right
                     now, which says nothing about whether it can do the job.

        Observed 2026-08-08: a backfill pass recorded 51 http-404 and 15
        timeouts and reported all 66 as "failed fidelity". Zero were fidelity
        failures. Treating a busy model as incapable permanently excludes
        exactly the popular models we most want, so `busy` must be retried on a
        later pass rather than held against the model.
        """
        body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0,
            "stream": False,
        }
        payload, att = self.dial(
            model_id, body, op_class, source="probe",
            timeout=self.probe_timeout(op_class) if timeout is None else timeout,
        )
        if payload is None or att.status != OK:
            status = str(att.status or "")
            if status in ("http-404", "http-410"):
                return "unusable", f"not servable ({status})"
            # The fidelity gate inside dial already classified a
            # broken-JSON/empty-content 200 as FIDELITY_FAIL_STATUS and
            # recorded the sample — a capability signal, same family as an
            # http reject, never "busy" (the pre-gate behaviour here called
            # these unusable via its own check; the gate moved upstream, so
            # the verdict must follow the sample, not re-derive it).
            if status == FIDELITY_FAIL_STATUS:
                return "rejected", f"fidelity: {str(att.detail or '')[:120]}"
            # A 4xx reject is a CAPABILITY verdict, not overload: the provider
            # parsed the request and refused it, so re-probing produces the
            # identical answer. Calling it `busy` (the pre-2026-08-08 behaviour)
            # meant a model that can never serve this op_class was re-probed on
            # every pass and stayed eligible for real traffic. It is recorded
            # per-op_class by dial, so `unrebutted_reject` gates it — but it is
            # NOT mark_gone: the model may serve other op_classes perfectly
            # (nemotron-mini-4b only fails because 4096 completion tokens
            # exceeds its whole context, which is a per-request-shape fact).
            if status in REJECT_STATUS_NAMES:
                return "rejected", f"{status}: {str(att.detail or '')[:120]}"
            return "busy", f"{status}: {str(att.detail or '')[:120]}"

        # A 200 that reaches this line already passed the fidelity gate inside
        # dial — re-running check_fidelity here would re-derive a decision
        # the sample log already holds (and could disagree with it).
        return "pass", ""

    # -- the cascade ---------------------------------------------------------

    def route(
        self,
        candidates: list[str],
        body: dict,
        op_class: str,
        probe_messages: Optional[list[dict]] = None,
    ) -> RouteResult:
        result = RouteResult(ok=False)
        # One id per cascade. Every attempt this route makes — including the
        # re-probe arm's retries — carries it, so a reader can ask "did the
        # CLIENT get an answer" instead of inferring it from timestamps.
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.cfg.total_budget_s

        def _remaining() -> float:
            return deadline - time.monotonic()

        def _dial(mid: str, source: str,
                   grant_s: Optional[float] = None) -> tuple[bool, Attempt, Optional[dict]]:
            """One upstream call that counts only if the body is usable.
            Fidelity failures return a payload but must not end the cascade:
            the client would receive output its own parser rejects. This one
            predicate is the whole cascade — main loop and re-probe retries
            share it, so neither arm can drift into accepting prose.

            `grant_s` MUST be passed by request-time callers: the caller has
            already compared the remaining budget against
            min_useful_dial_s, and letting _dial re-read the clock here leaves
            a window in which time passes between that check and this dial.
            Re-reading is how a 10s dial slipped past a 12s guard."""
            payload, att = self.dial(
                mid, body, op_class, source=source,
                timeout=min(self.request_timeout(op_class),
                            grant_s if grant_s is not None else _remaining()),
                request_id=request_id,
            )
            result.attempts.append(att)   # telemetry: EVERY dial is recorded
            if payload is not None and att.status == OK:
                return True, att, payload
            return False, att, None

        order = self.ranked(candidates, op_class)
        for model_id in order[: self.cfg.max_attempts]:
            left = _remaining()
            if left <= 1.0:
                result.attempts.append(
                    Attempt(model_id, "skipped-budget", None,
                            "cascade budget exhausted")
                )
                break
            # Our budget, not the model's fault: do not spend the last few
            # seconds on a dial that cannot finish, and above all do not
            # record the resulting timeout as evidence against this model.
            if left < self.cfg.min_useful_dial_s:
                result.attempts.append(
                    Attempt(model_id, "skipped-budget", None,
                            f"remaining budget {left:.1f}s below "
                            f"min_useful_dial_s {self.cfg.min_useful_dial_s}s")
                )
                break
            ok, att, payload = _dial(model_id, "request", grant_s=left)
            if ok:
                result.ok, result.model_id, result.response = (
                    True, model_id, payload,
                )
                return result

        # Total miss -> the re-probe arm. The index may be stale (models EOL'd,
        # provider-side incident cleared); measure NOW and try once more.
        # `reprobed` means a probe RAN — set only after the loop proves it,
        # never speculatively: a budget exhausted before the first probe must
        # not report work it did not do (audit 2026-08-24).
        #
        # TIME-BOXED, not gated: this arm must stay reachable when ranking is
        # EMPTY because auth-cooled models are legitimately re-probeable (a
        # restored credential is invisible until something calls). But when
        # the whole pool is merely busy, each hung probe burns its full
        # timeout — live-measured 2026-08-24 as six ~45s probes eating the
        # entire budget before the sweep could dial once. So the arm gets at
        # most a quarter of the remaining budget; the sweep below always has
        # real money left for direct dials.
        if probe_messages:
            reprobe_box = time.monotonic() + max(15.0, _remaining() * 0.25)

            def _box_left() -> float:
                return reprobe_box - time.monotonic()

            fresh: list[str] = []
            for model_id in candidates:
                if (len(fresh) >= self.cfg.reprobe_top_n
                        or _remaining() <= 1.0 or _box_left() <= 1.0):
                    break
                b = self.index.breaker_get(model_id)
                # Only 'gone' is truly terminal. 'down' AND 'auth' models are
                # re-probed here: this arm exists because persisted state may be
                # stale, and a restored credential is exactly that case.
                if b["state"] == "gone":
                    continue
                # A probe is a REQUEST-TIME call in disguise — it dials the
                # upstream and its timeout comes out of the same budget. It
                # was issuing grants as small as 4-8s once earlier arms had
                # drained the budget, and those dials were recorded exactly
                # like the main loop's starved dials. Same floor applies.
                probe_left = min(_remaining(), _box_left())
                if probe_left < self.cfg.min_useful_dial_s:
                    result.attempts.append(
                        Attempt(model_id, "skipped-budget", None,
                                f"re-probe budget {probe_left:.1f}s below "
                                f"min_useful_dial_s "
                                f"{self.cfg.min_useful_dial_s}s")
                    )
                    continue
                if self.probe(model_id, op_class, probe_messages,
                              # `probe_left` is the value the guard above
                              # compared. Re-reading _remaining()/_box_left()
                              # here is the check-to-dial gap grant_s closes
                              # for request dials: time can pass in between,
                              # and the re-read can go to zero or negative.
                              timeout=min(self.probe_timeout(op_class),
                                          probe_left)):
                    fresh.append(model_id)
            reprobed_any = bool(fresh)
            for model_id in fresh:
                left = _remaining()
                if left <= 1.0:
                    result.attempts.append(
                        Attempt(model_id, "skipped-budget", None,
                                "cascade budget exhausted")
                    )
                    continue
                # Same rule as the main loop: a retry that cannot finish is
                # skipped rather than charged to the model as a timeout.
                if left < self.cfg.min_useful_dial_s:
                    result.attempts.append(
                        Attempt(model_id, "skipped-budget", None,
                                f"remaining budget {left:.1f}s below "
                                f"min_useful_dial_s "
                                f"{self.cfg.min_useful_dial_s}s")
                    )
                    continue
                ok, att, payload = _dial(model_id, "request", grant_s=left)
                if ok:
                    result.reprobed = reprobed_any
                    result.ok, result.model_id, result.response = (
                        True, model_id, payload,
                    )
                    return result

        # Last-resort sweep. Both arms above only ever touch ELIGIBLE models:
        # when the floors exclude everything (whole-pool episode), ranked()
        # returns [] and a naive sweep would never run either — observed live
        # 2026-08-24 as an 11ms 503 on auto/consolidation with zero dials.
        # So the sweep's universe is the FULL candidate list: the ranked
        # tail first, then every candidate the ranking excluded (ineligible
        # floors, stale evidence) — a total miss means the floors' caution
        # has bought nothing and trying beats refusing. Dedupe against EVERY
        # dial this request already made (main loop + re-probe retries); one
        # failure is evidence enough and an upstream must never see two
        # dials for one client call. 'gone' stays terminal — EOL'd ghosts
        # are not candidates.
        if self.cfg.sweep_on_total_miss and not result.ok:
            dialed = {a.model_id for a in result.attempts}
            swept_any = False
            swept_count = 0
            seen = set(order)
            sweep_order = order[self.cfg.max_attempts:] + [
                m for m in candidates if m not in seen
            ]
            for model_id in sweep_order:
                if swept_count >= self.cfg.sweep_max_models:
                    break
                if model_id in dialed or self.index.breaker_get(
                        model_id)["state"] == "gone":
                    continue
                left = _remaining()
                if left <= 1.0 or left < self.cfg.min_useful_dial_s:
                    # Same rule as the main loop: the sweep wins real requests
                    # in 0.9-2.0s, so this floor is deliberately low, but a
                    # dial that cannot finish must not be charged to the
                    # model as a failure.
                    #
                    # `continue`, not `break`: budget does not come back, so
                    # no LATER model can be afforded either, but recording a
                    # skipped-budget attempt for each of them is what tells an
                    # operator how deep the sweep actually got. `break` here
                    # stopped the arm after the first unaffordable model and
                    # silently truncated its own reach.
                    result.attempts.append(
                        Attempt(model_id, "skipped-budget", None,
                                f"remaining budget {left:.1f}s below "
                                f"min_useful_dial_s "
                                f"{self.cfg.min_useful_dial_s}s")
                    )
                    continue
                payload, att = self.dial(
                    model_id, body, op_class, source="sweep",
                    # `left` is the value the guard above compared; re-reading
                    # the clock here is what let starved dials through.
                    timeout=min(self.request_timeout(op_class), left),
                    request_id=request_id,
                )
                result.attempts.append(att)   # EVERY dial is recorded
                swept_any = True
                swept_count += 1
                if payload is not None and att.status == OK:
                    # Fidelity gate already ran inside dial: a 200 here is a
                    # contract-obeying answer, same guarantee as every arm.
                    result.swept = True
                    result.ok, result.model_id, result.response = (
                        True, model_id, payload,
                    )
                    return result
                dialed.add(model_id)
            if swept_any:
                result.swept = True

        return result
