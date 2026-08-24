"""model-mesh: request-time router.

The part nim-proxy never had. Per request: resolve alias -> ranked candidates
-> cascade with circuit breaking -> live re-probe on total miss -> 503 with
evidence only after everything (including a fresh probe pass) failed.

Failure taxonomy (each class routes differently — collapsing them is how the
predecessor silently lost redundancy):

  transient  429 / 5xx / timeout / malformed-JSON  -> breaker counts, cascade
  auth       401 / 403                             -> mark 'auth', skip provider, NO breaker poison
  gone       404 / 410                             -> index.mark_gone NOW, cascade
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .index import FIDELITY_FAIL_STATUS, Index, OK
from .opclass import check_fidelity
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


@dataclass
class RouterConfig:
    breaker_threshold: int = 3          # consecutive fails -> down
    breaker_cooldown_s: float = 120.0   # first cooldown
    breaker_cooldown_max_s: float = 1800.0
    # Attempt COUNT must not be the binding constraint — the budget should be.
    # Measured 2026-08-10: a deterministic 4xx reject costs 0.26s median, so a
    # cascade can afford many of them, while max_attempts=3 gave up after three
    # cheap rejects with ~99% of the budget still unspent. Set high enough that
    # `left <= 1.0` (real time) is what ends a cascade, never an arbitrary count.
    max_attempts: int = 8               # candidates tried before the re-probe arm
    reprobe_top_n: int = 4              # models re-probed on total miss
    # 90s, not 120s: a timeout burns this ENTIRE value (measured median 120.1s
    # per http-598, 4323s total across 36 timeouts), so the per-attempt timeout
    # sets the price of one failure. At 120s only two failures fit in a 240s
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
    # Consecutive fidelity violations (an upstream 200 whose body violates the
    # op_class JSON contract) that drop a model from an op_class until the
    # recheck window elapses or one success intervenes. One strike is not a
    # verdict — the cascade absorbs it and the client was served — but two
    # unrebutted is the settled signal an http reject is. Mirrored in
    # config.py DEFAULTS["router"]; test_config_defaults_match_dataclass
    # asserts the two stay in sync.
    fidelity_fails_for_floor: int = 2
    # Absolute-failure gate for the thin-evidence arm of the success floor,
    # used only below min_samples_for_floor. 2 = "failed twice", which no
    # amount of missing samples explains away.
    min_failures_for_thin_floor: int = 2
    # Latency floor, sibling of min_success_rate. A model can pass the success
    # floor and still be unusable: llama-3.3-70b sat at 61% success with p95
    # 96.9s and stayed ranked #2 on retain, so two attempts exhausted the 240s
    # budget and the op wedged until the watchdog reset it (2026-08-07).
    # Default 75s: an attempt must be able to run twice inside total_budget_s
    # (2 x 75 = 150 < 240) so the cascade always has a real second try left.
    # Raise it only if total_budget_s rises too — audit-timeout-chain.py checks
    # the relationship.
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
    # failure this budget exists to prevent. audit-timeout-chain.py asserts
    # total_budget_s < RETAIN_LLM_TIMEOUT and REFLECT_LLM_TIMEOUT.
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
            try:
                payload = json.loads(e.read().decode())
            except Exception:
                payload = {"error": str(e)}
            return e.code, payload
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return 598, {"error": f"{type(e).__name__}: {e}"}

    # -- breaker ------------------------------------------------------------

    def _eligible(self, model_id: str, op_class: Optional[str] = None) -> bool:
        b = self.index.breaker_get(model_id)
        if b["state"] == "gone":
            return False
        if b["state"] == "auth":
            # Retry-after-cooldown, not terminal: see auth_cooldown_s.
            if time.time() >= b["cooldown_until"]:
                self.index.breaker_set(model_id, state="recovering")
                return True
            return False
        if b["state"] == "down":
            if time.time() >= b["cooldown_until"]:
                # cooldown expired -> recovering: admit one real request
                self.index.breaker_set(model_id, state="recovering")
                return True
            return False
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
            # Fidelity floor, sibling of the reject floor above. An http reject
            # is deterministic; a broken-JSON/empty-content 200 is usually
            # stochastic, so ONE violation is not a verdict (the cascade
            # already absorbed it). `fidelity_fails_for_floor` consecutive
            # violations with no intervening success IS settled evidence, and
            # the same REJECT_RECHECK_S window buys the weekly retry.
            if self.index.unrebutted_fidelity_fails(
                model_id, op_class, need=self.cfg.fidelity_fails_for_floor,
            ) is not None:
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
            # Not an eviction: probes bypass _eligible(), so a model that
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
            if (s is not None
                    and s.n >= self.cfg.min_samples_for_floor
                    and s.p95_ms > self.cfg.max_p95_ms_for_eligibility):
                return False
        return True  # healthy | recovering

    def _on_success(self, model_id: str) -> None:
        self.index.breaker_set(
            model_id, state="healthy", consec_fails=0,
            cooldown_until=0.0, cooldown_s=0.0,
        )

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
        elif consec >= self.cfg.breaker_threshold:
            cd = self.cfg.breaker_cooldown_s
            self.index.breaker_set(
                model_id, state="down", consec_fails=consec,
                cooldown_s=cd, cooldown_until=time.time() + cd,
            )
        else:
            self.index.breaker_set(model_id, consec_fails=consec)

    # -- single upstream call ----------------------------------------------

    def _call(
        self, model_id: str, body: dict, op_class: str, source: str,
        timeout: Optional[float] = None, request_id: Optional[str] = None,
    ) -> tuple[Optional[dict], Attempt]:
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
            url, upstream_body, headers, timeout or self.cfg.request_timeout_s
        )
        ms = (time.monotonic() - t0) * 1000.0

        if status_code == 200:
            # Fidelity gate, enforced at the ONE place every response passes.
            # It used to run only inside probe_verdict, and the 200 below was
            # recorded as `ok` BEFORE any check — so a prose-leaking model
            # accrued success_rate=1.0, topped the ranking, and served real
            # memory traffic output hindsight cannot parse (audit 2026-08-24).
            # The response still RETURNS here: this cascade already paid for
            # it, and refusing it would spend another full upstream call. What
            # changes is the evidence — the sample reads fidelity-fail, so the
            # success-rate floor demotes the model and two violations without
            # an intervening success drop it from the cascade entirely
            # (_eligible / unrebutted_fidelity_fails).
            ok, why = check_fidelity(payload, op_class)
            if not ok:
                self.index.record(model_id, op_class, source,
                                  FIDELITY_FAIL_STATUS, ms, payload_chars,
                                  request_id=request_id)
                logger.warning(
                    "fidelity-fail model=%s op_class=%s source=%s "
                    "payload_chars=%d why=%s",
                    model_id, op_class, source, payload_chars, why,
                )
                return payload, Attempt(model_id, FIDELITY_FAIL_STATUS, ms, why)
            self.index.record(model_id, op_class, source, OK, ms, payload_chars,
                              request_id=request_id)
            self._on_success(model_id)
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
        # transient (incl. 598 network / 599 malformed body)
        self.index.record(model_id, op_class, source, status, ms, payload_chars,
                              request_id=request_id)
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
            if not self._eligible(m, op_class):
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
        """Back-compat boolean probe. Prefer probe_verdict() — a bare bool
        cannot distinguish "this model cannot do the job" from "this model was
        busy just now"."""
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
        payload, att = self._call(
            model_id, body, op_class, source="probe",
            timeout=timeout or self.cfg.probe_timeout_s,
        )
        if payload is None or att.status != OK:
            status = str(att.status or "")
            if status in ("http-404", "http-410"):
                return "unusable", f"not servable ({status})"
            # The fidelity gate inside _call already classified a
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
            # per-op_class by _call, so `unrebutted_reject` gates it — but it is
            # NOT mark_gone: the model may serve other op_classes perfectly
            # (nemotron-mini-4b only fails because 4096 completion tokens
            # exceeds its whole context, which is a per-request-shape fact).
            if status in REJECT_STATUS_NAMES:
                return "rejected", f"{status}: {str(att.detail or '')[:120]}"
            return "busy", f"{status}: {str(att.detail or '')[:120]}"

        # A 200 that reaches this line already passed the fidelity gate inside
        # _call — re-running check_fidelity here would re-derive a decision
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

        def _dial(mid: str, source: str) -> tuple[bool, Attempt, Optional[dict]]:
            """One upstream call that counts only if the body is usable.
            Fidelity failures return a payload but must not end the cascade:
            the client would receive output its own parser rejects. This one
            predicate is the whole cascade — main loop and re-probe retries
            share it, so neither arm can drift into accepting prose."""
            payload, att = self._call(
                mid, body, op_class, source=source,
                timeout=min(self.cfg.request_timeout_s, _remaining()),
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
            ok, att, payload = _dial(model_id, "request")
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
        # Gated on a non-empty ranking: this arm refreshes state FOR RANKED
        # MODELS. When the floors excluded everything (whole-pool episode)
        # there is nothing to refresh and each busy probe burns its full
        # timeout — live-measured 2026-08-24 as six ~45s probes eating the
        # whole 280s budget BEFORE the sweep could dial once. The sweep below
        # is strictly stronger there: it dials the real body directly.
        if probe_messages and order:
            fresh: list[str] = []
            for model_id in candidates:
                if len(fresh) >= self.cfg.reprobe_top_n or _remaining() <= 1.0:
                    break
                b = self.index.breaker_get(model_id)
                # Only 'gone' is truly terminal. 'down' AND 'auth' models are
                # re-probed here: this arm exists because persisted state may be
                # stale, and a restored credential is exactly that case.
                if b["state"] == "gone":
                    continue
                if self.probe(model_id, op_class, probe_messages,
                              timeout=min(self.cfg.probe_timeout_s, _remaining())):
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
                ok, att, payload = _dial(model_id, "request")
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
                if left <= 1.0:
                    result.attempts.append(
                        Attempt(model_id, "skipped-budget", None,
                                "cascade budget exhausted")
                    )
                    break
                payload, att = self._call(
                    model_id, body, op_class, source="sweep",
                    timeout=min(self.cfg.request_timeout_s, left),
                    request_id=request_id,
                )
                result.attempts.append(att)   # EVERY dial is recorded
                swept_any = True
                swept_count += 1
                if payload is not None and att.status == OK:
                    # Fidelity gate already ran inside _call: a 200 here is a
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
