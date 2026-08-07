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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .index import Index, OK

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


@dataclass
class RouterConfig:
    breaker_threshold: int = 3          # consecutive fails -> down
    breaker_cooldown_s: float = 120.0   # first cooldown
    breaker_cooldown_max_s: float = 1800.0
    max_attempts: int = 3               # candidates tried before the re-probe arm
    reprobe_top_n: int = 3              # models re-probed on total miss
    request_timeout_s: float = 120.0
    probe_timeout_s: float = 30.0
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
    # Latency floor, sibling of min_success_rate. A model can pass the success
    # floor and still be unusable: llama-3.3-70b sat at 61% success with p95
    # 96.9s and stayed ranked #2 on retain, so two attempts exhausted the 240s
    # budget and the op wedged until the watchdog reset it (2026-08-07).
    # Default 75s: an attempt must be able to run twice inside total_budget_s
    # (2 x 75 = 150 < 240) so the cascade always has a real second try left.
    # Raise it only if total_budget_s rises too — audit-timeout-chain.py checks
    # the relationship.
    max_p95_ms_for_eligibility: float = 75_000.0
    # Whole-cascade budget. Must stay under the CLIENT's timeout or it gives up
    # mid-cascade and the failover never completes: hindsight's retain timeout
    # is 300s while 3 x 120s = 360s. Per-attempt timeout shrinks to fit what's
    # left, so the cascade always gets to try every candidate.
    total_budget_s: float = 240.0


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
            s = self.index.score(model_id, op_class)
            if (s is not None
                    and s.n >= self.cfg.min_samples_for_floor
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
        timeout: Optional[float] = None,
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
            self.index.record(model_id, op_class, source, OK, ms, payload_chars)
            self._on_success(model_id)
            return payload, Attempt(model_id, OK, ms)

        status = f"http-{status_code}"
        if status_code in GONE_CODES:
            self.index.mark_gone(model_id, status)
            self.index.record(model_id, op_class, source, status, ms, payload_chars)
            return None, Attempt(model_id, status, ms, "gone: EOL'd at request time")
        if status_code in AUTH_CODES:
            self.index.breaker_set(
                model_id, state="auth", consec_fails=0,
                cooldown_s=self.cfg.auth_cooldown_s,
                cooldown_until=time.time() + self.cfg.auth_cooldown_s,
            )
            self.index.record(model_id, op_class, source, status, ms, payload_chars)
            return None, Attempt(model_id, status, ms, "auth: check API key")
        if status_code in REJECT_CODES:
            # Recorded (so the success-rate floor sees it and stops picking this
            # model for this op_class) but NOT breaker-counted: the model is
            # healthy, it just refuses this shape of request.
            self.index.record(model_id, op_class, source, status, ms, payload_chars)
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
        self.index.record(model_id, op_class, source, status, ms, payload_chars)
        self._on_transient_fail(model_id)
        detail = str(payload.get("error", payload.get("detail", "")))[:200]
        logger.warning(
            "transient model=%s op_class=%s status=%s ms=%.0f detail=%s",
            model_id, op_class, status, ms, detail or "(empty)",
        )
        return None, Attempt(model_id, status, ms, detail)

    # -- ranking ------------------------------------------------------------

    def ranked(self, candidates: list[str], op_class: str) -> list[str]:
        """Eligible candidates, best stability score first.

        Unknowns (no samples in-window) sort AFTER scored models but stay in
        the list — a new model must be reachable through the cascade or it
        can never earn a score.
        """
        scored, unknown = [], []
        for m in candidates:
            if not self._eligible(m, op_class):
                continue
            s = self.index.score(m, op_class)
            (unknown if s is None else scored).append((m, s))
        scored.sort(key=lambda t: t[1].score, reverse=True)
        return [m for m, _ in scored] + [m for m, _ in unknown]

    # -- probe (used by the re-probe arm and discovery) ----------------------

    def probe(
        self, model_id: str, op_class: str, messages: list[dict],
        timeout: Optional[float] = None,
    ) -> bool:
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
        return payload is not None and att.status == OK

    # -- the cascade ---------------------------------------------------------

    def route(
        self,
        candidates: list[str],
        body: dict,
        op_class: str,
        probe_messages: Optional[list[dict]] = None,
    ) -> RouteResult:
        result = RouteResult(ok=False)
        deadline = time.monotonic() + self.cfg.total_budget_s

        def _remaining() -> float:
            return deadline - time.monotonic()

        order = self.ranked(candidates, op_class)
        for model_id in order[: self.cfg.max_attempts]:
            left = _remaining()
            if left <= 1.0:
                result.attempts.append(
                    Attempt(model_id, "skipped-budget", None,
                            "cascade budget exhausted")
                )
                break
            # Shrink the per-attempt timeout to what's left so the cascade never
            # overruns the client's own timeout mid-failover.
            payload, att = self._call(
                model_id, body, op_class, source="request",
                timeout=min(self.cfg.request_timeout_s, left),
            )
            result.attempts.append(att)
            if payload is not None:
                result.ok, result.model_id, result.response = True, model_id, payload
                return result

        # Total miss -> the re-probe arm. The index may be stale (models EOL'd,
        # provider-side incident cleared); measure NOW and try once more.
        result.reprobed = True
        if probe_messages:
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
            for model_id in fresh:
                left = _remaining()
                if left <= 1.0:
                    break
                payload, att = self._call(
                    model_id, body, op_class, source="request",
                    timeout=min(self.cfg.request_timeout_s, left),
                )
                result.attempts.append(att)
                if payload is not None:
                    result.ok, result.model_id, result.response = (
                        True, model_id, payload,
                    )
                    return result

        return result
