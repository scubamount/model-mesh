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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .index import Index, OK

TRANSIENT_CODES = {429, 500, 502, 503, 504}
AUTH_CODES = {401, 403}
GONE_CODES = {404, 410}


@dataclass
class RouterConfig:
    breaker_threshold: int = 3          # consecutive fails -> down
    breaker_cooldown_s: float = 120.0   # first cooldown
    breaker_cooldown_max_s: float = 1800.0
    max_attempts: int = 3               # candidates tried before the re-probe arm
    reprobe_top_n: int = 3              # models re-probed on total miss
    request_timeout_s: float = 120.0
    probe_timeout_s: float = 30.0


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
        api_key: str,
        cfg: Optional[RouterConfig] = None,
        # injectable for tests: (url, body, headers, timeout) -> (status, dict)
        transport: Optional[Callable] = None,
    ):
        self.index = index
        self.upstream_base = upstream_base.rstrip("/")
        self.api_key = api_key
        self.cfg = cfg or RouterConfig()
        self._transport = transport or self._http_post

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

    def _eligible(self, model_id: str) -> bool:
        b = self.index.breaker_get(model_id)
        if b["state"] in ("gone", "auth"):
            return False
        if b["state"] == "down":
            if time.time() >= b["cooldown_until"]:
                # cooldown expired -> recovering: admit one real request
                self.index.breaker_set(model_id, state="recovering")
                return True
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
            self.index.breaker_set(model_id, state="auth")
            self.index.record(model_id, op_class, source, status, ms, payload_chars)
            return None, Attempt(model_id, status, ms, "auth: check API key")
        # transient (incl. 598 network / 599 malformed body)
        self.index.record(model_id, op_class, source, status, ms, payload_chars)
        self._on_transient_fail(model_id)
        detail = str(payload.get("error", payload.get("detail", "")))[:200]
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
            if not self._eligible(m):
                continue
            s = self.index.score(m, op_class)
            (unknown if s is None else scored).append((m, s))
        scored.sort(key=lambda t: t[1].score, reverse=True)
        return [m for m, _ in scored] + [m for m, _ in unknown]

    # -- probe (used by the re-probe arm and discovery) ----------------------

    def probe(self, model_id: str, op_class: str, messages: list[dict]) -> bool:
        body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0,
            "stream": False,
        }
        payload, att = self._call(
            model_id, body, op_class, source="probe",
            timeout=self.cfg.probe_timeout_s,
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

        order = self.ranked(candidates, op_class)
        for model_id in order[: self.cfg.max_attempts]:
            payload, att = self._call(model_id, body, op_class, source="request")
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
                if len(fresh) >= self.cfg.reprobe_top_n:
                    break
                b = self.index.breaker_get(model_id)
                # gone/auth stay excluded, but 'down' models ARE re-probed here:
                # this arm exists precisely because breaker state may be stale.
                if b["state"] in ("gone", "auth"):
                    continue
                if self.probe(model_id, op_class, probe_messages):
                    fresh.append(model_id)
            for model_id in fresh:
                payload, att = self._call(model_id, body, op_class, source="request")
                result.attempts.append(att)
                if payload is not None:
                    result.ok, result.model_id, result.response = (
                        True, model_id, payload,
                    )
                    return result

        return result
