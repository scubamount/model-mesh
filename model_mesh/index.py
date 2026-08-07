"""model-mesh: the model index.

SQLite-backed persistent memory of every model ever seen, every probe and real
request outcome, and current circuit-breaker state. The router consults this on
every request; discovery updates it daily; real traffic updates it for free.

Design rules:
- Models are NEVER deleted. A model that vanishes gets `eol_at` set — EOL is
  data (the maverick 410 of 2026-07-27 is exactly what we want a record of).
- Real requests and synthetic probes land in the same `samples` table,
  distinguished by `source`. Traffic is telemetry.
- All writes go through this module; WAL mode so the daemon's reader threads
  never block the writer.
"""

from __future__ import annotations

import os
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          TEXT PRIMARY KEY,        -- e.g. openai/gpt-oss-120b
    provider    TEXT NOT NULL,           -- e.g. nim
    first_seen  REAL NOT NULL,           -- unix ts of first catalog appearance
    last_seen   REAL NOT NULL,           -- unix ts of latest catalog appearance
    eol_at      REAL,                    -- set when it vanishes / 404s / 410s
    eol_reason  TEXT                     -- 'catalog-drop' | 'http-410' | 'http-404'
);
CREATE TABLE IF NOT EXISTS samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     TEXT NOT NULL,
    ts           REAL NOT NULL,
    op_class     TEXT NOT NULL,          -- retain | consolidation | reflect | generic
    source       TEXT NOT NULL,          -- request | probe | discovery
    latency_ms   REAL,                   -- NULL when the call never returned
    status       TEXT NOT NULL,          -- 'ok' | 'http-429' | 'timeout' | 'parse-fail' | ...
    payload_chars INTEGER
);
CREATE INDEX IF NOT EXISTS idx_samples_model_ts ON samples (model_id, ts);
CREATE TABLE IF NOT EXISTS breaker (
    model_id       TEXT PRIMARY KEY,
    state          TEXT NOT NULL DEFAULT 'healthy',  -- healthy|down|recovering|auth|gone
    consec_fails   INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0,
    cooldown_s     REAL NOT NULL DEFAULT 0,
    updated_at     REAL NOT NULL DEFAULT 0
);
"""

OK = "ok"

# Samples required before a model's score is trusted at face value. Below this,
# the score is shrunk toward NEUTRAL_PRIOR so a single fast probe cannot outrank
# a model proven over dozens of real requests. 8 ~= two days of discovery probes
# plus live traffic for an actually-used model.
CONFIDENT_N = 8
# Mid-scale prior: a barely-measured model sorts among mid-ranked proven models,
# above unknowns but below anything with a real track record.
NEUTRAL_PRIOR = 50.0

# Thin evidence sorts just BELOW the prior so that a proven model scoring at
# the prior still outranks it. Without this margin the two tie at 50.0 and
# ordering falls to dict insertion order, which is alphabetical by model id —
# i.e. arbitrary. Small enough that newcomers still sort well above unknowns.
THIN_EVIDENCE_MARGIN = 0.1

# Statuses that mean "the provider understood the request and refused it" — a
# deterministic capability verdict, not a transient failure. Recorded by
# Router._call as http-<code>. Router derives its own copy from REJECT_CODES
# and a test asserts the two agree, because router imports index (not the
# reverse) and duplicating the set silently is how these drift.
REJECT_STATUSES = ("http-400", "http-413", "http-422")

# How long an unrebutted capability rejection excludes a model from an op_class
# before one retry is admitted. Mirrors EOL_RECHECK_S: a reject is strong
# evidence but never permanent, because NIM changes what it serves.
REJECT_RECHECK_S = 7 * 86400.0

# How long a model retired with a request-time 404/410 stays retired before one
# retry probe. NIM lists models before deploying them, so a 404 is often
# temporary — but it is still LISTED, so un-retiring it on catalog presence
# alone re-probes it every pass forever (measured: 17 of 18 EOL'd models were
# still in the catalog). One week trades a stale exclusion for ~1 probe/model/
# week instead of ~1/day.
EOL_RECHECK_S = 7 * 86400.0


@dataclass
class Score:
    model_id: str
    score: float          # 0-100, higher = better
    p95_ms: float
    jitter: float         # stdev/median
    spike_rate: float
    success_rate: float
    n: int


class Index:
    def __init__(self, db_path: str | Path):
        self.path = Path(os.path.expanduser(str(db_path)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- catalog ------------------------------------------------------------

    def sync_catalog(self, provider: str, live_ids: set[str]) -> dict:
        """Reconcile the index against a live catalog listing.

        Returns {'new': [...], 'eol': [...], 'returned': [...]}.
        `returned` = models previously marked EOL that reappeared (it happens;
        providers un-deprecate). Their eol_at is cleared but the history stays.
        """
        now = time.time()
        new, eol, returned = [], [], []
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, eol_at, eol_reason FROM models WHERE provider = ?",
                (provider,),
            )
            known = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
            for mid in sorted(live_ids):
                if mid not in known:
                    self._conn.execute(
                        "INSERT INTO models (id, provider, first_seen, last_seen)"
                        " VALUES (?,?,?,?)",
                        (mid, provider, now, now),
                    )
                    new.append(mid)
                else:
                    self._conn.execute(
                        "UPDATE models SET last_seen=? WHERE id=?", (now, mid)
                    )
                    eol_at, eol_reason = known[mid]
                    # Un-EOL a model retired for VANISHING from the catalog as
                    # soon as it reappears. A model retired because it 404s is
                    # still listed — the catalog advertises it, the provider
                    # will not serve it — so clearing its EOL on sight
                    # resurrects it every single pass: un-EOL -> re-probe ->
                    # 404 -> EOL -> repeat, forever. Measured 2026-08-08: 17 of
                    # 18 EOL'd models were still in the catalog, i.e. the whole
                    # retirement set was being re-probed daily for nothing.
                    #
                    # But 404 must not be permanent either: NIM lists models
                    # before deploying them, so today's 404 is often tomorrow's
                    # working model. Retry one such model only after
                    # EOL_RECHECK_S has elapsed, which costs a single probe per
                    # model per week instead of per day.
                    if eol_at is not None and (
                        eol_reason == "catalog-drop"
                        or (now - eol_at) >= EOL_RECHECK_S
                    ):
                        self._conn.execute(
                            "UPDATE models SET eol_at=NULL, eol_reason=NULL"
                            " WHERE id=?",
                            (mid,),
                        )
                        self._set_breaker_locked(mid, "healthy", 0, 0, 0)
                        returned.append(mid)
            for mid in set(known) - live_ids:
                # known[mid] is now (eol_at, eol_reason); the drop test is on
                # eol_at, not on the tuple. Comparing the tuple to None is
                # always False, which would silently stop retiring models that
                # genuinely vanished from the catalog.
                if known[mid][0] is None:
                    self._conn.execute(
                        "UPDATE models SET eol_at=?, eol_reason='catalog-drop'"
                        " WHERE id=?",
                        (now, mid),
                    )
                    self._set_breaker_locked(mid, "gone", 0, 0, now)
                    eol.append(mid)
            self._conn.commit()
        return {"new": new, "eol": eol, "returned": returned}

    def mark_gone(self, model_id: str, reason: str) -> None:
        """Request-time EOL: a 404/410 means gone NOW, not at next discovery."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE models SET eol_at=COALESCE(eol_at, ?), eol_reason=?"
                " WHERE id=?",
                (now, reason, model_id),
            )
            self._set_breaker_locked(model_id, "gone", 0, 0, now)
            self._conn.commit()

    def live_models(self, provider: Optional[str] = None) -> list[str]:
        q = "SELECT id FROM models WHERE eol_at IS NULL"
        args: tuple = ()
        if provider:
            q += " AND provider=?"
            args = (provider,)
        with self._lock:
            return [r[0] for r in self._conn.execute(q, args).fetchall()]

    # -- samples ------------------------------------------------------------

    def record(
        self,
        model_id: str,
        op_class: str,
        source: str,
        status: str,
        latency_ms: Optional[float],
        payload_chars: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO samples (model_id, ts, op_class, source, latency_ms,"
                " status, payload_chars) VALUES (?,?,?,?,?,?,?)",
                (model_id, time.time(), op_class, source, latency_ms, status,
                 payload_chars),
            )
            self._conn.commit()

    def last_sample_ts(self, model_id: str, op_class: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(ts) FROM samples WHERE model_id=? AND op_class=?",
                (model_id, op_class),
            ).fetchone()
        return row[0] or 0.0

    def unrebutted_reject(
        self, model_id: str, op_class: str, recheck_s: float = REJECT_RECHECK_S
    ) -> Optional[float]:
        """Timestamp of a capability rejection that no later success rebuts.

        A 400/413/422 means the provider parsed the request and refused it: the
        model cannot do this op_class in this shape, and retrying is guaranteed
        to fail the same way. Observed 2026-08-08 —
        nvidia/nemotron-mini-4b-instruct has a 4096-token context and every
        op_class asks for 4096 completion tokens, so it returned http-400
        ("maximum context length is 4096 tokens... you requested 6508") on all
        three probes and can never succeed.

        The success-rate floor structurally cannot catch this. It needs
        min_samples_for_floor (4) samples before it engages, but a model that
        deterministically rejects never accrues a 4th sample from real traffic
        — one attempt per cascade, and nothing about failing makes it get
        sampled again. So it sat at n=1, success_rate=0.0, eligible=True,
        wasting a cascade slot on every single memory operation, forever.

        Returns None when the model's most recent evidence for this op_class is
        not a reject (any later success rebuts it), or when the reject is older
        than recheck_s — NIM changes what it serves, so this is never permanent.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT status, ts FROM samples WHERE model_id=? AND op_class=?"
                " ORDER BY ts DESC LIMIT 1",
                (model_id, op_class),
            ).fetchone()
        if not row:
            return None
        status, ts = row
        if status not in REJECT_STATUSES:
            return None
        if (time.time() - ts) >= recheck_s:
            return None  # stale: admit one retry
        return ts

    # -- scoring ------------------------------------------------------------

    def score(
        self, model_id: str, op_class: str, window_s: float = 86400.0
    ) -> Optional[Score]:
        """FCM-inspired stability score over the sliding window.

        0.30 p95-latency + 0.30 jitter + 0.20 spike-rate + 0.20 success-rate.
        Latency components are normalized against a 30s ceiling — anything at
        or beyond that is 0. Returns None when there are no samples in-window
        (an unknown model is neither good nor bad; the router probes it first).
        """
        cutoff = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT latency_ms, status FROM samples"
                " WHERE model_id=? AND op_class=? AND ts>=?",
                (model_id, op_class, cutoff),
            ).fetchall()
        if not rows:
            return None
        oks = [r[0] for r in rows if r[1] == OK and r[0] is not None]
        n = len(rows)
        success_rate = len(oks) / n
        if not oks:
            # All failures in-window. p95 uses the 30s ceiling (not inf — the
            # value must survive JSON serialization in /mesh/status).
            return Score(model_id, 0.0, 30_000.0, 1.0, 1.0, 0.0, n)

        med = statistics.median(oks)
        p95 = sorted(oks)[max(0, int(len(oks) * 0.95) - 1)]
        jitter = (statistics.pstdev(oks) / med) if med > 0 else 1.0
        spikes = sum(1 for v in oks if v > 3 * med)
        spike_rate = spikes / len(oks)

        ceil_ms = 30_000.0
        lat_comp = max(0.0, 1.0 - (p95 / ceil_ms))
        jit_comp = max(0.0, 1.0 - min(jitter, 1.0))
        spk_comp = 1.0 - spike_rate
        raw = 100.0 * (
            0.30 * lat_comp + 0.30 * jit_comp + 0.20 * spk_comp
            + 0.20 * success_rate
        )

        # Confidence shrinkage. A single lucky probe is not evidence of
        # reliability: on 2026-08-08 a widened pool put llama-3.1-8b (n=1,
        # p95 0.4s, score 99.6) above gpt-oss-120b (n=20, p95 28.6s, score
        # 56.5), which would have handed production retain traffic to a model
        # measured exactly once. Jitter and spike-rate are also degenerate at
        # n=1 (pstdev of one sample is 0, so the model scores a perfect
        # consistency it has not demonstrated).
        #
        # Shrink toward a prior, then CAP the result at that prior while
        # evidence is thin. Two earlier attempts were both wrong:
        #
        #   raw*c + 50.0*(1-c)          lifted an n=1 model to 56.2, above
        #                               nemotron-super-49b (n=29, honest 51.8)
        #   raw*c + min(50.0,raw)*(1-c) same outcome — capping the PRIOR does
        #                               nothing when raw is huge (min(50,99)=50)
        #
        # An under-evidenced model must never outrank a well-evidenced one on
        # the strength of a single fast sample, so below CONFIDENT_N its score
        # is held at or under NEUTRAL_PRIOR. Newcomers still sort above
        # unknowns and converge to their true score as samples accumulate.
        #
        # The cap alone is one-sided, and that asymmetry is a bug (observed
        # 2026-08-08): it pins thin evidence AT the prior but lets a proven
        # model score BELOW it, so unknowns outrank proof. nemotron-super-49b-v1
        # — 36/36 consolidation successes, 100%, actively serving — scored 49.2
        # (p95 74.7s is legitimately slow) and fell to rank 13, behind eleven
        # n=1 models pinned at exactly 50.0. With max_candidates=8 it was
        # evicted from the cascade entirely: the one model PROVEN to do the job
        # could no longer be chosen for it.
        #
        # So a tie against thin evidence must break toward the proven model.
        # Ranking sorts by score alone, so the tiebreak has to live in the
        # score: hold thin evidence just BELOW the prior (not at it) and floor
        # a confident model at the prior. Both bounds are needed — dropping
        # either one restores the eviction.
        confidence = min(1.0, n / CONFIDENT_N)
        if confidence < 1.0:
            shrunk = raw * confidence + NEUTRAL_PRIOR * (1.0 - confidence)
            score = min(shrunk, NEUTRAL_PRIOR - THIN_EVIDENCE_MARGIN)
        else:
            # A model with CONFIDENT_N samples of real evidence ranks on its
            # merits, but never loses its cascade slot to an unproven model.
            score = max(raw, NEUTRAL_PRIOR)

        return Score(model_id, round(score, 1), p95, round(jitter, 3),
                     round(spike_rate, 3), round(success_rate, 3), n)

    # -- breaker ------------------------------------------------------------

    def _set_breaker_locked(
        self, model_id: str, state: str, consec: int, cooldown_s: float,
        cooldown_until: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO breaker (model_id, state, consec_fails, cooldown_until,"
            " cooldown_s, updated_at) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(model_id) DO UPDATE SET state=excluded.state,"
            " consec_fails=excluded.consec_fails,"
            " cooldown_until=excluded.cooldown_until,"
            " cooldown_s=excluded.cooldown_s, updated_at=excluded.updated_at",
            (model_id, state, consec, cooldown_until, cooldown_s, time.time()),
        )

    def breaker_get(self, model_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT state, consec_fails, cooldown_until, cooldown_s"
                " FROM breaker WHERE model_id=?",
                (model_id,),
            ).fetchone()
        if row is None:
            return {"state": "healthy", "consec_fails": 0,
                    "cooldown_until": 0.0, "cooldown_s": 0.0}
        return {"state": row[0], "consec_fails": row[1],
                "cooldown_until": row[2], "cooldown_s": row[3]}

    def breaker_set(self, model_id: str, **fields) -> None:
        cur = self.breaker_get(model_id)
        cur.update(fields)
        with self._lock:
            self._set_breaker_locked(
                model_id, cur["state"], cur["consec_fails"],
                cur["cooldown_s"], cur["cooldown_until"],
            )
            self._conn.commit()

    def breaker_all(self) -> dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_id, state, consec_fails, cooldown_until, cooldown_s"
                " FROM breaker"
            ).fetchall()
        return {
            r[0]: {"state": r[1], "consec_fails": r[2],
                   "cooldown_until": r[3], "cooldown_s": r[4]}
            for r in rows
        }
