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
                "SELECT id, eol_at FROM models WHERE provider = ?", (provider,)
            )
            known = {r[0]: r[1] for r in cur.fetchall()}
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
                    if known[mid] is not None:
                        self._conn.execute(
                            "UPDATE models SET eol_at=NULL, eol_reason=NULL"
                            " WHERE id=?",
                            (mid,),
                        )
                        self._set_breaker_locked(mid, "healthy", 0, 0, 0)
                        returned.append(mid)
            for mid in set(known) - live_ids:
                if known[mid] is None:
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
        # is held at or under NEUTRAL_PRIOR. A proven model scoring above the
        # prior therefore always wins; a proven model scoring below it has
        # genuinely earned that position. Newcomers still sort above unknowns
        # and converge to their true score as samples accumulate.
        confidence = min(1.0, n / CONFIDENT_N)
        if confidence < 1.0:
            shrunk = raw * confidence + NEUTRAL_PRIOR * (1.0 - confidence)
            score = min(shrunk, NEUTRAL_PRIOR)
        else:
            score = raw

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
