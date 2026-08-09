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

# Samples required before a model is treated as having a real track record.
# Used by the pool-breadth invariant to distinguish "proven" from "measured
# once". 8 ~= two days of discovery probes plus live traffic for an actively
# used model.
#
# This no longer feeds a score. Until 2026-08-09 it drove confidence shrinkage
# toward NEUTRAL_PRIOR (50.0) with a THIN_EVIDENCE_MARGIN tiebreak; that whole
# mechanism went with the blended float. See Index.score().
CONFIDENT_N = 8

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

# Sliding window for Index.score(). ONE definition, because discovery's
# re-probe threshold must track it: score() ignores samples older than this, so
# any model whose newest sample falls outside the window scores None and ranks
# as "unknown" regardless of how good it measured. If discovery's staleness
# threshold were larger than this window, models would expire out of scoring
# faster than they get re-probed and the pool would silently collapse to
# whichever model happens to carry live traffic.
SCORE_WINDOW_S = 86400.0


@dataclass
class Score:
    """Measured evidence, not a verdict.

    Deliberately holds no aggregate: ranking is `quality.rank_key()`, which
    reads these fields directly. A single blended float lived here until
    2026-08-09 and had to go — it ranked availability while presenting itself
    as quality, and had collapsed to a 0.1-wide band across a 100x latency
    spread. Adding one back re-creates both failures.
    """
    model_id: str
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

    def dormant_since(
        self, model_id: str, dormant_after_s: float
    ) -> Optional[float]:
        """Timestamp of the last success, when a model has failed ever since.

        Returns None if the model succeeded within `dormant_after_s`, or has no
        samples at all (never tried is not dormant — it is unknown, and unknown
        is exactly what discovery exists to resolve).

        NIM lists models it will not actually serve: measured 2026-08-09, 17 of
        18 models the index had EOL'd on request-time 404s were still present in
        the catalog. Catalog presence therefore cannot answer "is this model
        real", and without a dormancy test every one of those is re-probed on
        every pass forever — burning the probe budget that newly-released models
        need, on models that have not served a request in weeks.

        Deliberately reads the sample log rather than a flag: a model that
        starts working again rebuts its own dormancy the moment one success
        lands, with no separate resurrection path to maintain, and no state that
        can disagree with the evidence.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(CASE WHEN status=? THEN ts END), MAX(ts)"
                " FROM samples WHERE model_id=?",
                (OK, model_id),
            ).fetchone()
        if not row or row[1] is None:
            return None  # never sampled: unknown, not dormant
        last_ok, last_any = row
        now = time.time()
        if last_ok is None:
            # Never once succeeded. Dormant only after the window has elapsed
            # since we started trying, so a model discovered minutes ago is not
            # written off before it has had a fair chance.
            return last_any if (now - last_any) >= dormant_after_s else None
        if (now - last_ok) >= dormant_after_s:
            return last_ok
        return None

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
        self, model_id: str, op_class: str, window_s: float = SCORE_WINDOW_S
    ) -> Optional[Score]:
        """Measured evidence for one model on one op_class, over the window.

        Returns the raw measurements — p95, jitter, spike-rate, success-rate,
        sample count — and does NOT reduce them to a single number. Ranking is
        `quality.rank_key()`, which reads these fields directly.

        There used to be a blended float here (0.30 p95 + 0.30 jitter + 0.20
        spike + 0.20 success, with confidence shrinkage toward a neutral prior).
        It was removed on 2026-08-09 with the ranking rewrite: every term
        measured availability, none measured quality, so a 1b model outranked a
        120b whenever it answered faster. Worse, the blend had no resolution —
        23 live models spanning p95 0.4s-44.3s all landed inside [49.9, 50.0]
        and ordering fell through to the alphabetical tiebreak.

        Returns None when there are no samples in-window: an unknown model is
        neither good nor bad, and `availability_bucket` sorts it above anything
        measured failing but below anything measured healthy.
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
            # success_rate 0.0 is what puts this model in BUCKET_FAILING.
            return Score(model_id, 30_000.0, 1.0, 1.0, 0.0, n)

        med = statistics.median(oks)
        p95 = sorted(oks)[max(0, int(len(oks) * 0.95) - 1)]
        jitter = (statistics.pstdev(oks) / med) if med > 0 else 1.0
        spikes = sum(1 for v in oks if v > 3 * med)
        spike_rate = spikes / len(oks)

        return Score(model_id, p95, round(jitter, 3),
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
