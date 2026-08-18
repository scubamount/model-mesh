"""request_id groups the attempts of one cascade.

Without it, client-visible reliability is not computable from mesh.db: attempts
must be clustered by timestamp, and concurrent traffic (hindsight issues
parallel retains) makes that cluster meaningless. Measured 2026-08-18, the
timestamp method reported 5.00% client-visible failure over 24h; its worst
"failed cascade" was one timeout surrounded by 40+ successes belonging to other
requests.
"""
import sqlite3
import uuid

from model_mesh.index import Index


def _index(tmp_path):
    return Index(str(tmp_path / "t.db"))


def test_record_persists_request_id(tmp_path):
    idx = _index(tmp_path)
    rid = uuid.uuid4().hex
    idx.record("m/a", "retain", "request", "http-598", 90000.0, request_id=rid)
    idx.record("m/b", "retain", "request", "ok", 1200.0, request_id=rid)

    rows = idx._conn.execute(
        "SELECT model_id, status, request_id FROM samples ORDER BY id"
    ).fetchall()
    assert [r[2] for r in rows] == [rid, rid], "both attempts must carry the cascade id"


def test_one_failed_attempt_does_not_make_a_failed_request(tmp_path):
    """The exact misreading the column exists to prevent."""
    idx = _index(tmp_path)
    rid = uuid.uuid4().hex
    # one cascade: first model times out, second succeeds -> CLIENT SUCCEEDED
    idx.record("m/slow", "retain", "request", "http-598", 90000.0, request_id=rid)
    idx.record("m/fast", "retain", "request", "ok", 900.0, request_id=rid)
    # an unrelated concurrent request that also failed once then succeeded
    rid2 = uuid.uuid4().hex
    idx.record("m/slow", "retain", "request", "http-598", 90000.0, request_id=rid2)
    idx.record("m/fast", "retain", "request", "ok", 950.0, request_id=rid2)

    rows = idx._conn.execute(
        "SELECT request_id, SUM(status='ok') FROM samples"
        " WHERE request_id IS NOT NULL GROUP BY request_id"
    ).fetchall()
    assert len(rows) == 2, "two distinct client requests"
    assert all(r[1] >= 1 for r in rows), "each request got an answer -> 0% client-visible failure"

    # attempt-level failure rate is 50%, and that is NOT the client's experience
    total = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    failed = idx._conn.execute("SELECT COUNT(*) FROM samples WHERE status!='ok'").fetchone()[0]
    assert failed / total == 0.5
    assert len(rows) == 2 and all(r[1] >= 1 for r in rows)


def test_migration_adds_column_to_existing_db(tmp_path):
    """An existing mesh.db must gain the column without losing rows."""
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE samples ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " model_id TEXT NOT NULL, ts REAL NOT NULL, op_class TEXT NOT NULL,"
        " source TEXT NOT NULL, latency_ms REAL, status TEXT NOT NULL,"
        " payload_chars INTEGER);"
    )
    con.execute(
        "INSERT INTO samples (model_id, ts, op_class, source, latency_ms, status)"
        " VALUES ('legacy/m', 1.0, 'retain', 'request', 10.0, 'ok')"
    )
    con.commit()
    con.close()

    idx = Index(str(db))  # must migrate, not crash
    cols = {r[1] for r in idx._conn.execute("PRAGMA table_info(samples)")}
    assert "request_id" in cols, "migration must add the column to an existing db"

    kept = idx._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert kept == 1, "migration must not drop existing samples"
    legacy_rid = idx._conn.execute(
        "SELECT request_id FROM samples WHERE model_id='legacy/m'"
    ).fetchone()[0]
    assert legacy_rid is None, "pre-existing rows stay ungrouped (NULL), never faked into one request"
