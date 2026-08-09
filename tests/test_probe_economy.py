"""Probe economy: probe what we would route to, stop probing the dead.

Probing is not free — every probe is a real request against a shared free NIM
endpoint, and the budget it burns is the budget a newly-released model needs.
Two rules follow from what probing is FOR (detecting overload and disappearance,
the only two things that change unpredictably):

  - a model the cascade would never dial has health worth knowing nothing about
  - a model that has not served a request in a week is retired, whatever the
    catalog still claims

Both are bounded, self-rebutting, and reversible: a dormant model gets one probe
per window, and a single success clears its dormancy with no separate
resurrection path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_mesh.discovery import DORMANT_AFTER_S, discover  # noqa: E402
from model_mesh.index import SCORE_WINDOW_S, Index, OK  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


class FakeRouter:
    """Records what was probed. Never touches the network."""

    def __init__(self, verdict=("pass", "")):
        self.probed = []
        self._verdict = verdict

    def probe_verdict(self, mid, oc, messages, timeout=None):
        self.probed.append((mid, oc))
        return self._verdict


CATALOG = [
    "nvidia/nemotron-3-ultra-550b-a55b",   # tier 5
    "openai/gpt-oss-120b",                 # tier 5
    "meta/llama-3.1-70b-instruct",         # tier 4
    "google/gemma-4-31b-it",               # tier 3
    "openai/gpt-oss-20b",                  # tier 3
    "nvidia/nvidia-nemotron-nano-9b-v2",   # tier 2
    "meta/llama-3.1-8b-instruct",          # tier 2
    "meta/llama-3.2-3b-instruct",          # tier 1
    "meta/llama-3.2-1b-instruct",          # tier 1
]

ALIASES = {"auto/retain": {"op_class": "retain", "include": [], "exclude": []}}


def _index(tmp) -> Index:
    idx = Index(tmp / f"pe-{time.time_ns()}.db")
    idx.sync_catalog("nim", set(CATALOG))
    return idx


def _run(idx, router, **kw):
    return discover(
        idx, router, "nim", "http://unused.invalid/v1", "key", ALIASES,
        probe_new=True,
        fetch=lambda *a, **k: set(CATALOG),
        log=lambda *a, **k: None,
        **kw,
    )


@check("probe_top_n probes only the best few candidates, not the whole catalog")
def t_top_n(tmp):
    idx, r = _index(tmp), FakeRouter()
    _run(idx, r, probe_top_n=3)
    assert len(r.probed) == 3, r.probed


@check("the few probed are the STRONGEST, not the alphabetically first")
def t_top_n_picks_best(tmp):
    # Alphabetical ordering is what starved openai/* of probes for weeks.
    idx, r = _index(tmp), FakeRouter()
    _run(idx, r, probe_top_n=3)
    probed = {m for m, _ in r.probed}
    assert "meta/llama-3.2-1b-instruct" not in probed, probed
    assert "nvidia/nemotron-3-ultra-550b-a55b" in probed, probed


@check("probe_top_n=None keeps probing everything (opt-in, no silent change)")
def t_top_n_optional(tmp):
    idx, r = _index(tmp), FakeRouter()
    _run(idx, r, probe_top_n=None)
    assert len(r.probed) == len(CATALOG), len(r.probed)


@check("a model that has failed for a week is not probed again")
def t_dormant_skipped(tmp):
    idx, r = _index(tmp), FakeRouter()
    dead = "openai/gpt-oss-120b"
    old = time.time() - (DORMANT_AFTER_S + 3600)
    with idx._lock:
        for _ in range(4):
            idx._conn.execute(
                "INSERT INTO samples (model_id, ts, op_class, source, latency_ms,"
                " status, payload_chars) VALUES (?,?,?,?,?,?,?)",
                (dead, old, "retain", "probe", None, "http-404", None),
            )
        idx._conn.commit()
    _run(idx, r, probe_top_n=None)
    assert dead not in {m for m, _ in r.probed}, r.probed


@check("a model that failed only recently IS still probed")
def t_recent_failure_still_probed(tmp):
    # Overload is temporary. Writing a model off after one bad hour is exactly
    # the failure mode we are avoiding — a popular model must not be excluded
    # for being popular.
    #
    # The failure must be aged past the SCORING window (else the staleness gate
    # skips the probe for an unrelated and correct reason: evidence is fresh, so
    # no probe is needed) but well inside the DORMANCY window. That gap is the
    # whole point: stale-but-not-dormant is precisely the state that earns a
    # re-probe.
    idx, r = _index(tmp), FakeRouter()
    mid = "openai/gpt-oss-120b"
    aged = time.time() - (SCORE_WINDOW_S + 3600)
    assert aged > time.time() - DORMANT_AFTER_S, "fixture must not be dormant"
    with idx._lock:
        idx._conn.execute(
            "INSERT INTO samples (model_id, ts, op_class, source, latency_ms,"
            " status, payload_chars) VALUES (?,?,?,?,?,?,?)",
            (mid, aged, "retain", "probe", None, "http-429", None),
        )
        idx._conn.commit()
    _run(idx, r, probe_top_n=None)
    assert mid in {m for m, _ in r.probed}, r.probed


@check("one success clears dormancy — no separate resurrection path")
def t_dormancy_self_rebuts(tmp):
    idx = _index(tmp)
    mid = "openai/gpt-oss-120b"
    old = time.time() - (DORMANT_AFTER_S + 3600)
    with idx._lock:
        idx._conn.execute(
            "INSERT INTO samples (model_id, ts, op_class, source, latency_ms,"
            " status, payload_chars) VALUES (?,?,?,?,?,?,?)",
            (mid, old, "retain", "probe", None, "http-404", None),
        )
        idx._conn.commit()
    assert idx.dormant_since(mid, DORMANT_AFTER_S) is not None
    idx.record(mid, "retain", "probe", OK, 900.0)
    assert idx.dormant_since(mid, DORMANT_AFTER_S) is None


@check("a never-sampled model is unknown, not dormant")
def t_never_sampled_not_dormant(tmp):
    idx = _index(tmp)
    assert idx.dormant_since("openai/gpt-oss-120b", DORMANT_AFTER_S) is None


@check("FIXTURE CHECK: without top_n the fixture really does probe all 9")
def t_fixture_not_vacuous(tmp):
    # If the catalog fixture were empty or the router never called, every
    # "probed fewer" assertion above would pass vacuously.
    idx, r = _index(tmp), FakeRouter()
    _run(idx, r)
    assert len(r.probed) == 9, len(r.probed)


def main():
    import tempfile
    passed = failed = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in CHECKS:
            try:
                fn(tmp)
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(CHECKS)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
