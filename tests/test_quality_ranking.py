"""Quality-first ranking: the best model that is actually up, right now.

These tests encode the OBJECTIVE, not the implementation. The mesh keeps
hindsight alive on free NIM endpoints where models go overloaded or vanish
unpredictably; we do not need the world's best model, but a better model beats
a worse one, and availability beats both.

Each test states the real-world situation it stands for. Re-derived from the
live 2026-08-09 failure: 23 scored retain models, every score inside
[49.9, 50.0], ranking collapsed to alphabetical, and gemma-4-31b-it holding
rank 0 at p95 44.3s over 0.4s challengers.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_mesh.index import Index, OK  # noqa: E402
from model_mesh.quality import (  # noqa: E402
    BUCKET_FAILING,
    BUCKET_HEALTHY,
    BUCKET_OVERLOADED,
    BUCKET_UNKNOWN,
    availability_bucket,
    rank_key,
    tier,
)
from model_mesh.router import Router, RouterConfig  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _index(tmp) -> Index:
    return Index(tmp / f"q-{time.time_ns()}.db")


def _router(idx, **cfg) -> Router:
    return Router(idx, "http://unused.invalid/v1", lambda: "k", RouterConfig(**cfg))


def _sample(idx, mid, oc, latency_ms, status=OK, n=1):
    for _ in range(n):
        idx.record(mid, oc, "probe", status, latency_ms)


# -- the quality prior ------------------------------------------------------

@check("parameter count sets the tier")
def t_params():
    assert tier("openai/gpt-oss-120b") > tier("openai/gpt-oss-20b")
    assert tier("meta/llama-3.1-70b-instruct") > tier("meta/llama-3.1-8b-instruct")
    assert tier("meta/llama-3.2-1b-instruct") < tier("meta/llama-3.1-8b-instruct")


@check("MoE ids tier on TOTAL params, not active")
def t_moe():
    # nemotron-3-ultra-550b-a55b is 550b total / 55b active. Matching the
    # active count would tier NIM's flagship as a mid-size model.
    assert tier("nvidia/nemotron-3-ultra-550b-a55b") == 5
    assert tier("nvidia/nemotron-3-super-120b-a12b") == 5


@check("version numbers are not read as parameter counts")
def t_version_not_params():
    # llama-3.1-8b must tier on 8, not 3.1.
    assert tier("meta/llama-3.1-8b-instruct") == tier("meta/llama-x-8b")


@check("param count beats the size adjective when both are present")
def t_params_beat_adjective():
    # "nano" is relative to the vendor's own lineup; 30b is a fact.
    assert tier("nvidia/nemotron-3-nano-30b-a3b") == 3


@check("size adjective is used when no count is published")
def t_adjective():
    assert tier("stepfun-ai/step-3.7-flash") < tier("mistralai/mistral-large-2")


@check("unknown-size ids sort MID, never bottom")
def t_unknown_mid():
    # glm-5.2 / minimax-m3 publish no size. Tiering them lowest would exclude
    # exactly the new frontier models we most want to pick up.
    assert tier("z-ai/glm-5.2") == 3
    assert tier("minimaxai/minimax-m3") == 3
    assert tier("z-ai/glm-5.2") > tier("meta/llama-3.2-1b-instruct")


@check("config override beats the heuristic")
def t_override():
    assert tier("meta/llama-3.2-1b-instruct", {"meta/llama-3.2-1b-instruct": 5}) == 5


# -- availability buckets ---------------------------------------------------

@check("no evidence is UNKNOWN, not failing")
def t_bucket_unknown():
    assert availability_bucket(None, 20_000.0) == BUCKET_UNKNOWN


@check("slow-but-working is OVERLOADED, not failing")
def t_bucket_overloaded():
    class S:
        success_rate, p95_ms = 1.0, 44_000.0
    assert availability_bucket(S(), 20_000.0) == BUCKET_OVERLOADED


@check("all-failures is FAILING")
def t_bucket_failing():
    class S:
        success_rate, p95_ms = 0.0, 900.0
    assert availability_bucket(S(), 20_000.0) == BUCKET_FAILING


# -- the ordering itself ----------------------------------------------------

@check("REGRESSION: strong model beats a faster weak one when both are healthy")
def t_quality_beats_speed(tmp):
    # THE BUG. Pure-latency ranking put llama-3.2-1b (0.4s) above gpt-oss-120b
    # (2.6s) for memory extraction. Both are healthy; the stronger one wins.
    idx = _index(tmp)
    r = _router(idx)
    _sample(idx, "openai/gpt-oss-120b", "retain", 2600.0)
    _sample(idx, "meta/llama-3.2-1b-instruct", "retain", 400.0)
    order = r.ranked(["meta/llama-3.2-1b-instruct", "openai/gpt-oss-120b"], "retain")
    assert order[0] == "openai/gpt-oss-120b", order


@check("availability outranks quality: overloaded flagship loses to a healthy small model")
def t_availability_first(tmp):
    # Andrew's morning scenario: the big model is up but buried under other
    # users' load. Uptime is the mesh's whole job, so we take the healthy one.
    idx = _index(tmp)
    r = _router(idx)
    _sample(idx, "nvidia/nemotron-3-ultra-550b-a55b", "retain", 44_000.0)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 800.0)
    order = r.ranked(
        ["nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.1-8b-instruct"], "retain"
    )
    assert order[0] == "meta/llama-3.1-8b-instruct", order


@check("an overloaded model recovers its rank when it speeds back up")
def t_recovers(tmp):
    # Overload is transient by nature, so demotion must be too. Nothing marks
    # or un-marks the model; the ordering just follows the fresh measurement.
    idx = _index(tmp)
    r = _router(idx)
    _sample(idx, "openai/gpt-oss-120b", "retain", 44_000.0)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 800.0)
    pool = ["openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct"]
    assert r.ranked(pool, "retain")[0] == "meta/llama-3.1-8b-instruct"
    # ...load clears, next probe is fast.
    _sample(idx, "openai/gpt-oss-120b", "retain", 2000.0, n=20)
    assert r.ranked(pool, "retain")[0] == "openai/gpt-oss-120b"


@check("unknown model outranks a measured-broken one")
def t_unknown_beats_broken(tmp):
    idx = _index(tmp)
    r = _router(idx)
    _sample(idx, "meta/llama-3.1-70b-instruct", "retain", 500.0, status="http-500", n=3)
    order = r.ranked(
        ["meta/llama-3.1-70b-instruct", "openai/gpt-oss-20b"], "retain"
    )
    assert order[0] == "openai/gpt-oss-20b", order


@check("ranking is deterministic — same evidence, same order, every call")
def t_deterministic(tmp):
    # Explicitly required: no randomized exploration arm in the routing path.
    idx = _index(tmp)
    r = _router(idx)
    pool = ["openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct",
            "z-ai/glm-5.2", "meta/llama-3.2-1b-instruct"]
    for m in pool:
        _sample(idx, m, "retain", 1000.0)
    orders = {tuple(r.ranked(pool, "retain")) for _ in range(25)}
    assert len(orders) == 1, orders


@check("ties break by name, never by dict insertion order")
def t_tiebreak(tmp):
    idx = _index(tmp)
    r = _router(idx)
    for m in ("b/model-8b", "a/model-8b"):
        _sample(idx, m, "retain", 1000.0)
    assert r.ranked(["b/model-8b", "a/model-8b"], "retain") == \
        r.ranked(["a/model-8b", "b/model-8b"], "retain")


@check("eligibility gates still bind — ranking cannot resurrect a broken model")
def t_gates_bind(tmp):
    # A tier-5 model that fails the success floor must not be promoted by its
    # quality prior. Quality orders candidates; it never admits them.
    idx = _index(tmp)
    r = _router(idx, min_samples_for_floor=4, min_success_rate=0.5)
    _sample(idx, "nvidia/nemotron-3-ultra-550b-a55b", "retain", 900.0,
            status="http-500", n=6)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 900.0)
    order = r.ranked(
        ["nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.1-8b-instruct"], "retain"
    )
    assert "nvidia/nemotron-3-ultra-550b-a55b" not in order, order


@check("FIXTURE CHECK: the old score really did collapse to a tie")
def t_fixture_reproduces(tmp):
    # Guards against a vacuous suite. Reproduces the LIVE 2026-08-09 shape:
    # a thin-evidence fast model pinned just under the prior (49.9), and a
    # heavily-sampled slow/jittery incumbent floored AT the prior (50.0) — a
    # 0.1 gap across a 100x latency difference, which is what made ranking
    # degenerate to the alphabetical tiebreak.
    #
    # Latencies must VARY: identical samples give pstdev 0, so the incumbent
    # scores on latency alone and the collapse does not reproduce. The real
    # incumbent had jitter 4.77.
    idx = _index(tmp)
    _sample(idx, "fast/model-1b", "retain", 400.0)
    for i in range(36):
        _sample(idx, "slow/model-120b", "retain", 2000.0 + (i % 6) * 14_000.0)
    a = idx.score("fast/model-1b", "retain").score
    b = idx.score("slow/model-120b", "retain").score
    assert abs(a - b) < 1.0, (a, b)
    assert b >= a, ("incumbent must have outranked the 100x faster model", a, b)


@check("rank_key is a pure function of (id, evidence)")
def t_pure():
    class S:
        success_rate, p95_ms = 1.0, 1000.0
    k1 = rank_key("openai/gpt-oss-120b", S(), 20_000.0)
    k2 = rank_key("openai/gpt-oss-120b", S(), 20_000.0)
    assert k1 == k2
    assert k1[0] == BUCKET_HEALTHY


def main():
    import tempfile
    passed = failed = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in CHECKS:
            try:
                fn(tmp) if fn.__code__.co_argcount else fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(CHECKS)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
