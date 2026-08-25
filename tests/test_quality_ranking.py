"""Quality-first ranking: the best model that is actually up, right now.

These tests encode the OBJECTIVE, not the implementation. The mesh keeps its
client alive on free NIM endpoints where models go overloaded or vanish
unpredictably; we do not need the world's best model, but a better model beats
a worse one, and availability beats both.

Each test states the real-world situation it stands for. Re-derived from the
live 2026-08-09 failure: 23 scored retain models, every score inside
[49.9, 50.0], ranking collapsed to alphabetical, and gemma-4-31b-it holding
rank 0 at p95 44.3s over 0.4s challengers.

These were `@check`-decorated functions collected into a module-level CHECKS
list and run by a `main()` that nothing invoked — pytest collected ZERO of them,
so every guarantee below was decoration while the suite reported green (audit
2026-08-25, the third file found with this defect after test_probe_economy.py
and test_stale_evidence_reprobe.py). They are plain pytest tests now.
"""

import time

from model_mesh.index import Index, OK
from model_mesh.quality import (
    BUCKET_FAILING,
    BUCKET_HEALTHY,
    BUCKET_OVERLOADED,
    BUCKET_UNKNOWN,
    availability_bucket,
    rank_key,
    tier,
)
from model_mesh.router import Router, RouterConfig


def _index(tmp) -> Index:
    return Index(tmp / f"q-{time.time_ns()}.db")


def _router(idx, **cfg) -> Router:
    return Router(idx, "http://unused.invalid/v1", lambda: "k", RouterConfig(**cfg))


def _sample(idx, mid, oc, latency_ms, status=OK, n=1):
    # In production every sampled id is already catalog-listed; tests must
    # reproduce that precondition explicitly (record() refuses phantoms).
    idx.ensure_model(mid)
    for _ in range(n):
        idx.record(mid, oc, "probe", status, latency_ms)


# -- the quality prior ------------------------------------------------------

def test_parameter_count_sets_the_tier():
    assert tier("openai/gpt-oss-120b") > tier("openai/gpt-oss-20b")
    assert tier("meta/llama-3.1-70b-instruct") > tier("meta/llama-3.1-8b-instruct")
    assert tier("meta/llama-3.2-1b-instruct") < tier("meta/llama-3.1-8b-instruct")


def test_moe_ids_tier_on_total_params_not_active():
    # nemotron-3-ultra-550b-a55b is 550b total / 55b active. Matching the
    # active count would tier NIM's flagship as a mid-size model.
    assert tier("nvidia/nemotron-3-ultra-550b-a55b") == 5
    assert tier("nvidia/nemotron-3-super-120b-a12b") == 5


def test_version_numbers_are_not_read_as_parameter_counts():
    # llama-3.1-8b must tier on 8, not 3.1.
    assert tier("meta/llama-3.1-8b-instruct") == tier("meta/llama-x-8b")


def test_param_count_beats_the_size_adjective():
    # "nano" is relative to the vendor's own lineup; 30b is a fact.
    assert tier("nvidia/nemotron-3-nano-30b-a3b") == 3


def test_size_adjective_is_used_when_no_count_is_published():
    assert tier("stepfun-ai/step-3.7-flash") < tier("mistralai/mistral-large-2")


def test_unknown_size_ids_sort_mid_never_bottom():
    # glm-5.2 / minimax-m3 publish no size. Tiering them lowest would exclude
    # exactly the new frontier models we most want to pick up.
    assert tier("z-ai/glm-5.2") == 3
    assert tier("minimaxai/minimax-m3") == 3
    assert tier("z-ai/glm-5.2") > tier("meta/llama-3.2-1b-instruct")


def test_config_override_beats_the_heuristic():
    assert tier("meta/llama-3.2-1b-instruct", {"meta/llama-3.2-1b-instruct": 5}) == 5


# -- availability buckets ---------------------------------------------------

def test_no_evidence_is_unknown_not_failing():
    assert availability_bucket(None, 20_000.0) == BUCKET_UNKNOWN


def test_slow_but_working_is_overloaded_not_failing():
    class S:
        success_rate, p95_ms = 1.0, 44_000.0
    assert availability_bucket(S(), 20_000.0) == BUCKET_OVERLOADED


def test_all_failures_is_failing():
    class S:
        success_rate, p95_ms = 0.0, 900.0
    assert availability_bucket(S(), 20_000.0) == BUCKET_FAILING


# -- the ordering itself ----------------------------------------------------

def test_regression_strong_model_beats_faster_weak_one_when_both_healthy(tmp_path):
    # THE BUG. Pure-latency ranking put llama-3.2-1b (0.4s) above gpt-oss-120b
    # (2.6s) for memory extraction. Both are healthy; the stronger one wins.
    idx = _index(tmp_path)
    r = _router(idx)
    _sample(idx, "openai/gpt-oss-120b", "retain", 2600.0)
    _sample(idx, "meta/llama-3.2-1b-instruct", "retain", 400.0)
    order = r.ranked(["meta/llama-3.2-1b-instruct", "openai/gpt-oss-120b"], "retain")
    assert order[0] == "openai/gpt-oss-120b", order


def test_availability_outranks_quality(tmp_path):
    # Morning scenario: the big model is up but buried under other users' load.
    # Uptime is the mesh's whole job, so we take the healthy one.
    idx = _index(tmp_path)
    r = _router(idx)
    _sample(idx, "nvidia/nemotron-3-ultra-550b-a55b", "retain", 44_000.0)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 800.0)
    order = r.ranked(
        ["nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.1-8b-instruct"], "retain"
    )
    assert order[0] == "meta/llama-3.1-8b-instruct", order


def test_overloaded_model_recovers_rank_when_it_speeds_back_up(tmp_path):
    # Overload is transient by nature, so demotion must be too. Nothing marks
    # or un-marks the model; the ordering just follows the fresh measurement.
    idx = _index(tmp_path)
    r = _router(idx)
    _sample(idx, "openai/gpt-oss-120b", "retain", 44_000.0)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 800.0)
    pool = ["openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct"]
    assert r.ranked(pool, "retain")[0] == "meta/llama-3.1-8b-instruct"
    # ...load clears, next probe is fast.
    _sample(idx, "openai/gpt-oss-120b", "retain", 2000.0, n=20)
    assert r.ranked(pool, "retain")[0] == "openai/gpt-oss-120b"


def test_unknown_model_outranks_a_measured_broken_one(tmp_path):
    idx = _index(tmp_path)
    r = _router(idx)
    _sample(idx, "meta/llama-3.1-70b-instruct", "retain", 500.0, status="http-500", n=3)
    order = r.ranked(["meta/llama-3.1-70b-instruct", "openai/gpt-oss-20b"], "retain")
    assert order[0] == "openai/gpt-oss-20b", order


def test_ranking_is_deterministic(tmp_path):
    # Explicitly required: no randomized exploration arm in the routing path.
    idx = _index(tmp_path)
    r = _router(idx)
    pool = ["openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct",
            "z-ai/glm-5.2", "meta/llama-3.2-1b-instruct"]
    for m in pool:
        _sample(idx, m, "retain", 1000.0)
    orders = {tuple(r.ranked(pool, "retain")) for _ in range(25)}
    assert len(orders) == 1, orders


def test_ties_break_by_name_not_dict_insertion_order(tmp_path):
    idx = _index(tmp_path)
    r = _router(idx)
    for m in ("b/model-8b", "a/model-8b"):
        _sample(idx, m, "retain", 1000.0)
    assert r.ranked(["b/model-8b", "a/model-8b"], "retain") == \
        r.ranked(["a/model-8b", "b/model-8b"], "retain")


def test_eligibility_gates_still_bind(tmp_path):
    # A tier-5 model that fails the success floor must not be promoted by its
    # quality prior. Quality orders candidates; it never admits them.
    idx = _index(tmp_path)
    r = _router(idx, min_samples_for_floor=4, min_success_rate=0.5)
    _sample(idx, "nvidia/nemotron-3-ultra-550b-a55b", "retain", 900.0,
            status="http-500", n=6)
    _sample(idx, "meta/llama-3.1-8b-instruct", "retain", 900.0)
    order = r.ranked(
        ["nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.1-8b-instruct"], "retain"
    )
    assert "nvidia/nemotron-3-ultra-550b-a55b" not in order, order


def test_no_blended_score_field_exists(tmp_path):
    """Guards the 2026-08-09 fix at its root: `Score` must hold measured
    evidence and NO aggregate.

    A single blended float used to live here. It ranked availability while
    presenting itself as quality, and had collapsed to a 0.1-wide band across a
    100x latency spread — which is what made ranking degenerate to the
    alphabetical tiebreak. Ranking is `quality.rank_key()` reading these fields
    directly; re-adding an aggregate re-creates both failures.

    This check replaces a `@check`-era fixture assertion that read
    `idx.score(...).score` — a field deleted by that very fix. Because the file
    was never collected by pytest, the assertion sat broken and unnoticed
    instead of failing (audit 2026-08-25).
    """
    idx = _index(tmp_path)
    _sample(idx, "fast/model-1b", "retain", 400.0)
    s = idx.score("fast/model-1b", "retain")
    assert s is not None
    assert not hasattr(s, "score"), (
        "Score regained a blended aggregate; ranking must read p95/jitter/"
        "success_rate directly via quality.rank_key()"
    )
    for f in ("p95_ms", "jitter", "spike_rate", "success_rate", "n"):
        assert hasattr(s, f), f"Score lost measured field {f!r}"


def test_ranking_does_not_collapse_across_a_100x_latency_spread(tmp_path):
    """The live 2026-08-09 shape, asserted on the ORDER rather than on a float.

    A thin-evidence fast model against a heavily-sampled slow/jittery incumbent.
    Latencies must VARY: identical samples give pstdev 0, so the incumbent would
    score on latency alone and the pathology would not reproduce (the real
    incumbent had jitter 4.77). Under the old blended score these two landed
    0.1 apart and the tiebreak decided the lane; now the overloaded incumbent
    must lose outright.
    """
    idx = _index(tmp_path)
    r = _router(idx)
    _sample(idx, "fast/model-1b", "retain", 400.0)
    for i in range(36):
        _sample(idx, "slow/model-120b", "retain", 2000.0 + (i % 6) * 14_000.0)
    order = r.ranked(["slow/model-120b", "fast/model-1b"], "retain")
    assert order[0] == "fast/model-1b", (
        "an overloaded incumbent outranked a healthy challenger — the 2026-08-09 "
        "collapse is back", order,
    )


def test_rank_key_is_a_pure_function_of_id_and_evidence():
    class S:
        success_rate, p95_ms = 1.0, 1000.0
    k1 = rank_key("openai/gpt-oss-120b", S(), 20_000.0)
    k2 = rank_key("openai/gpt-oss-120b", S(), 20_000.0)
    assert k1 == k2
    assert k1[0] == BUCKET_HEALTHY
