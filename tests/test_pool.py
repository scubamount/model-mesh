"""model-mesh: candidate-pool breadth and discovery backfill.

These guard the 2026-08-08 finding that the mesh could only ever reach 5 of
NIM's 102 live models. The failure was silent — nothing errored, the pool was
simply narrow, and it got NARROWER as NIM's catalog grew. Tests that assert
"the pool is wide" would rot the same way, so these assert the PROPERTY that
kept it narrow can't come back.
"""

from __future__ import annotations

import pytest

from model_mesh.config import DEFAULTS, _NON_TEXT
from model_mesh.discovery import candidates_for, discover, eligible_for_alias
from model_mesh.index import Index


# A slice of NIM's real 2026-08-08 catalog: text models that must be reachable,
# plus non-text heads that must not be.
TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-medium-3.5-128b",
    "deepseek-ai/deepseek-v4-pro",
    "meta/llama-3.1-70b-instruct",
    "google/gemma-4-31b-it",
    "minimaxai/minimax-m3",
    "stepfun-ai/step-3.7-flash",
    "writer/palmyra-creative-122b",
    "ibm/granite-3.0-8b-instruct",
]

NON_TEXT_MODELS = [
    "nvidia/nemotron-3-embed-1b",
    "nvidia/nv-embedqa-e5-v5",
    "snowflake/arctic-embed-l",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "adept/fuyu-8b",
    "microsoft/kosmos-2",
    "nvidia/neva-22b",
    "nvidia/vila",
    "meta/llama-guard-4-12b",
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nvidia/nemotron-4-340b-reward",
    "nvidia/nemoretriever-parse",
    "nvidia/nvclip",
    "writer/palmyra-med-70b",
    "nvidia/riva-translate-4b-instruct",
    "google/diffusiongemma-26b-a4b-it",
]


@pytest.fixture()
def index(tmp_path):
    return Index(tmp_path / "t.db")


def _seed(index, models):
    index.sync_catalog("nim", set(models))
    return index


@pytest.mark.parametrize("model", TEXT_MODELS)
def test_text_models_are_admitted_to_the_retain_pool(model):
    """The regression that mattered: a family whitelist excluded NIM's current
    flagships because their names were not on a list written months earlier."""
    cfg = DEFAULTS["aliases"]["auto/retain"]
    assert eligible_for_alias(model, cfg), (
        f"{model} is a general text model but was filtered out of the pool"
    )


@pytest.mark.parametrize("model", NON_TEXT_MODELS)
def test_non_text_models_are_excluded(model):
    cfg = DEFAULTS["aliases"]["auto/retain"]
    assert not eligible_for_alias(model, cfg), (
        f"{model} cannot serve a JSON text op_class but reached the pool"
    )


def test_no_alias_ships_an_include_whitelist():
    """An include list re-narrows the pool to names known at authoring time.
    This is the property that rotted; assert it cannot return by default."""
    for alias, cfg in DEFAULTS["aliases"].items():
        assert not cfg.get("include"), (
            f"{alias} ships a default include whitelist — the pool will silently "
            f"shrink as the provider's catalog grows"
        )


def test_operator_can_still_pin_a_pool_explicitly():
    """Exclusion-by-default must not remove the ability to pin deliberately."""
    cfg = {"include": ["gpt-oss"], "exclude": []}
    assert eligible_for_alias("openai/gpt-oss-120b", cfg)
    assert not eligible_for_alias("moonshotai/kimi-k2.6", cfg)


def test_candidates_for_does_not_truncate(index):
    """cap must not be applied before ranking: live_models() is in DB insertion
    order, so slicing here keeps an arbitrary subset and can drop the best
    model. Passing cap= must NOT shorten the result."""
    _seed(index, TEXT_MODELS)
    cfg = DEFAULTS["aliases"]["auto/retain"]
    full = candidates_for(index, "nim", cfg)
    capped = candidates_for(index, "nim", cfg, cap=3)
    assert len(full) == len(TEXT_MODELS)
    assert capped == full, "candidates_for truncated before ranking"


def test_ranked_then_capped_keeps_the_best_model(index):
    """End-to-end shape of the app call site: rank first, cap second — the best
    scorer survives even when it sits last in DB order."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    _seed(index, TEXT_MODELS)
    best = TEXT_MODELS[-1]  # last in insertion order
    # record() is (model, op_class, source, status, latency_ms) — status BEFORE
    # latency. Passing them swapped records every sample as failed, which the
    # success floor then rejects, and the model vanishes from ranked().
    for _ in range(5):
        index.record(best, "retain", "request", OK, 900.0, 100)
    for m in TEXT_MODELS[:3]:
        for _ in range(5):
            index.record(m, "retain", "request", OK, 25_000.0, 100)

    router = Router(index, "https://x/v1", "k", RouterConfig())
    pool = candidates_for(index, "nim", DEFAULTS["aliases"]["auto/retain"])
    top = router.ranked(pool, "retain")[:3]
    assert best in top, "ranking-then-capping dropped the fastest model"


def test_discovery_probes_unproven_models_not_just_new_ones(index, monkeypatch):
    """The bootstrap trap: models present at first sync are never 'new' again,
    so probing only report['new'] left them permanently unranked."""
    _seed(index, TEXT_MODELS)          # first sync: all are "new"
    report_new = index.sync_catalog("nim", set(TEXT_MODELS))["new"]
    assert report_new == [], "precondition: nothing is new on the second sync"

    probed: list[tuple[str, str]] = []

    class FakeRouter:
        def probe_verdict(self, mid, oc, msgs):
            probed.append((mid, oc))
            return "pass", ""

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    discover(index, FakeRouter(), "nim", "https://x/v1", "k",
             {"auto/retain": DEFAULTS["aliases"]["auto/retain"]},
             log=lambda *a: None)

    assert probed, "no unproven model was probed — pool can never widen"
    assert {m for m, _ in probed} == set(TEXT_MODELS)


def test_discovery_does_not_reprobe_models_that_have_evidence(index, monkeypatch):
    """Self-limiting: probing must be once-per-op_class, or the daily job burns
    CPU forever re-measuring what real traffic already maintains."""
    from model_mesh.index import OK

    _seed(index, TEXT_MODELS)
    for m in TEXT_MODELS:
        index.record(m, "retain", "request", OK, 1000.0, 100)

    probed = []

    class FakeRouter:
        def probe_verdict(self, mid, oc, msgs):
            probed.append(mid)
            return "pass", ""

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    discover(index, FakeRouter(), "nim", "https://x/v1", "k",
             {"auto/retain": DEFAULTS["aliases"]["auto/retain"]},
             log=lambda *a: None)

    assert probed == [], f"re-probed models that already have evidence: {probed}"


def test_probe_budget_bounds_a_single_pass(index, monkeypatch):
    """A catalog explosion must not turn the daily job into an unbounded burn."""
    _seed(index, TEXT_MODELS)
    probed = []

    class FakeRouter:
        def probe_verdict(self, mid, oc, msgs):
            probed.append(mid)
            return "pass", ""

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    discover(index, FakeRouter(), "nim", "https://x/v1", "k",
             {"auto/retain": DEFAULTS["aliases"]["auto/retain"]},
             max_probes=3, log=lambda *a: None)

    assert len(set(probed)) == 3, f"probe budget not enforced: {len(set(probed))}"


def test_non_text_filter_entries_are_anchored_enough():
    """A bare '-med' also matched mistral-medium-3.5-128b. Substring filters
    are dangerous; keep them from silently eating general models."""
    cfg = {"include": [], "exclude": _NON_TEXT}
    for model in TEXT_MODELS:
        assert eligible_for_alias(model, cfg), (
            f"_NON_TEXT entry is too broad — it swallowed {model}"
        )


# -- overload vs incapability -------------------------------------------------
# 2026-08-08: a backfill reported "20 failed fidelity". Zero had failed
# fidelity — 51 samples were http-404 (catalog lists a model NIM won't serve)
# and 15 were timeouts (model busy). A bare boolean probe collapsed
# "cannot do this" and "was busy just now" into one FAIL, which permanently
# excludes exactly the popular models we most want.


class _FakeAttempt:
    def __init__(self, status, detail=""):
        self.status = status
        self.detail = detail


def _router_returning(index, status, payload=None, detail=""):
    from model_mesh.router import Router, RouterConfig

    r = Router(index, "https://x/v1", "k", RouterConfig())
    r._call = lambda *a, **k: (payload, _FakeAttempt(status, detail))
    return r


def test_timeout_is_busy_not_unusable(index):
    """A model that times out is overloaded, NOT incapable."""
    r = _router_returning(index, "http-598", None, "TimeoutError")
    verdict, _ = r.probe_verdict("meta/llama-3.1-70b-instruct", "retain", [])
    assert verdict == "busy", f"timeout classified as {verdict!r}"


def test_rate_limit_is_busy_not_unusable(index):
    """429 is the definitive overload signal — never hold it against a model."""
    r = _router_returning(index, "http-429", None, "Too Many Requests")
    verdict, _ = r.probe_verdict("deepseek-ai/deepseek-v4-pro", "retain", [])
    assert verdict == "busy", f"429 classified as {verdict!r}"


def test_server_error_is_busy_not_unusable(index):
    r = _router_returning(index, "http-500", None, "Already borrowed")
    verdict, _ = r.probe_verdict("nvidia/nemotron-3-super-120b-a12b", "retain", [])
    assert verdict == "busy", f"500 classified as {verdict!r}"


def test_404_is_unusable_not_busy(index):
    """Listed in the catalog but not servable: permanent, and must be EOL'd so
    it is not re-probed forever."""
    r = _router_returning(index, "http-404", None, "Not Found")
    verdict, detail = r.probe_verdict("01-ai/yi-large", "retain", [])
    assert verdict == "unusable"
    assert "not servable" in detail


def test_busy_models_are_not_marked_gone(index, monkeypatch):
    """The regression that matters: a discovery pass where every model is
    merely overloaded must EOL nothing."""
    _seed(index, TEXT_MODELS)

    class BusyRouter:
        def probe_verdict(self, mid, oc, msgs):
            return "busy", "http-598: timeout"

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    report = discover(index, BusyRouter(), "nim", "https://x/v1", "k",
                      {"auto/retain": DEFAULTS["aliases"]["auto/retain"]},
                      log=lambda *a: None)

    assert report["unusable"] == [], "a busy model was permanently excluded"
    assert set(report["deferred_busy"]) == set(TEXT_MODELS)
    assert len(index.live_models("nim")) == len(TEXT_MODELS), (
        "overloaded models were EOL'd — they must be retried on a later pass"
    )


def test_busy_models_are_reprobed_on_the_next_pass(index, monkeypatch):
    """Deferral must be real: a model busy today is probed again tomorrow."""
    _seed(index, TEXT_MODELS)
    calls = []

    class FlakyRouter:
        def __init__(self):
            self.first = True

        def probe_verdict(self, mid, oc, msgs):
            calls.append(mid)
            return ("busy", "http-429") if self.first else ("pass", "")

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    router = FlakyRouter()
    alias = {"auto/retain": DEFAULTS["aliases"]["auto/retain"]}

    discover(index, router, "nim", "https://x/v1", "k", alias, log=lambda *a: None)
    first_round = len(calls)
    router.first = False
    discover(index, router, "nim", "https://x/v1", "k", alias, log=lambda *a: None)

    assert len(calls) > first_round, "busy models were never retried"


def test_unusable_404_is_marked_gone_once(index, monkeypatch):
    """The complement: a genuinely non-servable model IS retired, so passes
    don't re-probe it forever."""
    _seed(index, TEXT_MODELS)

    class GoneRouter:
        def probe_verdict(self, mid, oc, msgs):
            return "unusable", "not servable (http-404)"

    monkeypatch.setattr("model_mesh.discovery.fetch_catalog",
                        lambda *a, **k: set(TEXT_MODELS))
    report = discover(index, GoneRouter(), "nim", "https://x/v1", "k",
                      {"auto/retain": DEFAULTS["aliases"]["auto/retain"]},
                      log=lambda *a: None)

    assert set(report["unusable"]) == set(TEXT_MODELS)
    assert index.live_models("nim") == [], "non-servable models stayed live"


def test_boolean_probe_still_works_for_callers(index):
    """probe() keeps its bool contract; only `pass` is True."""
    assert _router_returning(index, "http-598").probe("m", "retain", []) is False
    assert _router_returning(index, "http-404").probe("m", "retain", []) is False


# -- confidence weighting ------------------------------------------------------
# 2026-08-08: widening the pool put llama-3.1-8b (n=1, p95 0.4s, score 99.6)
# above gpt-oss-120b (n=20, p95 28.6s, score 56.5). Payload sizes were
# comparable (12221 vs 12066 median chars), so this was not a probe-realism
# problem — it was one observation outweighing twenty.


def test_single_sample_does_not_outrank_a_proven_model(index):
    from model_mesh.index import OK

    for _ in range(20):
        index.record("proven", "retain", "request", OK, 28_000.0, 12_000)
    index.record("newcomer", "retain", "probe", OK, 400.0, 12_000)

    proven = index.score("proven", "retain")
    newcomer = index.score("newcomer", "retain")
    assert proven.score > newcomer.score, (
        f"n=1 newcomer ({newcomer.score}) outranked n=20 proven ({proven.score})"
    )


def test_newcomer_still_beats_an_unknown(index):
    """Shrinkage must not freeze the pool: a probed model still sorts above a
    never-measured one, or nothing new can ever earn its way in."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    index.sync_catalog("nim", {"newcomer", "unknown"})
    index.record("newcomer", "retain", "probe", OK, 400.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["unknown", "newcomer"], "retain")
    assert order[0] == "newcomer", f"probed model did not outrank unknown: {order}"


def test_confidence_converges_with_evidence(index):
    """A genuinely fast model must reach the top once it has real evidence —
    shrinkage is a delay, not a permanent penalty."""
    from model_mesh.index import OK, CONFIDENT_N

    for _ in range(20):
        index.record("slow", "retain", "request", OK, 28_000.0, 12_000)
    for _ in range(CONFIDENT_N):
        index.record("fast", "retain", "request", OK, 400.0, 12_000)

    assert index.score("fast", "retain").score > index.score("slow", "retain").score


def test_shrinkage_does_not_rescue_a_failing_model(index):
    """The prior must not lift a model that is measurably bad."""
    from model_mesh.index import OK

    for _ in range(20):
        index.record("good", "retain", "request", OK, 2_000.0, 12_000)
    for _ in range(20):
        index.record("bad", "retain", "request", "http-500", None, 12_000)

    assert index.score("good", "retain").score > index.score("bad", "retain").score


def test_shrinkage_never_lifts_a_newcomer_above_a_proven_model(index):
    """Regression: shrinking toward a FIXED 50.0 prior lifted an n=1 model above
    nemotron-super-49b (n=29, p95 66.2s, raw 53.5) — a real consolidation
    workhorse — because the prior sat above the workhorse's honest score. The
    prior must be capped by the raw score so shrinkage is only ever a penalty."""
    from model_mesh.index import OK, NEUTRAL_PRIOR
    import random

    # Proven but genuinely slow AND variable — the measured nemotron-super-49b
    # consolidation profile (n=29, median 35.8s, jitter 0.550, p95 66.2s ->
    # raw 53.5). Identical latencies give zero jitter and a 70.0 score, and a
    # uniform spread only reaches jitter 0.355; neither reproduces the live
    # bug. The jitter component is what drags the honest score under the prior,
    # so the spread has to match what was actually observed.
    rnd = random.Random(7)
    for _ in range(29):
        # lognormal-ish: mostly near the median with a long slow tail
        v = rnd.choice([rnd.uniform(8_000.0, 20_000.0)] * 2
                       + [rnd.uniform(30_000.0, 45_000.0)] * 3
                       + [rnd.uniform(60_000.0, 90_000.0)])
        index.record("workhorse", "consolidation", "request", OK, v, 12_000)
    # newcomer with one fast sample
    index.record("newcomer", "consolidation", "probe", OK, 500.0, 12_000)

    work = index.score("workhorse", "consolidation")
    new = index.score("newcomer", "consolidation")
    # The live failure had the workhorse at raw 53.5 — only slightly above the
    # 50.0 prior. That small margin is the whole bug: shrinking a fast n=1
    # model toward 50.0 lands it at ~56, which clears 53.5. Assert the regime
    # (workhorse close to the prior), not an arbitrary threshold.
    assert abs(work.score - NEUTRAL_PRIOR) < 10.0, (
        f"precondition: this regression needs the workhorse's honest score near "
        f"the prior ({NEUTRAL_PRIOR}); got {work.score}"
    )
    assert work.score > new.score, (
        f"n=1 newcomer ({new.score}) outranked n=29 proven workhorse "
        f"({work.score}) — the prior is subsidizing thin evidence"
    )


def test_shrinkage_is_never_a_subsidy(index):
    """General property: a model's shrunk score never EXCEEDS its raw score."""
    from model_mesh.index import OK, CONFIDENT_N, NEUTRAL_PRIOR

    for latency in (300.0, 5_000.0, 29_000.0, 66_000.0):
        mid = f"m{int(latency)}"
        index.record(mid, "retain", "probe", OK, latency, 12_000)
        s = index.score(mid, "retain")
        # reconstruct the unshrunk score with a fully-confident model
        for _ in range(CONFIDENT_N):
            index.record(f"{mid}_full", "retain", "request", OK, latency, 12_000)
        full = index.score(f"{mid}_full", "retain")
        assert s.score <= full.score + 0.05, (
            f"{mid}: n=1 score {s.score} exceeds fully-evidenced {full.score}"
        )


# -- EOL churn -----------------------------------------------------------------
# 2026-08-08: 17 of 18 EOL'd models were STILL IN NIM's catalog. sync_catalog
# cleared eol_at for any model present in the listing, so every pass ran
# un-EOL -> re-probe -> 404 -> EOL -> repeat. The retirement set was being
# re-probed daily, forever, for nothing — the exact CPU burn this design is
# supposed to avoid.


def test_404_model_still_listed_is_not_resurrected_every_pass(index):
    """A model retired for 404 stays retired while it remains in the catalog."""
    catalog = {"a/model", "b/model"}
    index.sync_catalog("nim", catalog)
    index.mark_gone("a/model", "http-404")
    assert "a/model" not in index.live_models("nim")

    # the model is STILL listed by the provider on the next pass
    report = index.sync_catalog("nim", catalog)
    assert "a/model" not in report["returned"], (
        "404-retired model was resurrected while still listed — every pass "
        "will re-probe it forever"
    )
    assert "a/model" not in index.live_models("nim")


def test_catalog_drop_model_returns_immediately_when_relisted(index):
    """The complement: a model that genuinely VANISHED must come straight back
    when the provider re-lists it. Providers do un-deprecate."""
    index.sync_catalog("nim", {"a/model", "b/model"})
    index.sync_catalog("nim", {"b/model"})           # a/model vanishes
    assert "a/model" not in index.live_models("nim")

    report = index.sync_catalog("nim", {"a/model", "b/model"})  # relisted
    assert "a/model" in report["returned"]
    assert "a/model" in index.live_models("nim")


def test_404_model_is_rechecked_after_the_recheck_window(index):
    """404 must not be permanent — NIM lists models before deploying them, so
    today's 404 is often tomorrow's working model."""
    import time as _t
    from model_mesh.index import EOL_RECHECK_S

    catalog = {"a/model"}
    index.sync_catalog("nim", catalog)
    index.mark_gone("a/model", "http-404")

    # backdate the retirement past the recheck window
    with index._lock:
        index._conn.execute(
            "UPDATE models SET eol_at=? WHERE id=?",
            (_t.time() - EOL_RECHECK_S - 60, "a/model"),
        )
        index._conn.commit()

    report = index.sync_catalog("nim", catalog)
    assert "a/model" in report["returned"], (
        "404-retired model was never rechecked — a model NIM later deploys "
        "can never come back"
    )


def test_vanished_model_is_still_retired(index):
    """Guard the tuple refactor: known[mid] became (eol_at, eol_reason), and
    `known[mid] is None` is always False on a tuple — which would silently stop
    retiring models that genuinely left the catalog."""
    index.sync_catalog("nim", {"a/model", "b/model"})
    report = index.sync_catalog("nim", {"b/model"})
    assert "a/model" in report["eol"], "a vanished model was not retired"
    assert index.live_models("nim") == ["b/model"]


# -- deterministic capability rejection ---------------------------------------
# 2026-08-08: nvidia/nemotron-mini-4b-instruct has a 4096-token context window
# and every op_class asks for max_tokens=4096, so NIM returned http-400 ("maximum
# context length is 4096 tokens. However, you requested 6508") on all three
# probes. It can never succeed, yet it sat at n=1, success_rate=0.0,
# eligible=True — burning a cascade slot on every memory operation.
#
# The success-rate floor structurally cannot catch this: it engages only at
# min_samples_for_floor (4) samples, and a deterministically-rejecting model
# never earns a 4th sample. One attempt per cascade, and failing does not cause
# resampling. A floor that needs evidence can't police a model that never
# accrues any.


def test_reject_is_its_own_verdict_not_busy(index):
    """400 is a capability answer, not overload. Classifying it `busy` meant
    re-probing a model whose answer cannot change."""
    r = _router_returning(index, "http-400", None, "maximum context length")
    verdict, detail = r.probe_verdict("nvidia/nemotron-mini-4b-instruct", "retain", [])
    assert verdict == "rejected", f"400 classified as {verdict!r}"
    assert "http-400" in detail


@pytest.mark.parametrize("status", ["http-400", "http-413", "http-422"])
def test_all_reject_codes_are_rejected_verdicts(index, status):
    r = _router_returning(index, status, None, "refused")
    verdict, _ = r.probe_verdict("some/model", "retain", [])
    assert verdict == "rejected", f"{status} classified as {verdict!r}"


def test_reject_status_sets_agree_across_modules():
    """index.REJECT_STATUSES and router.REJECT_STATUS_NAMES are the same set
    expressed twice (router imports index, not the reverse). Silently drifting
    apart would leave a code rejected by one layer and admitted by the other."""
    from model_mesh.index import REJECT_STATUSES
    from model_mesh.router import REJECT_CODES, REJECT_STATUS_NAMES

    assert set(REJECT_STATUSES) == REJECT_STATUS_NAMES
    assert REJECT_STATUS_NAMES == {f"http-{c}" for c in REJECT_CODES}


def test_deterministically_rejecting_model_is_ineligible(index):
    """The live bug, at the layer all real traffic routes through."""
    from model_mesh.router import Router, RouterConfig

    mid = "nvidia/nemotron-mini-4b-instruct"
    index.record(mid, "retain", "probe", "http-400", 267.0, 11653)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    s = index.score(mid, "retain")
    assert s.n < RouterConfig().min_samples_for_floor, (
        "precondition: this bug exists BECAUSE n stays under the floor "
        f"(n={s.n}); if the floor could see it, no new gate would be needed")
    assert not r._eligible(mid, "retain"), (
        "a model that deterministically rejects retain must not stay eligible")


def test_reject_is_scoped_to_the_op_class(index):
    """A reject is about the request shape, not the model. nemotron-super-49b
    rejects retain (11.5k-12.6k char payloads) while serving consolidation at
    100% — excluding it wholesale would lose the consolidation winner."""
    from model_mesh.router import Router, RouterConfig

    mid = "nvidia/llama-3.3-nemotron-super-49b-v1"
    index.record(mid, "retain", "request", "http-400", 300.0, 12618)
    index.record(mid, "consolidation", "request", "ok", 35_000.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    assert not r._eligible(mid, "retain")
    assert r._eligible(mid, "consolidation"), (
        "a reject on one op_class must not exclude the model from another")


def test_a_later_success_rebuts_the_reject(index):
    """Never permanent: if the model later serves the op_class, the reject is
    stale evidence and must stop gating."""
    from model_mesh.router import Router, RouterConfig

    mid = "some/model"
    index.record(mid, "retain", "probe", "http-400", 250.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())
    assert not r._eligible(mid, "retain")

    index.record(mid, "retain", "request", "ok", 9_000.0, 12000)
    assert r._eligible(mid, "retain"), "a later success must rebut the reject"


def test_reject_goes_stale_and_admits_a_retry(index):
    """NIM changes what it serves. An old reject must not exclude forever."""
    import time as _t

    from model_mesh.index import REJECT_RECHECK_S

    mid = "some/model"
    index.record(mid, "retain", "probe", "http-400", 250.0, 12000)
    with index._lock:
        index._conn.execute(
            "UPDATE samples SET ts=? WHERE model_id=?",
            (_t.time() - REJECT_RECHECK_S - 60, mid),
        )
        index._conn.commit()

    assert index.unrebutted_reject(mid, "retain") is None, (
        "a reject older than REJECT_RECHECK_S must go stale")


def test_transient_failure_is_not_treated_as_a_reject(index):
    """The inverse error: excluding a model for being overloaded is exactly the
    bug this whole verdict split exists to prevent."""
    from model_mesh.router import Router, RouterConfig

    mid = "openai/gpt-oss-120b"
    index.record(mid, "retain", "request", "http-503", 900.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())
    assert index.unrebutted_reject(mid, "retain") is None
    assert r._eligible(mid, "retain"), "503 is overload, not a capability verdict"
