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
        index.ensure_model(best)
        index.record(best, "retain", "request", OK, 900.0, 100)
    for m in TEXT_MODELS[:3]:
        for _ in range(5):
            index.ensure_model(m)
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
        index.ensure_model(m)
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
    r.dial = lambda *a, **k: (payload, _FakeAttempt(status, detail))
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


def test_probe_verdict_distinguishes_busy_from_unusable(index):
    """A bare bool collapsed "overloaded right now" into "cannot do the job" —
    the 2026-08-08 bug that permanently excluded popular models. The verdict
    must keep them distinct: 5xx -> busy (retryable), 404 -> unusable."""
    assert _router_returning(index, "http-598").probe_verdict("m", "retain", [])[0] == "busy"
    assert _router_returning(index, "http-404").probe_verdict("m", "retain", [])[0] == "unusable"


# -- confidence weighting ------------------------------------------------------
# 2026-08-08: widening the pool put llama-3.1-8b (n=1, p95 0.4s, score 99.6)
# above gpt-oss-120b (n=20, p95 28.6s, score 56.5). Payload sizes were
# comparable (12221 vs 12066 median chars), so this was not a probe-realism
# problem — it was one observation outweighing twenty.


# -- evidence vs. entitlement --------------------------------------------------
# These six tests were written against the blended stability score and its
# confidence shrinkage (raw*c + NEUTRAL_PRIOR*(1-c), capped by
# THIN_EVIDENCE_MARGIN). That machinery was deleted on 2026-08-09 along with the
# score itself, so every one of them failed on an ImportError or AttributeError.
#
# They are retargeted rather than removed. The arithmetic they asserted is gone,
# but the GUARANTEE behind them is not: a model measured once must not displace
# one with a real track record, and a probed model must still be able to earn
# its way in. Under quality-first ranking that guarantee is enforced structurally
# by rank_key rather than by score arithmetic, so the assertions move to
# Router.ranked() — the thing that actually decides.


def test_single_sample_does_not_outrank_a_proven_model(index):
    """A newcomer with one lucky fast probe must not displace a proven model.

    Under the old score this was arithmetic (shrinkage toward a prior). Now it
    is structural: both are the same quality tier, so ordering falls to the
    availability bucket, and a model measured once at 400ms is HEALTHY exactly
    like the proven one — the tiebreak is p95, which is the honest comparison.
    The failure this guards against is the newcomer winning while the proven
    model is healthy and the newcomer is NOT.
    """
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    # proven: healthy and fast. newcomer: one sample, but overloaded.
    for _ in range(20):
        index.ensure_model("nvidia/proven-31b")
        index.ensure_model("nvidia/proven-31b")
        index.ensure_model("nvidia/proven-31b")
        index.record("nvidia/proven-31b", "retain", "request", OK, 2_000.0, 12_000)
    index.ensure_model("nvidia/newcomer-31b")
    index.ensure_model("nvidia/newcomer-31b")
    index.ensure_model("nvidia/newcomer-31b")
    index.record("nvidia/newcomer-31b", "retain", "probe", OK, 44_000.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["nvidia/newcomer-31b", "nvidia/proven-31b"], "retain")
    assert order[0] == "nvidia/proven-31b", (
        f"an overloaded n=1 newcomer outranked a healthy proven model: {order}"
    )


def test_newcomer_still_beats_an_unknown(index):
    """The pool must not freeze: a probed model sorts above a never-measured
    one, or nothing new can ever earn its way in."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    index.sync_catalog("nim", {"newcomer", "unknown"})
    index.ensure_model("newcomer")
    index.ensure_model("newcomer")
    index.ensure_model("newcomer")
    index.record("newcomer", "retain", "probe", OK, 400.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["unknown", "newcomer"], "retain")
    assert order[0] == "newcomer", f"probed model did not outrank unknown: {order}"


def test_evidence_is_not_entitlement(index):
    """A model that USED to work does not keep a slot once it degrades.

    This is the inverse of the test above and the reason the old shrinkage had
    to go: it floored a proven model at the prior, which meant a well-evidenced
    model kept its rank while measurably degrading. 2026-08-09, live: gemma
    (n=36, 97% success) held rank 0 at p95 44.3s while challengers measured
    under 3s.
    """
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    for _ in range(36):
        index.ensure_model("nvidia/incumbent-31b")
        index.ensure_model("nvidia/incumbent-31b")
        index.ensure_model("nvidia/incumbent-31b")
        index.record("nvidia/incumbent-31b", "retain", "request", OK, 44_000.0, 12_000)
    for _ in range(3):
        index.ensure_model("nvidia/challenger-31b")
        index.ensure_model("nvidia/challenger-31b")
        index.ensure_model("nvidia/challenger-31b")
        index.record("nvidia/challenger-31b", "retain", "request", OK, 2_500.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["nvidia/incumbent-31b", "nvidia/challenger-31b"], "retain")
    assert order[0] == "nvidia/challenger-31b", (
        f"a degraded incumbent kept rank 0 on track record alone: {order}"
    )


def test_a_failing_model_ranks_below_everything_measured_working(index):
    """Measured failure is the worst bucket — no amount of evidence rescues it."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    for _ in range(20):
        index.ensure_model("nvidia/good-31b")
        index.ensure_model("nvidia/good-31b")
        index.ensure_model("nvidia/good-31b")
        index.record("nvidia/good-31b", "retain", "request", OK, 2_000.0, 12_000)
    for _ in range(20):
        index.ensure_model("nvidia/bad-31b")
        index.ensure_model("nvidia/bad-31b")
        index.ensure_model("nvidia/bad-31b")
        index.record("nvidia/bad-31b", "retain", "request", "http-500", None, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["nvidia/bad-31b", "nvidia/good-31b"], "retain")
    assert order[0] == "nvidia/good-31b", f"a failing model ranked first: {order}"
    # Stronger than "ranks last": eligibility drops it from the cascade
    # entirely, so it never burns an attempt. Asserting last place would have
    # been satisfied by a one-element list — check membership explicitly.
    assert "nvidia/bad-31b" not in order, (
        f"a model measured 0% success over 20 samples is still eligible: {order}"
    )


def test_quality_outranks_speed_at_equal_availability(index):
    """The defect that motivated the rewrite: a small fast model beating a large
    one. Both healthy -> the stronger model wins even though it is slower."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    index.ensure_model("meta/llama-3.1-8b-instruct")

    index.ensure_model("meta/llama-3.1-8b-instruct")

    index.ensure_model("meta/llama-3.1-8b-instruct")

    index.record("meta/llama-3.1-8b-instruct", "retain", "probe", OK, 400.0, 12_000)
    index.ensure_model("openai/gpt-oss-120b")
    index.ensure_model("openai/gpt-oss-120b")
    index.ensure_model("openai/gpt-oss-120b")
    index.record("openai/gpt-oss-120b", "retain", "probe", OK, 3_000.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(["meta/llama-3.1-8b-instruct", "openai/gpt-oss-120b"], "retain")
    assert order[0] == "openai/gpt-oss-120b", (
        f"a 8b model outranked a 120b on speed alone: {order}"
    )


def test_ranking_has_resolution_across_a_latency_spread(index):
    """The collapse that hid the old bug: 23 models spanning p95 0.4s-44.3s all
    scored inside [49.9, 50.0], so ordering fell through to the alphabetical
    tiebreak. Assert the ranking actually SEPARATES its inputs — a metric that
    cannot distinguish its inputs is not measuring them."""
    from model_mesh.index import OK
    from model_mesh.router import Router, RouterConfig

    # same tier, same bucket, latencies an order of magnitude apart, and named
    # so that alphabetical order is the REVERSE of the correct order.
    ids = ["nvidia/a-slow-31b", "nvidia/b-mid-31b", "nvidia/c-fast-31b"]
    for mid, lat in zip(ids, (18_000.0, 6_000.0, 900.0)):
        index.ensure_model(mid)
        index.record(mid, "retain", "probe", OK, lat, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    order = r.ranked(list(ids), "retain")
    assert order == ["nvidia/c-fast-31b", "nvidia/b-mid-31b", "nvidia/a-slow-31b"], (
        f"ranking did not separate a 20x latency spread (alphabetical fallback?): {order}"
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
    index.ensure_model(mid)
    index.record(mid, "retain", "probe", "http-400", 267.0, 11653)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    s = index.score(mid, "retain")
    assert s.n < RouterConfig().min_samples_for_floor, (
        "precondition: this bug exists BECAUSE n stays under the floor "
        f"(n={s.n}); if the floor could see it, no new gate would be needed")
    assert not r.eligible(mid, "retain"), (
        "a model that deterministically rejects retain must not stay eligible")


def test_thin_evidence_repeated_failure_is_ineligible(index):
    """The live 2026-08-16 bug: quality-first ranking put a measurably
    failing model first in line for memory traffic.

    openai/gpt-oss-120b sat at n=3 / success_rate 0.333 — below
    min_samples_for_floor, so the sustained-failure floor never engaged, while
    its tier-5 size sorted it to #1 on both auto/retain and auto/reflect. Two
    failures out of three is evidence, not thin data.
    """
    from model_mesh.router import Router, RouterConfig

    mid = "openai/gpt-oss-120b"
    index.ensure_model(mid)
    index.record(mid, "retain", "chat", "ok", 1740.0, 900)
    index.ensure_model(mid)
    index.record(mid, "retain", "chat", "http-500", 1740.0, 900)
    index.ensure_model(mid)
    index.record(mid, "retain", "chat", "http-500", 1740.0, 900)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    s = index.score(mid, "retain")
    assert s.n < RouterConfig().min_samples_for_floor, (
        "precondition: the bug exists BECAUSE n stays under the sample floor "
        f"(n={s.n})")
    assert s.success_rate < 0.5, f"precondition: majority-failing (got {s.success_rate})"
    assert not r.eligible(mid, "retain"), (
        "a model that has failed twice of three must not take live memory "
        "traffic just because it has not yet earned a 4th sample")


def test_thin_evidence_floor_spares_a_single_failure(index):
    """One failure is noise. The thin-evidence floor must not evict on it, or
    every model gets knocked out by its first hiccup and the pool collapses."""
    from model_mesh.router import Router, RouterConfig

    mid = "nvidia/nemotron-3-super-120b-a12b"
    index.ensure_model(mid)
    index.record(mid, "retain", "chat", "http-500", 1200.0, 900)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    assert r.eligible(mid, "retain"), (
        "a single failure is not evidence of a failing model")


def test_reject_is_scoped_to_the_op_class(index):
    """A reject is about the request shape, not the model. nemotron-super-49b
    rejects retain (11.5k-12.6k char payloads) while serving consolidation at
    100% — excluding it wholesale would lose the consolidation winner."""
    from model_mesh.router import Router, RouterConfig

    mid = "nvidia/llama-3.3-nemotron-super-49b-v1"
    index.ensure_model(mid)
    index.record(mid, "retain", "request", "http-400", 300.0, 12618)
    index.ensure_model(mid)
    index.record(mid, "consolidation", "request", "ok", 35_000.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())

    assert not r.eligible(mid, "retain")
    assert r.eligible(mid, "consolidation"), (
        "a reject on one op_class must not exclude the model from another")


def test_a_later_success_rebuts_the_reject(index):
    """Never permanent: if the model later serves the op_class, the reject is
    stale evidence and must stop gating."""
    from model_mesh.router import Router, RouterConfig

    mid = "some/model"
    index.ensure_model(mid)
    index.record(mid, "retain", "probe", "http-400", 250.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())
    assert not r.eligible(mid, "retain")

    index.ensure_model(mid)
    index.record(mid, "retain", "request", "ok", 9_000.0, 12000)
    assert r.eligible(mid, "retain"), "a later success must rebut the reject"


def test_reject_goes_stale_and_admits_a_retry(index):
    """NIM changes what it serves. An old reject must not exclude forever."""
    import time as _t

    from model_mesh.index import REJECT_RECHECK_S

    mid = "some/model"
    index.ensure_model(mid)
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
    index.ensure_model(mid)
    index.record(mid, "retain", "request", "http-503", 900.0, 12000)
    r = Router(index, "https://x/v1", "k", RouterConfig())
    assert index.unrebutted_reject(mid, "retain") is None
    assert r.eligible(mid, "retain"), "503 is overload, not a capability verdict"


# -- proven models keep their cascade slot -------------------------------------
# 2026-08-08: the confidence cap was one-sided. It pinned thin evidence AT
# NEUTRAL_PRIOR but let a proven model score BELOW it, so unknowns outranked
# proof. nvidia/llama-3.3-nemotron-super-49b-v1 — 36/36 consolidation successes,
# 100%, actively serving — scored 49.2 (p95 74.7s is legitimately slow) and fell
# to rank 13 behind eleven n=1 models sitting at exactly 50.0. The old
# max_candidates=8 pre-router cap
# then evicted it from the cascade: the one model PROVEN to do the job could no
# longer be chosen for it.


def _seed_live_consolidation_regime(index, mid):
    """Replay the ACTUAL measured latencies, not an invented distribution.

    These are the 36 real consolidation samples from
    nvidia/llama-3.3-nemotron-super-49b-v1 on 2026-08-08 (median 31.4s, p95
    74.7s, jitter 0.694), which score 49.2 raw — just BELOW NEUTRAL_PRIOR.
    That sub-prior position is the entire bug: it is what let eleven n=1
    models pinned at 50.0 outrank a model with 36/36 successes.

    Two earlier synthetic fixtures were vacuous and both passed with the bug
    reinstated. One drew a tail to 80s (p95 77.8s), over the 75s eligibility
    floor, so the model was dropped before ranking ran. The next scored 55.3 —
    ABOVE the prior — so max(raw, NEUTRAL_PRIOR) was a no-op and removing the
    floor changed nothing. Guessing at a distribution kept missing the narrow
    regime where the bug lives; replaying the measurement cannot.
    """
    for seconds in [
        2.6, 5.4, 6.1, 8.3, 11.2, 12.4, 13.1, 15.0, 15.4, 16.2, 16.8, 18.3,
        21.4, 25.1, 26.3, 27.2, 28.4, 29.1, 34.0, 34.6, 36.2, 38.1, 40.3,
        41.2, 42.4, 43.1, 47.2, 50.3, 53.1, 54.2, 55.4, 56.1, 66.3, 74.7,
        83.2, 88.4,
    ]:
        index.ensure_model(mid)
        index.record(mid, "consolidation", "request", "ok", seconds * 1000.0, 12_000)


def test_proven_model_is_not_evicted_by_unproven_ones(index):
    """The live regression, restated for quality-first ranking.

    ORIGINAL FORM (2026-08-08): a 36/36 proven model scored 49.2 on the old
    blended float, eleven n=1 newcomers were pinned at exactly 50.0, and with
    max_candidates=8 the one model proven to do the job was evicted from the
    cascade entirely.

    That arithmetic is gone — ranking is now (availability, quality, latency)
    and there is no prior for thin evidence to hide behind. The GUARANTEE is
    what survives, and it is stronger here: a model with a real track record
    must never be pushed out of the cascade by models that have merely been
    probed once.

    The fixture keeps the proven model HEALTHY (its p95 is inside the overload
    threshold). The original fixture's 74.7s p95 is, under the new rules,
    genuinely an overloaded model, and preferring healthy newcomers to it is
    the intended behavior rather than a regression — so testing eviction needs
    a fixture where availability does not decide the outcome.
    """
    from model_mesh.router import Router, RouterConfig

    proven = "nvidia/llama-3.3-nemotron-super-49b-v1"   # tier 4
    for _ in range(36):
        index.ensure_model(proven)
        index.record(proven, "consolidation", "request", "ok", 3_000.0, 12_000)

    s = index.score(proven, "consolidation")
    # SCORE_RECENT_N caps evidence at the newest 20 of the 36 samples —
    # recency-capped scoring (2026-08-30); all are ok either way.
    assert s.success_rate == 1.0 and s.n == 20
    assert s.p95_ms < RouterConfig().overload_p95_ms, (
        "precondition: the proven model must be HEALTHY, or it is demoted for "
        "being overloaded and the test proves nothing about eviction")

    # Eleven single-probe newcomers, each far faster but a weaker tier.
    newcomers = [f"vendor/newcomer-{i:02d}-3b" for i in range(11)]
    for m in newcomers:
        index.ensure_model(m)
        index.record(m, "consolidation", "probe", "ok", 500.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    ranked = r.ranked([proven] + newcomers, "consolidation")

    assert ranked[0] == proven, (
        f"a proven, healthy, higher-tier model must outrank single-probe "
        f"newcomers; got {ranked[:3]}")
    assert proven in ranked[:8], (
        "a proven model must sit in the head of the ranking that the main "
        "cascade dials (max_attempts=8)")


def test_thin_evidence_still_cannot_outrank_a_better_proven_model(index):
    """A lucky fast probe on a weak model does not beat a stronger model.

    Same guarantee as before the ranking change, and now it holds for a
    structural reason rather than a numerical one: llama-3.1-8b is tier 2 and
    gpt-oss-120b is tier 5, so no amount of thin fast evidence promotes the
    smaller model while both are healthy.
    """
    from model_mesh.router import Router, RouterConfig

    import random
    rnd = random.Random(5)
    for _ in range(20):
        index.ensure_model("openai/gpt-oss-120b")
        index.ensure_model("openai/gpt-oss-120b")
        index.ensure_model("openai/gpt-oss-120b")
        index.record("openai/gpt-oss-120b", "retain", "request", "ok",
                     rnd.uniform(3_000, 9_000), 12_000)
    index.ensure_model("meta/llama-3.1-8b-instruct")
    index.ensure_model("meta/llama-3.1-8b-instruct")
    index.ensure_model("meta/llama-3.1-8b-instruct")
    index.record("meta/llama-3.1-8b-instruct", "retain", "probe", "ok", 400.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    ranked = r.ranked(["meta/llama-3.1-8b-instruct", "openai/gpt-oss-120b"], "retain")
    assert ranked[0] == "openai/gpt-oss-120b", (
        "an n=1 probe on a small model must not outrank a proven larger one")


def test_exact_ties_break_deterministically_not_by_insertion_order(index):
    """Two models identical on every ranked dimension must still order stably.

    This replaces an assertion about the THIN_EVIDENCE_MARGIN, which existed so
    thin evidence and a proven model could not tie at exactly 50.0 and fall
    through to dict insertion order. The margin is gone with the score, but the
    hazard it guarded is permanent: rank_key ends in model_id precisely so an
    exact tie resolves by name instead of by whatever order the caller happened
    to build the list in. Same input, either order in, same order out.
    """
    from model_mesh.router import Router, RouterConfig

    a, b = "nvidia/tie-one-31b", "nvidia/tie-two-31b"
    for mid in (a, b):
        for _ in range(10):
            index.ensure_model(mid)
            index.record(mid, "consolidation", "request", "ok", 5_000.0, 12_000)

    r = Router(index, "https://x/v1", "k", RouterConfig())
    forward = r.ranked([a, b], "consolidation")
    reverse = r.ranked([b, a], "consolidation")
    assert forward == reverse, (
        f"tied models ordered by insertion, not deterministically: "
        f"{forward} vs {reverse}"
    )
