"""Probe economy: probe what we would route to, stop probing the dead.

Probing is not free — every probe is a real request against a shared free NIM
endpoint, and the budget it burns is the budget a newly-released model needs.
Two rules follow from what probing is FOR (detecting overload and disappearance,
the only two things that change unpredictably):

  - a model the cascade would never dial has health worth knowing nothing about
  - a model with zero successes across its attempts inside the dormancy window
    is skipped, whatever the catalog still claims

The skip is keyed to the WINDOW, never to "has ever been dormant": gating on a
dormant-flag made a once-dormant model unprobed FOREVER (audit 2026-08-24) —
the flag only clears via a success, only a probe produces one, and the probe
was the thing being skipped. Keyed to the window, the documented weekly
recheck emerges for free and a single success clears dormancy outright.

These were `check()`-decorated functions with their own main() that NOTHING
invoked — pytest collected zero of them, so every guarantee below was
decoration while the suite stayed green (audit 2026-08-24). They are plain
pytest tests now.
"""

from __future__ import annotations

import time

import pytest

from model_mesh.discovery import DORMANT_AFTER_S, discover
from model_mesh.index import SCORE_WINDOW_S, Index, OK


class FakeRouter:
    """Records what was probed. A passing verdict writes the sample a real
    router would — otherwise nothing ever marks a recheck as consumed."""

    def __init__(self, index: Index, verdict=("pass", "")):
        self.index = index
        self.probed = []
        self._verdict = verdict

    def probe_verdict(self, mid, oc, messages, timeout=None):
        self.probed.append((mid, oc))
        if self._verdict[0] == "pass":
            self.index.record(mid, oc, "probe", OK, 900.0)
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


@pytest.fixture()
def index(tmp_path):
    idx = Index(tmp_path / "pe.db")
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


def _seed_sample(idx, mid, status, age_s):
    """Insert one aged sample the way the daemon would have written it."""
    with idx._lock:
        idx._conn.execute(
            "INSERT INTO samples (model_id, ts, op_class, source, latency_ms,"
            " status, payload_chars) VALUES (?,?,?,?,?,?,?)",
            (mid, time.time() - age_s, "retain", "probe", None, status, None),
        )
        idx._conn.commit()


def test_probe_top_n_probes_only_the_best_few(index):
    router = FakeRouter(index)
    _run(index, router, probe_top_n=3)
    assert len(router.probed) == 3, router.probed


def test_the_few_probed_are_the_strongest_not_the_alphabetical_first(index):
    # Alphabetical ordering is what starved openai/* of probes for weeks.
    router = FakeRouter(index)
    _run(index, router, probe_top_n=3)
    probed = {m for m, _ in router.probed}
    assert "meta/llama-3.2-1b-instruct" not in probed, probed
    assert "nvidia/nemotron-3-ultra-550b-a55b" in probed, probed


def test_probe_top_n_none_keeps_probing_everything(index):
    router = FakeRouter(index)
    _run(index, router, probe_top_n=None)
    assert len(router.probed) == len(CATALOG), len(router.probed)


def test_zero_successes_inside_the_window_is_skipped(index):
    # Failed two days ago, never succeeded: the catalog listing is not
    # evidence the model serves. Probing it again today is exactly the burn
    # the dormancy gate exists to stop.
    mid = "openai/gpt-oss-120b"
    _seed_sample(index, mid, "http-404", 2 * 86400)
    router = FakeRouter(index)
    _run(index, router, probe_top_n=None)
    assert mid not in {m for m, _ in router.probed}, router.probed


def test_once_dormant_lane_is_rechecked_after_the_window_exactly_once(index):
    """The audit 2026-08-24 ratchet. A model whose last attempt aged out of
    the window MUST earn one recheck probe; that probe writes a sample, which
    suppresses the next window. Under the dormant-flag gate this model was
    never probed again — silently contradicting DORMANT_AFTER_S's contract."""
    mid = "openai/gpt-oss-120b"
    _seed_sample(index, mid, "http-404", DORMANT_AFTER_S + 3600)

    router = FakeRouter(index)
    _run(index, router, probe_top_n=None)
    assert mid in {m for m, _ in router.probed}, (
        f"once-dormant model was never re-probed — one-way door: {router.probed}"
    )

    # The recheck wrote a sample; the next pass must not re-probe it again.
    router2 = FakeRouter(index)
    _run(index, router2, probe_top_n=None)
    assert mid not in {m for m, _ in router2.probed}, router2.probed


def test_one_success_inside_the_window_rebutts_dormancy(index):
    # Failure five days ago, success four days ago: the model came back, so
    # the failures do not make it dormant. Stale enough to NEED a probe
    # (outside the scoring window) yet inside the dormancy window WITH a
    # success — precisely the state that must stay probed.
    mid = "openai/gpt-oss-120b"
    _seed_sample(index, mid, "http-500", 5 * 86400)
    _seed_sample(index, mid, OK, 4 * 86400)
    router = FakeRouter(index)
    _run(index, router, probe_top_n=None)
    assert mid in {m for m, _ in router.probed}, router.probed


def test_a_never_sampled_model_is_unknown_not_dormant(index):
    router = FakeRouter(index)
    _run(index, router, probe_top_n=None)
    assert len(router.probed) == 9, router.probed


def test_scoring_window_and_dormancy_window_leave_a_probeable_gap():
    """The staleness gate skips anything fresher than SCORE_WINDOW_S; the
    dormancy gate skips anything tried within DORMANT_AFTER_S without success.
    If the scoring window reached past the dormancy window, a failed model
    would be 'fresh enough' forever and never rechecked."""
    assert SCORE_WINDOW_S < DORMANT_AFTER_S
