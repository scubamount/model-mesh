"""model-mesh: model quality prior and rank ordering.

WHY THIS EXISTS
---------------
The mesh exists to keep hindsight up on free NIM endpoints. NIM models go
overloaded or vanish unpredictably — any hour of any day — and that
unpredictability is the ONLY reason we probe anything. Everything else about a
model is stable.

The scoring that preceded this module ranked purely on p95 latency, jitter,
spike rate and success rate. All four measure *availability*, none measures
*quality*, so the router happily preferred llama-3.2-1b over gpt-oss-120b for
memory extraction whenever the 1b answered faster. For a memory backbone that
is the wrong objective: we do not need the best model in the world, but a
better model beats a worse one, and latency is not a proxy for better.

Worse, the availability score had collapsed. Measured live 2026-08-09 on
auto/retain: 23 scored models spanning p95 0.4s to 44.3s, and EVERY score
landed in [49.9, 50.0] because confidence shrinkage pinned thin evidence just
below a neutral prior and floored proven models at it. Ranking degenerated to
the alphabetical tiebreak, and gemma-4-31b-it held rank 0 at p95 44.3s, jitter
4.77, spike 0.29 — visibly degrading — over a 0.4s challenger. A score with no
resolution is not a ranking.

THE MODEL, RESTATED
-------------------
Two properties, different timescales, and conflating them into one float is
what broke:

  quality       slow-moving, effectively static. A 120b is a better extractor
                than a 1b today and will be tomorrow. Not measurable by any
                probe we run.
  availability  fast-moving. Up / overloaded / down, changing hour to hour.
                Measurable ONLY by probing, which is why probing exists.

So quality ORDERS the candidates and availability GATES them: take the best
model that is actually up right now. Deterministic, no weights to tune, no
arithmetic in which a fast weak model can out-score a slow strong one.

ON DERIVING TIER FROM THE NAME
------------------------------
This module reads parameter counts out of model ids, which sits in tension
with a rule the rest of the codebase is built on — `eligible_for_alias` and
the fidelity gate exist precisely because guessing capability from a name is
what rots.

The tension resolves on WHICH question is being asked. "Can this model do the
job at all?" is measurable, so it is measured (fidelity probe) and never
guessed. "Is this model stronger than that one?" is not measurable by any
probe in this system — a latency probe cannot tell a 550b from a 1b except by
the 1b being faster, which is the exact inversion we are fixing. Parameter
count is a coarse but honest prior for the question we cannot measure, and it
is how a human picks: prefer the 120b to the 8b.

Three properties keep it from rotting the way a static candidate list did:
it is DERIVED from the live catalog rather than enumerated, so new models are
tiered on arrival with no edit; it is a PRIOR, not a verdict — a top-tier
model that is down or failing loses to a healthy lower tier every time; and it
is OVERRIDABLE per model id in config for the cases the heuristic gets wrong.
"""

from __future__ import annotations

import re
from typing import Optional

# Parameter count in the model id: 8b, 70b, 120b, 550b, 1.5b.
#
# The leading \b is load-bearing for mixture-of-experts ids. NIM writes those as
# `nemotron-3-ultra-550b-a55b` (550b total, 55b active); without the boundary
# the pattern also matches `55b` inside `a55b`. Total parameters is the better
# quality proxy, so matching only the unprefixed number is the intent, and
# taking max() over the matches makes that explicit rather than incidental.
#
# Version numbers are not at risk: `llama-3.1-8b` yields 8, not 3.1, because a
# version is not followed by `b`.
_PARAM_RE = re.compile(r"\b(\d+(?:\.\d+)?)b\b")

# Size adjectives, for the ~30% of NIM ids that carry no parameter count at all
# (minimax-m3, glm-5.2, mistral-nemotron, inkling, step-3.7-flash). Vendors are
# consistent about these words even when they omit the number.
_SIZE_WORDS = {
    "ultra": 5, "xxl": 5,
    "super": 4, "large": 4, "xl": 4,
    "medium": 3, "mid": 3,
    "mini": 2, "small": 2, "lite": 2, "flash": 2, "xs": 2,
    "nano": 1, "tiny": 1, "micro": 1,
}

# Unknown size sorts mid, NOT low. An unrecognized id is far more often a new
# frontier model whose vendor skipped the parameter count (glm-5.2, minimax-m3)
# than it is a tiny one, and tiering those at the bottom would reproduce the
# bug this module fixes — quietly excluding the best models for a naming
# convention.
DEFAULT_TIER = 3
MAX_TIER = 5

# Parameter thresholds, in billions. Calibrated against NIM's live catalog
# rather than round numbers — the published sizes cluster into natural bands:
#
#   550, 340, 120  frontier          -> 5
#   90, 70, 49     large             -> 4
#   31, 30, 26, 20 mid               -> 3
#   12, 11, 9, 8   small             -> 2
#   4, 3, 1        tiny              -> 1
#
# Boundaries sit inside the gaps between those clusters, so a new release lands
# in the intended band without a code change and no cluster is split.
_PARAM_TIERS = ((100.0, 5), (45.0, 4), (18.0, 3), (7.0, 2))

# Availability buckets. Ordering is the whole design: quality never promotes a
# model past a healthier one, so the worst case of a bad tier guess is picking
# the wrong healthy model, never picking a dead one.
BUCKET_HEALTHY = 0     # recent success, responding promptly
BUCKET_OVERLOADED = 1  # recent success, but slow enough to signal queueing
BUCKET_UNKNOWN = 2     # no evidence in the scoring window
BUCKET_FAILING = 3     # recent evidence, no successes in it


def tier(model_id: str, overrides: Optional[dict] = None) -> int:
    """Quality prior for a model id, 1 (weakest) to 5 (strongest).

    Explicit config override wins, then parameter count, then size adjective,
    then DEFAULT_TIER. Parameter count outranks the adjective on purpose:
    `nemotron-3-nano-30b-a3b` says both "nano" and "30b", and 30b is the fact
    while "nano" is marketing relative to that vendor's own lineup.
    """
    if overrides:
        for key, val in overrides.items():
            if key == model_id:
                return max(1, min(MAX_TIER, int(val)))

    mid = model_id.lower()

    matches = _PARAM_RE.findall(mid)
    if matches:
        params = max(float(m) for m in matches)
        for threshold, t in _PARAM_TIERS:
            if params >= threshold:
                return t
        return 1

    for word, t in _SIZE_WORDS.items():
        if re.search(rf"\b{word}\b", mid):
            return t

    return DEFAULT_TIER


def availability_bucket(score, overload_p95_ms: float) -> int:
    """Classify measured evidence into an availability bucket.

    `score` is an index.Score or None. Latency lands here rather than in the
    ranking value on purpose: on a free shared endpoint a slow response is not
    a property of the model, it is the symptom of other people using it, and
    the model that is answering in 44s is the one about to start timing out.
    Treating latency as an overload SIGNAL rather than a quality measure is
    what stops a temporarily-busy strong model from being permanently demoted
    below a weak one.
    """
    if score is None:
        return BUCKET_UNKNOWN
    if score.success_rate <= 0.0:
        return BUCKET_FAILING
    if score.p95_ms >= overload_p95_ms:
        return BUCKET_OVERLOADED
    return BUCKET_HEALTHY


def rank_key(
    model_id: str,
    score,
    overload_p95_ms: float,
    overrides: Optional[dict] = None,
) -> tuple:
    """Total order over candidates: availability first, then quality, then speed.

    Returns a plain tuple so ordering is a pure function of measured evidence
    and the model id — fully deterministic, and reproducible from
    /mesh/status without re-running anything. The trailing model_id breaks
    exact ties by name instead of leaving order to dict insertion, which is how
    the previous collapse-to-50.0 silently became an alphabetical ranking.
    """
    bucket = availability_bucket(score, overload_p95_ms)
    p95 = score.p95_ms if score is not None else 0.0
    return (bucket, -tier(model_id, overrides), p95, model_id)
