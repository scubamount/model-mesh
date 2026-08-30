"""model-mesh config loader.

YAML at `<state dir>/config.yaml`; sane NIM defaults ship in-code so the daemon
boots with zero config. Secrets come from env vars named in config — never
stored in the file or the DB.

The state directory is `$MESH_HOME` (default `~/.model-mesh`) and holds
everything an operator must back up: mesh.db, config.yaml, the key fallback
`.env`, logs, and the audit trail. One env var relocates the whole install, so
a second copy on the same machine cannot silently share the first one's state.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# The state directory: one env var relocates db, config, key fallback, logs and
# audit trail together. Everything else derives from this, so there is no way to
# move half an install.
STATE_DIR = Path(os.environ.get("MESH_HOME", "~/.model-mesh")).expanduser()
CONFIG_PATH = STATE_DIR / "config.yaml"

# Substrings identifying models that CANNOT serve a JSON-emitting text op_class:
# other modalities (vision/audio/video/image-gen), non-generative heads
# (embedding/rerank/reward/parse/OCR), and domain heads with no general
# instruction-following. Derived by auditing NIM's live 102-model catalog, not
# guessed. This is the ONLY name-based filter that remains, and it is a
# capability statement — everything textual is admitted and judged on measured
# fidelity + latency instead of on its family name.
_NON_TEXT = [
    # vision / multimodal
    "vl-", "-vl", "-vision", "vila", "fuyu", "kosmos", "neva", "deplot",
    "nvclip", "clip", "florence", "paddle", "ocr", "-parse", "depth",
    "dust3r", "segment", "optical", "molmo", "cosmos",
    # audio / speech
    "asr", "tts", "speech", "riva", "parakeet", "canary", "audio", "-stt",
    # image / video generation
    "diffusion", "stable-", "sana", "flux", "-video", "consistory", "edify",
    # non-generative heads
    "embed", "rerank", "retriever", "genrm", "reward", "-bge", "bge-",
    # safety / guard rails
    "safety", "guard", "shield", "topic-control", "jailbreak",
    # domain-specific, not general instruction-followers. Anchored with the
    # vendor prefix: a bare "-med" also matched mistral-medium-3.5-128b, a
    # general text model. Substring filters need anchoring, which is exactly
    # why this list is small and everything else is decided by probing.
    "bio", "protein", "esm", "dna", "chem", "molmim", "diffdock", "alphafold",
    "palmyra-med", "palmyra-fin", "weather", "earth",
]

DEFAULTS: dict = {
    "db_path": str(STATE_DIR / "mesh.db"),
    "listen": {"host": "127.0.0.1", "port": 8002},
    "provider": {
        "name": "nim",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "router": {
        "breaker_threshold": 3,
        # 30s -> x2 -> 300s cap, per free-coding-models' field-proven breaker
        # against these same NIM endpoints (v0.5.81 config.js:205-210).
        # Overload flips within minutes; the old 120s->1800s ladder kept a
        # recovered model benched up to 30min after a 2-minute episode.
        "breaker_cooldown_s": 30.0,
        "breaker_cooldown_max_s": 300.0,
        # Provider-wide 429 pause (shared-key throttle; see RouterConfig).
        "provider_pause_default_s": 5.0,
        "provider_pause_max_s": 60.0,
        # Attempt count must not be the binding constraint — the real-time
        # budget should be. A 4xx reject costs 0.26s median, so three cheap
        # rejects used to end a cascade with ~99% of the budget unspent.
        "max_attempts": 8,
        "reprobe_top_n": 4,
        # Last-resort sweep (2026-08-24): after max_attempts dials AND the
        # re-probe arm both miss, walk the REST of the ranked pool — deduped
        # against every model this request already dialed, 'gone' excluded,
        # budget-bound like every arm. Makes "no healthy candidates" require
        # a whole-pool failure inside one request. Must match RouterConfig;
        # test_config_defaults_match_dataclass asserts sync.
        "sweep_on_total_miss": True,
        "sweep_max_models": 12,
        # A timeout burns this whole value, so it is the price of one failure.
        # 90s aborts ~1% of measured successes (p95 51.6s, p99 88.4s) and those
        # cascade onward rather than being lost; 120s let only 2 failures fit.
        "request_timeout_s": 90.0,
        "probe_timeout_s": 45.0,
        # PER-OP_CLASS timeout overrides. Both numbers above were tuned on
        # RETAIN latency and then applied to every op_class, which silently
        # destroyed auto/consolidation.
        #
        # Measured 2026-08-24 on this box. Consolidation's own success p99 is
        # 71.6s (max 94.3s), so a 45s PROBE and a 90s REQUEST cut into the real
        # distribution: 1,183 of 2,203 consolidation timeouts in 24h landed in
        # the 43-47s and 88-92s bands — i.e. AT the ceilings, not past them.
        # Each one was recorded as a failure sample, and because ranking floors
        # on measured success_rate, the whole pool then fell below
        # min_success_rate: 25 of 26 candidates blocked by succ-floor, the 26th
        # by the fidelity floor, so ranked() returned [] and /health reported
        # "auto/consolidation: no healthy candidates". The mesh had timed out
        # every candidate it had, then correctly concluded none of them worked.
        #
        # Proof it was the ceiling and not the models: probed live at 100s,
        # gpt-oss-120b answered in 90.8s, gpt-oss-20b in 1.0s and
        # nemotron-3-super-120b in 0.9s, and ALL THREE passed check_fidelity —
        # while the index held them at success_rate 0.000, 0.091 and (fidelity-
        # floored) respectively. gpt-oss-120b has 632 lifetime consolidation
        # successes; it fails at 90s by less than a second.
        #
        # Keyed by op_class so a slow op_class cannot be fixed by inflating the
        # budget for a fast one. Anything absent falls back to the values above.
        # 2 * request stays under total_budget_s (2*135 = 270 < 280).
        "request_timeout_s_by_op_class": {"consolidation": 135.0},
        "probe_timeout_s_by_op_class": {"consolidation": 100.0},
        "min_success_rate": 0.5,
        "min_samples_for_floor": 4,
        "min_failures_for_thin_floor": 2,
        # Consecutive fidelity violations (200 with a body violating the
        # op_class JSON contract) that drop a model from an op_class. Must
        # match RouterConfig.fidelity_fails_for_floor; the defaults-match test
        # asserts the two files agree.
        "fidelity_fails_for_floor": 2,
        # Must satisfy 2 * this < total_budget_s so a slow model can't consume
        # the cascade; test_deployment_contract.py asserts the relationship.
        "max_p95_ms_for_eligibility": 75000.0,
        # Keep this UNDER the calling client's own timeout so the client never
        # abandons the request mid-cascade. 280 fits 3 full-price 90s attempts
        # (270 < 280) and leaves 20s headroom to a 300s client deadline.
        "total_budget_s": 280.0,
        # p95 at/above this = overloaded, demoted below healthy regardless of
        # quality tier; free re-promotion when measured p95 recovers.
        "overload_p95_ms": 20000.0,
        # Auth failures must expire (launchctl setenv does not survive reboot;
        # a daemon start with a missing key must not brick the index).
        "auth_cooldown_s": 300.0,
        # Per-model quality-tier overrides {model_id: 1..5}. Empty by design:
        # the heuristic derives from the live catalog, static pins rot.
        "tier_overrides": {},
    },
    # Cap on distinct models probed per discovery pass. Backfilling 67 unproven
    # models at ~14s each is ~15 min once; this bounds the tail so a catalog
    # explosion can't turn the daily job into an unbounded burn. Steady state is
    # ~0 probes: a model is probed once per op_class, then real traffic keeps it
    # scored for free.
    #
    # probe_top_n bounds it by RELEVANCE rather than by count: only the best few
    # candidates per alias are probed at all. The main cascade dials at most
    # max_attempts (8) and the sweep arm covers the rest, so knowing the health
    # of the 20th-best is worth little and costs a real request against a
    # shared free endpoint. 6 keeps freshly-measured alternatives ahead of the
    # ranking while conserving quota.
    "discovery": {"max_probes_per_pass": 25, "probe_top_n": 6},
    # Non-text modalities and non-generative heads. EXCLUSION-first on purpose:
    # an include-whitelist of families matched 5 of 102 live NIM models and
    # silently shrank as NIM's catalog grew (see discovery.eligible_for_alias).
    # Anything not excluded here is probed, and the fidelity gate admits it on
    # EVIDENCE rather than on its name.
    "aliases": {
        "auto/retain": {
            "op_class": "retain",
            "include": [],
            "exclude": _NON_TEXT + ["reasoning"],
        },
        "auto/consolidation": {
            "op_class": "consolidation",
            "include": [],
            "exclude": _NON_TEXT + ["reasoning"],
        },
        "auto/reflect": {
            "op_class": "retain",
            "include": [],
            "exclude": _NON_TEXT + ["reasoning"],
        },
        # Long-form prose/instruction-following lane (DSPy skill evolution).
        # Its OWN op_class on purpose: samples in `retain` drive hindsight's
        # model choice, so evolution traffic must not vote in that ranking.
        "auto/evolve": {
            "op_class": "evolve",
            "include": [],
            "exclude": _NON_TEXT,
        },
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = dict(DEFAULTS)
    if path.is_file() and yaml is not None:
        with path.open() as f:
            user = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user)
    return cfg


# Fallback file for the provider key. A launchd daemon does not inherit your
# interactive shell env, and `launchctl setenv` does not survive a machine
# restart, so the daemon can come up with an empty key and 401 every call.
# Reading a file as a fallback (at call time, not import time) makes that
# self-healing instead of an operator page. Lives in the state dir beside the
# db; override the file itself with MODEL_MESH_KEY_FALLBACK_FILE. Holds lines of
# `ENV_VAR=value`, mode 0600.
KEY_FALLBACK_FILE = Path(
    os.environ.get("MODEL_MESH_KEY_FALLBACK_FILE", str(STATE_DIR / ".env"))
).expanduser()


def resolve_api_key(env_var: str, fallback: Path = KEY_FALLBACK_FILE) -> str:
    """Env first, then `VAR=value` in the fallback file. Never logged."""
    val = os.environ.get(env_var, "")
    if val:
        return val
    try:
        for line in fallback.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""
