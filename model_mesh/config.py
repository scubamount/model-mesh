"""model-mesh config loader.

YAML at ~/.model-mesh/config.yaml; sane NIM defaults ship in-code so the
daemon boots with zero config. Secrets come from env vars named in config —
never stored in the file or the DB.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CONFIG_PATH = Path(os.path.expanduser("~/.model-mesh/config.yaml"))

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
    "db_path": "~/.model-mesh/mesh.db",
    "listen": {"host": "127.0.0.1", "port": 8002},
    "provider": {
        "name": "nim",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "router": {
        "breaker_threshold": 3,
        "breaker_cooldown_s": 120.0,
        "breaker_cooldown_max_s": 1800.0,
        # Attempt count must not be the binding constraint — the real-time
        # budget should be. A 4xx reject costs 0.26s median, so three cheap
        # rejects used to end a cascade with ~99% of the budget unspent.
        "max_attempts": 8,
        "reprobe_top_n": 4,
        # A timeout burns this whole value, so it is the price of one failure.
        # 90s aborts ~1% of measured successes (p95 51.6s, p99 88.4s) and those
        # cascade onward rather than being lost; 120s let only 2 failures fit.
        "request_timeout_s": 90.0,
        "probe_timeout_s": 45.0,
        "min_success_rate": 0.5,
        "min_samples_for_floor": 4,
        # Must satisfy 2 * this < total_budget_s so a slow model can't consume
        # the cascade; audit-timeout-chain.py asserts the relationship.
        "max_p95_ms_for_eligibility": 75000.0,
        # Under hindsight's 300s RETAIN_LLM_TIMEOUT so the client never abandons
        # mid-cascade (see 060-hindsight-setup.sh). 280 fits 3 full-price 90s
        # timeouts (270 < 280) and keeps 20s headroom to the client.
        "total_budget_s": 280.0,
    },
    # Cap on distinct models probed per discovery pass. Backfilling 67 unproven
    # models at ~14s each is ~15 min once; this bounds the tail so a catalog
    # explosion can't turn the daily job into an unbounded burn. Steady state is
    # ~0 probes: a model is probed once per op_class, then real traffic keeps it
    # scored for free.
    #
    # probe_top_n bounds it by RELEVANCE rather than by count: only the best few
    # candidates per alias are probed at all. The cascade tries at most 3 models,
    # so knowing the health of the 20th-best is worth nothing and costs a real
    # request against a shared free endpoint. 6 leaves double the cascade depth
    # in reserve, so several top models can be overloaded at once and a healthy,
    # freshly-measured alternative is still ready.
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
            "max_candidates": 8,
        },
        "auto/consolidation": {
            "op_class": "consolidation",
            "include": [],
            "exclude": _NON_TEXT + ["reasoning"],
            "max_candidates": 8,
        },
        "auto/reflect": {
            "op_class": "retain",
            "include": [],
            "exclude": _NON_TEXT + ["reasoning"],
            "max_candidates": 8,
        },
        # Long-form prose/instruction-following lane (DSPy skill evolution).
        # Its OWN op_class on purpose: samples in `retain` drive hindsight's
        # model choice, so evolution traffic must not vote in that ranking.
        "auto/evolve": {
            "op_class": "evolve",
            "include": [],
            "exclude": _NON_TEXT,
            "max_candidates": 8,
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


# Fallback file for the provider key. `launchctl setenv` does NOT survive a
# machine restart, so the launchd daemon can come up with an empty key and 401
# every call. Reading this file as a fallback (at call time, not import time)
# makes that self-healing instead of an operator page.
KEY_FALLBACK_FILE = Path(os.path.expanduser("~/.hermes/.env"))


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
