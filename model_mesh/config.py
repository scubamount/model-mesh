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
        "max_attempts": 3,
        "reprobe_top_n": 3,
        "request_timeout_s": 120.0,
        "probe_timeout_s": 45.0,
        "min_success_rate": 0.5,
        "min_samples_for_floor": 4,
        # Must satisfy 2 * this < total_budget_s so a slow model can't consume
        # the cascade; audit-timeout-chain.py asserts the relationship.
        "max_p95_ms_for_eligibility": 75000.0,
        # Under hindsight's 300s RETAIN_LLM_TIMEOUT so the client never abandons
        # mid-cascade (see 060-hindsight-setup.sh).
        "total_budget_s": 240.0,
    },
    "aliases": {
        "auto/retain": {
            "op_class": "retain",
            "include": ["gpt-oss", "llama-3.3", "nemotron-super", "qwen3"],
            "exclude": ["vl-", "-vision", "safety-guard", "embed", "rerank",
                        "nano-vl", "reasoning"],
            "max_candidates": 8,
        },
        "auto/consolidation": {
            "op_class": "consolidation",
            "include": ["gpt-oss", "llama-3.3", "nemotron-super", "qwen3"],
            "exclude": ["vl-", "-vision", "safety-guard", "embed", "rerank",
                        "nano-vl", "reasoning"],
            "max_candidates": 8,
        },
        "auto/reflect": {
            "op_class": "retain",
            "include": ["gpt-oss", "llama-3.3", "nemotron-super", "qwen3"],
            "exclude": ["vl-", "-vision", "safety-guard", "embed", "rerank",
                        "nano-vl", "reasoning"],
            "max_candidates": 8,
        },
        # Long-form prose/instruction-following lane (DSPy skill evolution).
        # Its OWN op_class on purpose: samples in `retain` drive hindsight's
        # model choice, so evolution traffic must not vote in that ranking.
        "auto/evolve": {
            "op_class": "evolve",
            "include": ["gpt-oss", "llama-3.3", "nemotron-super", "qwen3",
                        "deepseek-v4"],
            "exclude": ["vl-", "-vision", "safety-guard", "embed", "rerank",
                        "nano-vl"],
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
