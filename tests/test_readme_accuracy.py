"""README claims must match live code. Docs drift silently; gates do not.

Every number in the README's config table is a promise about behavior. When a
default moves and the table does not, the README becomes confidently wrong —
which is worse than absent, because a reader has no reason to doubt it. This has
already happened twice here: `max_attempts` was documented as 3 long after it
became 8, and the test count was stale by 45.

These tests parse the README's own tables and compare them against the values
the daemon actually loads, so a default change fails the suite until the doc is
updated in the same commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from model_mesh.config import DEFAULTS

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()


def _documented(key: str) -> str:
    """Pull the `default` cell for a `| \\`key\\` | \\`value\\` | ... |` table row."""
    m = re.search(
        r"^\|\s*`" + re.escape(key) + r"`\s*\|\s*`?([^|`]+)`?\s*\|",
        README,
        re.MULTILINE,
    )
    assert m, f"README documents no default for {key!r}"
    return m.group(1).strip()


# (config section, key, parser) — every row the README's config table claims.
NUMERIC_CLAIMS = [
    ("router", "max_attempts", int),
    ("router", "reprobe_top_n", int),
    ("router", "sweep_max_models", int),
    ("router", "total_budget_s", float),
    ("router", "request_timeout_s", float),
    ("router", "probe_timeout_s", float),
    ("router", "overload_p95_ms", float),
    ("router", "max_p95_ms_for_eligibility", float),
    ("router", "min_success_rate", float),
    ("router", "fidelity_fails_for_floor", int),
    ("router", "breaker_threshold", int),
    ("router", "breaker_cooldown_s", float),
    ("router", "provider_pause_default_s", float),
    ("router", "provider_pause_max_s", float),
    ("discovery", "probe_top_n", int),
    ("discovery", "max_probes_per_pass", int),
]


@pytest.mark.parametrize(
    "section,key,cast", NUMERIC_CLAIMS, ids=[f"{s}.{k}" for s, k, _ in NUMERIC_CLAIMS]
)
def test_readme_default_matches_code(section, key, cast):
    documented = cast(_documented(f"{section}.{key}"))
    live = cast(DEFAULTS[section][key])
    assert documented == live, (
        f"README says {section}.{key} = {documented}, code ships {live}. "
        f"Update the README in the same commit as the default."
    )


def test_readme_test_count_is_current():
    """The README states a test count; a stale one is a false claim about
    coverage. Counted by running collection, not by trusting the number."""
    m = re.search(r"\((\d+) tests;", README)
    assert m, "README no longer states a test count"
    documented = int(m.group(1))

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout
    collected = re.search(r"(\d+) tests collected", out)
    assert collected, f"could not parse collection output: {out[-300:]}"
    actual = int(collected.group(1))

    assert documented == actual, (
        f"README claims {documented} tests, suite collects {actual}"
    )


def test_readme_documents_no_absent_config_file():
    """The README used to assert config.yaml 'is currently absent on this
    machine' — a statement about one deployment baked into shipped docs, which
    the installer then falsified by writing the file."""
    assert "absent on this machine" not in README, (
        "README states a fact about one particular machine"
    )


def test_readme_carries_no_operator_identity():
    """Shipped docs must not name a specific person, host or private repo."""
    for leak in ("scubamount", "Scubamount", "Andrew", "/Users/", "hindsight"):
        assert leak not in README, f"README leaks operator-specific reference: {leak!r}"


def test_budget_invariant_the_readme_asserts():
    """The README tells operators to keep 2 x request_timeout under the budget.
    Assert the shipped defaults actually satisfy their own advice."""
    r = DEFAULTS["router"]
    assert 2 * r["request_timeout_s"] < r["total_budget_s"], (
        "shipped defaults violate the README's own budget rule"
    )
