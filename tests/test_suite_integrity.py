"""Meta-test: every test module must actually be collected by pytest.

Three separate files in this repo were written as `@check`-decorated functions
appended to a module-level list and executed by a `main()` that nothing ever
invoked — `test_probe_economy.py` (audit 2026-08-24), then
`test_stale_evidence_reprobe.py` and `test_quality_ranking.py` (audit
2026-08-25). Each collected ZERO tests. The suite reported green for weeks
while those guarantees were decoration, and one of them had silently rotted
against a field that a fix had deleted months earlier.

Three occurrences is a missing mechanism, not three mistakes. This test is that
mechanism: a `test_*.py` file that defines no collectable test, or that carries
the `main()`-runner shape, fails the suite here rather than passing silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# Helper modules: named `test_*` only because they live beside the suite and
# share its import path. They are imported BY tests and define none themselves.
# Anything added here must be a pure fixture/helper module.
HELPER_MODULES = {"test_stale_evidence_reprobe_fixtures.py"}


def _module_files() -> list[Path]:
    return sorted(
        p for p in TESTS_DIR.glob("test_*.py")
        if p.name != Path(__file__).name and p.name not in HELPER_MODULES
    )


def _top_level_defs(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_defines_collectable_tests(path: Path) -> None:
    """Every non-helper test module must define at least one `test_*` function
    or a `Test*` class. Zero means pytest collects nothing from it."""
    tree = ast.parse(path.read_text())
    funcs = [n.name for n in _top_level_defs(tree) if n.name.startswith("test_")]
    classes = [
        n.name for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name.startswith("Test")
    ]
    assert funcs or classes, (
        f"{path.name} defines no pytest-collectable tests. If it is a helper "
        f"module, add it to HELPER_MODULES here; otherwise its assertions are "
        f"invisible to the suite and enforce nothing."
    )


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_has_no_self_runner(path: Path) -> None:
    """Reject the exact shape that hid the three defects: a module that runs its
    own assertions from `main()`/`run()` under `if __name__ == "__main__"`.

    Such a module passes when executed by hand and enforces nothing under
    pytest, which is the failure mode that made this file necessary."""
    tree = ast.parse(path.read_text())
    runners = {n.name for n in _top_level_defs(tree)} & {"main", "run"}
    has_dunder_main = any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "__name__"
        for n in tree.body
    )
    assert not (runners and has_dunder_main), (
        f"{path.name} carries a self-runner ({', '.join(sorted(runners))}() plus "
        f"an `if __name__ == '__main__'` block). Assertions reached only that "
        f"way are invisible to pytest — convert them to `test_*` functions."
    )
