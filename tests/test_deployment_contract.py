"""Config keys must be load-bearing, and state must move as one unit.

`listen` shipped in DEFAULTS while the launchd plist passed --host/--port to
uvicorn on the command line. Config and reality were two sources of truth:
editing config.yaml moved nothing, and the key read as supported (audit
2026-08-25). A knob nobody reads is worse than no knob — it documents a
capability that does not exist.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from model_mesh.config import DEFAULTS

REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "scripts" / "install-launchd.sh"


def test_listen_is_read_by_the_entrypoint():
    """The serving address must come from config, not from plist arguments."""
    src = (REPO / "model_mesh" / "app.py").read_text()
    tree = ast.parse(src)
    main = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    assert main is not None, (
        "app.main() is gone. It is the entrypoint the launchd plist calls; "
        "without it the plist must hardcode host/port again and `listen` dies."
    )
    body = ast.unparse(main)
    assert "listen" in body, "main() no longer reads CFG['listen']"
    assert "uvicorn.run" in body, "main() no longer starts the server"


def test_installer_does_not_hardcode_the_listen_address():
    """The plist must not pass --host/--port: that is what orphaned `listen`."""
    text = INSTALLER.read_text()
    for flag in ("--host", "--port"):
        assert flag not in text, (
            f"installer passes {flag} to the server again — the listen address "
            f"is config's job, or config.yaml silently stops working"
        )


def test_installer_adopts_an_existing_install():
    """Reinstall must reuse the machine's existing label prefix and state dir.

    Forcing the default prefix would load a SECOND daemon against the same port
    while the first kept running, and strand the old mesh.db — learned state
    (sample history, breaker states, EOL marks) that nothing regenerates.
    """
    text = INSTALLER.read_text()
    assert "ADOPTED_PREFIX" in text and "ADOPTED_HOME" in text, (
        "installer lost its adoption arm; a reinstall would fork the install"
    )
    assert "*.model-mesh.plist" in text, (
        "adoption must DISCOVER the existing agent from the filesystem rather "
        "than matching a hardcoded label list"
    )


@pytest.mark.parametrize("key", ["db_path"])
def test_state_paths_live_under_the_state_dir(key):
    """db, config and key-fallback must derive from one root, so MESH_HOME
    cannot relocate half an install."""
    from model_mesh.config import STATE_DIR

    assert str(DEFAULTS[key]).startswith(str(STATE_DIR)), (
        f"{key} escaped the state dir; MESH_HOME would move config but leave "
        f"{key} behind"
    )


def test_mesh_home_relocates_every_state_path():
    """Behavioral check, in a child process: the module reads MESH_HOME at
    import time, so this cannot be proven by re-importing in-process."""
    code = (
        "from model_mesh.config import STATE_DIR, CONFIG_PATH, "
        "KEY_FALLBACK_FILE, DEFAULTS;"
        "print(STATE_DIR);print(CONFIG_PATH);"
        "print(DEFAULTS['db_path']);print(KEY_FALLBACK_FILE)"
    )
    env = dict(os.environ, MESH_HOME="/tmp/mesh-relocation-probe")
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=env, cwd=REPO,
    ).stdout.split()
    assert len(out) == 4
    for path in out:
        assert path.startswith("/tmp/mesh-relocation-probe"), (
            f"{path} ignored MESH_HOME — state would split across two dirs"
        )
