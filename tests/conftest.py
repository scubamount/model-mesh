"""Test-session isolation from the operator's live state.

`model_mesh.config` resolves STATE_DIR from $MESH_HOME at import, and
`model_mesh.app` opens `Index(CFG["db_path"])` at import. Without this file,
any test that imports the app opened the LIVE ~/.model-mesh/mesh.db (the Index
constructor runs schema migrations on it), read the live config.yaml, and --
once requests began appending to audit/exclusions.jsonl -- wrote fake
traffic into the operator's audit trail. Observed 2026-10-05: one test row
(`"pool": 1`) landed in the live exclusions log.

This module is imported by pytest before any test module, so setting
MESH_HOME here takes effect before model_mesh.config ever runs.
"""
import atexit
import os
import shutil
import tempfile

_TEST_HOME = tempfile.mkdtemp(prefix="model-mesh-test-home-")
os.environ["MESH_HOME"] = _TEST_HOME
atexit.register(shutil.rmtree, _TEST_HOME, True)
