#!/bin/sh
# Install model-mesh as a launchd user agent + daily discovery job (macOS).
# Idempotent: re-running rewrites the plists and reloads them.
#
# Everything below is overridable from the environment, because none of it is
# a property of model-mesh itself:
#
#   MESH_LABEL_PREFIX   reverse-DNS prefix for the launchd labels (default: local)
#   MESH_PORT           port to listen on          (default: 8002)
#   MESH_HOST           address to bind            (default: 127.0.0.1)
#   MESH_HOME           state dir: db, logs, audit (default: ~/.model-mesh)
#   MESH_PYTHON         interpreter used to create the venv (default: first
#                       python3.12+ found on PATH)
#   MESH_DISCOVER_HOUR  daily discovery hour, 0-23 (default: 6)
#   MESH_DISCOVER_MIN   daily discovery minute     (default: 15)
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"

MESH_LABEL_PREFIX="${MESH_LABEL_PREFIX:-local}"
MESH_PORT="${MESH_PORT:-8002}"
MESH_HOST="${MESH_HOST:-127.0.0.1}"
MESH_HOME="${MESH_HOME:-$HOME/.model-mesh}"
MESH_DISCOVER_HOUR="${MESH_DISCOVER_HOUR:-6}"
MESH_DISCOVER_MIN="${MESH_DISCOVER_MIN:-15}"

LABEL_DAEMON="$MESH_LABEL_PREFIX.model-mesh"
LABEL_DISCOVER="$MESH_LABEL_PREFIX.model-mesh-discover"
PY="$REPO_DIR/.venv/bin/python"

mkdir -p "$MESH_HOME" "$PLIST_DIR"

# Find an interpreter rather than hardcoding a Homebrew path: the same repo has
# to install on an Intel Mac (/usr/local), an Apple Silicon Mac (/opt/homebrew),
# and a pyenv/uv setup where neither exists.
if [ ! -x "$PY" ]; then
  BOOTSTRAP_PY="${MESH_PYTHON:-}"
  if [ -z "$BOOTSTRAP_PY" ]; then
    for c in python3.13 python3.12 python3; do
      if command -v "$c" >/dev/null 2>&1; then BOOTSTRAP_PY="$(command -v "$c")"; break; fi
    done
  fi
  [ -n "$BOOTSTRAP_PY" ] || { echo "no python3.12+ found; set MESH_PYTHON" >&2; exit 1; }
  "$BOOTSTRAP_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' || {
    echo "$BOOTSTRAP_PY is older than 3.12; set MESH_PYTHON" >&2; exit 1; }
  echo "venv missing — creating with $BOOTSTRAP_PY"
  env -u PYTHONPATH "$BOOTSTRAP_PY" -m venv "$REPO_DIR/.venv"
  "$REPO_DIR/.venv/bin/pip" -q install -e "$REPO_DIR"
fi

# --- daemon plist -----------------------------------------------------------
cat > "$PLIST_DIR/$LABEL_DAEMON.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL_DAEMON</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string>
    <string>-m</string><string>uvicorn</string>
    <string>model_mesh.app:app</string>
    <string>--host</string><string>$MESH_HOST</string>
    <string>--port</string><string>$MESH_PORT</string>
    <string>--no-access-log</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$MESH_HOME/mesh.log</string>
  <key>StandardErrorPath</key><string>$MESH_HOME/mesh.log</string>
</dict></plist>
EOF

# --- daily discovery plist ----------------------------------------------------
cat > "$PLIST_DIR/$LABEL_DISCOVER.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL_DISCOVER</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/curl</string>
    <string>-s</string><string>-m</string><string>1800</string>
    <string>-X</string><string>POST</string>
    <string>http://$MESH_HOST:$MESH_PORT/mesh/probe</string>
  </array>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>$MESH_DISCOVER_HOUR</integer>
    <key>Minute</key><integer>$MESH_DISCOVER_MIN</integer>
  </dict>
  <key>StandardOutPath</key><string>$MESH_HOME/discover.log</string>
  <key>StandardErrorPath</key><string>$MESH_HOME/discover.log</string>
</dict></plist>
EOF

# The provider API key is never written into a plist or a repo file. Supply it
# either way:
#   launchctl setenv NVIDIA_API_KEY <key>     (lost on restart)
#   echo 'NVIDIA_API_KEY=<key>' >> $MESH_HOME/.env && chmod 600 $MESH_HOME/.env
# The daemon reads the env first and falls back to that file at call time, so
# the file alone survives a reboot. Override its location with
# MODEL_MESH_KEY_FALLBACK_FILE.

launchctl unload "$PLIST_DIR/$LABEL_DAEMON.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DAEMON.plist"
launchctl unload "$PLIST_DIR/$LABEL_DISCOVER.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DISCOVER.plist"

echo "model-mesh loaded on $MESH_HOST:$MESH_PORT (labels: $LABEL_DAEMON, $LABEL_DISCOVER)"
echo "discovery runs daily at $MESH_DISCOVER_HOUR:$(printf '%02d' "$MESH_DISCOVER_MIN")"
echo "Verify: curl -s http://$MESH_HOST:$MESH_PORT/health"
