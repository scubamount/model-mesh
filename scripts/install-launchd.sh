#!/bin/sh
# Install model-mesh as a launchd daemon (:8002) + daily discovery job.
# Idempotent. nim-proxy stays on :8001 until cutover.
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"
MESH_HOME="$HOME/.model-mesh"
LABEL_DAEMON="com.scubamount.model-mesh"
LABEL_DISCOVER="com.scubamount.model-mesh-discover"
PY="$REPO_DIR/.venv/bin/python"

mkdir -p "$MESH_HOME" "$PLIST_DIR"

[ -x "$PY" ] || {
  echo "venv missing — creating"
  env -u PYTHONPATH /opt/homebrew/bin/python3.12 -m venv "$REPO_DIR/.venv"
  "$REPO_DIR/.venv/bin/pip" -q install -e "$REPO_DIR"
}

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
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8002</string>
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
    <string>-s</string><string>-m</string><string>600</string>
    <string>-X</string><string>POST</string>
    <string>http://127.0.0.1:8002/mesh/probe</string>
  </array>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>6</integer><key>Minute</key><integer>15</integer>
  </dict>
  <key>StandardOutPath</key><string>$MESH_HOME/discover.log</string>
  <key>StandardErrorPath</key><string>$MESH_HOME/discover.log</string>
</dict></plist>
EOF

# NVIDIA_API_KEY comes from the launchctl user env (same as nim-proxy):
#   launchctl setenv NVIDIA_API_KEY <key>
# We never write the key into a plist or repo file.

launchctl unload "$PLIST_DIR/$LABEL_DAEMON.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DAEMON.plist"
launchctl unload "$PLIST_DIR/$LABEL_DISCOVER.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DISCOVER.plist"

echo "model-mesh daemon loaded on 127.0.0.1:8002; discovery daily 06:15."
echo "Verify: curl -s http://127.0.0.1:8002/health"
