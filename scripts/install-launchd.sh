#!/bin/sh
# Install model-mesh as a launchd user agent + daily discovery job (macOS).
#
# Idempotent AND adoptive: if an install already exists on this machine, its
# label prefix and state directory are REUSED unless you override them. That
# matters on reinstall — changing the default prefix would load a second daemon
# against the same port while the first kept running, and would strand the old
# mesh.db (learned state: sample history, breaker states, EOL marks, which
# nothing regenerates).
#
# Overrides, all optional:
#   MESH_LABEL_PREFIX   reverse-DNS prefix for the launchd labels
#                       (adopted from an existing install; else "local")
#   MESH_HOME           state dir: db, config, .env, logs, audit
#                       (adopted from an existing install; else ~/.model-mesh)
#   MESH_HOST/MESH_PORT listen address, written to config.yaml (127.0.0.1:8002)
#   MESH_PYTHON         interpreter used to create the venv
#                       (default: first python3.12+ on PATH)
#   MESH_DISCOVER_INTERVAL_S                 seconds between discovery passes
#                                            (21600 = 6h). Must stay well under
#                                            SCORE_WINDOW_S * REFRESH_MARGIN
#                                            (19.2h) or traffic-free lanes lose
#                                            their evidence — see the plist
#                                            comment below.
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"

# --- adopt an existing install ----------------------------------------------
# Find any already-installed model-mesh agent and reuse its identity. Discovered
# from the filesystem rather than from a hardcoded list, so it adopts whatever
# prefix this machine actually used.
EXISTING_PLIST=""
for p in "$PLIST_DIR"/*.model-mesh.plist; do
  [ -e "$p" ] || continue
  EXISTING_PLIST="$p"
  break
done

ADOPTED_PREFIX=""
ADOPTED_HOME=""
if [ -n "$EXISTING_PLIST" ]; then
  b="$(basename "$EXISTING_PLIST" .model-mesh.plist)"
  ADOPTED_PREFIX="$b"
  # StandardOutPath is "<state dir>/mesh.log" — read the state dir back off it.
  logline="$(/usr/libexec/PlistBuddy -c 'Print :StandardOutPath' "$EXISTING_PLIST" 2>/dev/null || true)"
  [ -n "$logline" ] && ADOPTED_HOME="$(dirname "$logline")"
fi

MESH_LABEL_PREFIX="${MESH_LABEL_PREFIX:-${ADOPTED_PREFIX:-local}}"
MESH_HOME="${MESH_HOME:-${ADOPTED_HOME:-$HOME/.model-mesh}}"
MESH_HOST="${MESH_HOST:-127.0.0.1}"
MESH_PORT="${MESH_PORT:-8002}"
MESH_DISCOVER_INTERVAL_S="${MESH_DISCOVER_INTERVAL_S:-21600}"

LABEL_DAEMON="$MESH_LABEL_PREFIX.model-mesh"
LABEL_DISCOVER="$MESH_LABEL_PREFIX.model-mesh-discover"
PY="$REPO_DIR/.venv/bin/python"

if [ -n "$EXISTING_PLIST" ]; then
  echo "adopting existing install: $LABEL_DAEMON (state: $MESH_HOME)"
else
  echo "fresh install: $LABEL_DAEMON (state: $MESH_HOME)"
fi

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
fi
# Always (re)install: an adopted venv may predate the console entrypoint.
"$REPO_DIR/.venv/bin/pip" -q install -e "$REPO_DIR"

# --- listen address lives in config, not in the plist -------------------------
# The daemon serves whatever `listen` says, so this is the one place the address
# is defined. Written only when absent — never clobber an operator's config.
if [ ! -f "$MESH_HOME/config.yaml" ]; then
  printf 'listen:\n  host: %s\n  port: %s\n' "$MESH_HOST" "$MESH_PORT" > "$MESH_HOME/config.yaml"
  echo "wrote $MESH_HOME/config.yaml (listen $MESH_HOST:$MESH_PORT)"
else
  echo "keeping existing $MESH_HOME/config.yaml — edit it to change the address"
fi

# The address the discovery job must curl has to match what the daemon serves,
# so read it back from the config that the daemon itself will read. MESH_HOME is
# exported so this resolves the ADOPTED state dir, not the default one.
EFFECTIVE="$(MESH_HOME="$MESH_HOME" "$PY" -c 'from model_mesh.config import load_config
c = load_config()["listen"]
print("%s:%s" % (c["host"], c["port"]))')"

# --- daemon plist -----------------------------------------------------------
cat > "$PLIST_DIR/$LABEL_DAEMON.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL_DAEMON</string>
  <key>ProgramArguments</key><array>
    <string>$REPO_DIR/.venv/bin/model-mesh</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>MESH_HOME</key><string>$MESH_HOME</string>
  </dict>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$MESH_HOME/mesh.log</string>
  <key>StandardErrorPath</key><string>$MESH_HOME/mesh.log</string>
</dict></plist>
EOF

# --- discovery plist ----------------------------------------------------------
# StartInterval (elapsed-time), NOT StartCalendarInterval (wall-clock). Two
# independent reasons a "once daily at HH:MM" schedule cannot keep a lane scored:
#
#   1. PHASE. Index.score() ignores samples older than SCORE_WINDOW_S (24h) and
#      discovery refreshes at REFRESH_MARGIN * that window (19.2h). A pass that
#      runs once per 24h cannot honour a 19.2h threshold. Measured 2026-09-01:
#      auto/evolve opened 6 gaps >24h in 14 days (largest 41.45h) during which
#      it carried 0 scored models, so the router ordered its entire pool on
#      priors. Polling faster than the threshold is what makes it real.
#
#   2. SLEEP. launchd does not run a StartCalendarInterval job whose minute
#      passed while the machine was asleep, and does not run it late afterwards.
#      A laptop asleep at 06:15 loses that whole day's refresh silently.
#      StartInterval is elapsed-time and launchd runs a missed interval on wake.
#
# Why this matters for one lane in particular: auto/retain and
# auto/consolidation are refreshed for free by live hindsight traffic.
# auto/evolve has its OWN op_class deliberately (config.py — evolution traffic
# must not vote in hindsight's ranking), so it has no organic traffic and this
# pass is its only evidence source. The isolation that protects retain's
# ranking is exactly what starves evolve of evidence.
#
# Cost stays bounded: probe_top_n caps breadth per pass and dormant_after_s
# skips models with no successes in the window, so a pass is a handful of
# requests, not a catalog sweep — probing consumes shared free quota, so this
# must never become "probe every model".
cat > "$PLIST_DIR/$LABEL_DISCOVER.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL_DISCOVER</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/curl</string>
    <string>-s</string><string>-m</string><string>1800</string>
    <string>-X</string><string>POST</string>
    <string>http://$EFFECTIVE/mesh/probe</string>
  </array>
  <key>StartInterval</key><integer>$MESH_DISCOVER_INTERVAL_S</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$MESH_HOME/discover.log</string>
  <key>StandardErrorPath</key><string>$MESH_HOME/discover.log</string>
</dict></plist>
EOF

# The provider API key is never written into a plist or a repo file. Supply it
# either way:
#   launchctl setenv NVIDIA_API_KEY <key>     (lost on restart)
#   echo 'NVIDIA_API_KEY=<key>' >> $MESH_HOME/.env && chmod 600 $MESH_HOME/.env
# The daemon reads the env first and falls back to that file at call time, so
# the file alone survives a reboot.

launchctl unload "$PLIST_DIR/$LABEL_DAEMON.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DAEMON.plist"
launchctl unload "$PLIST_DIR/$LABEL_DISCOVER.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$LABEL_DISCOVER.plist"

echo "model-mesh loaded on $EFFECTIVE (labels: $LABEL_DAEMON, $LABEL_DISCOVER)"
echo "discovery runs every $((MESH_DISCOVER_INTERVAL_S / 3600))h (elapsed-time, survives sleep)"
echo "Verify: curl -s http://$EFFECTIVE/health"
