#!/usr/bin/env bash
# Download OS-specific raw logs from Zenodo, clone the repo, and place logs
# under raw_logs/<os>/.
#
# Usage:
#   ./scripts/fetch_and_prepare_logs.sh [TARGET_DIR]
#     TARGET_DIR defaults to ./AI-Driven-Log-Analysis-Tool-for-Predictive-Cybersecurity-Alerts
#
# Env overrides:
#   REPO_URL   - repository to clone (default: Thanfees/AI-Driven-Log-Analysis-Tool-for-Predictive-Cybersecurity-Alerts)
#   CURL_OPTS  - extra flags for curl (e.g., "--retry 3")
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Thanfees/AI-Driven-Log-Analysis-Tool-for-Predictive-Cybersecurity-Alerts.git}"
TARGET_DIR="${1:-AI-Driven-Log-Analysis-Tool-for-Predictive-Cybersecurity-Alerts}"

declare -a DATASETS=(
  "linux|https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1"
  "mac|https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1"
  "windows|https://zenodo.org/records/8196385/files/Windows.tar.gz?download=1"
)

log() { printf "\\n[%s] %s\\n" "$(date +'%F %T')" "$*"; }

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

log "Cloning/pulling repo into $TARGET_DIR"
if [[ -d "$TARGET_DIR/.git" ]]; then
  git -C "$TARGET_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

raw_base="$TARGET_DIR/raw_logs"
mkdir -p "$raw_base"

for entry in "${DATASETS[@]}"; do
  os="${entry%%|*}"
  url="${entry#*|}"
  os_dir="$raw_base/$os"
  mkdir -p "$os_dir"

  log "Downloading $os logs from $url"
  archive="$tmpdir/${os}.tar.gz"
  curl -L ${CURL_OPTS:-} "$url" -o "$archive"

  extract_dir="$tmpdir/extract_$os"
  mkdir -p "$extract_dir"
  tar -xzf "$archive" -C "$extract_dir"

  log "Placing $os logs into $os_dir"
  # Move all files (keep original filenames); overwrite if present.
  find "$extract_dir" -type f -print0 | while IFS= read -r -d '' file; do
    mv -f "$file" "$os_dir/"
  done
done

log "Done. Logs are under $raw_base/{linux,mac,windows}"
