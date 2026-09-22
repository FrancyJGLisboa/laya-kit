#!/usr/bin/env bash
# Install the laya skill into every agent CLI found on this machine.
# They share one layout: <config>/skills/<name>/SKILL.md
#   ./scripts/install.sh            symlink (recommended, the clone stays the source of truth)
#   ./scripts/install.sh --copy
#   ./scripts/install.sh --uninstall
set -euo pipefail

NAME="laya"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../skills/laya" && pwd)"
MODE="${1:---link}"

TARGETS=(
  "$HOME/.agents/skills:agents"
  "$HOME/.claude/skills:Claude Code"
  "$HOME/.codex/skills:Codex CLI"
  "$HOME/.copilot/skills:Copilot CLI"
  "$HOME/.gemini/skills:Gemini CLI"
)

for entry in "${TARGETS[@]}"; do
  dir="${entry%%:*}"; label="${entry#*:}"; parent="$(dirname "$dir")"
  if [ ! -d "$parent" ]; then
    printf '%-14s skipped (%s not present)\n' "$label" "$parent"; continue
  fi
  mkdir -p "$dir"; dest="$dir/$NAME"
  case "$MODE" in
    --uninstall) rm -rf "$dest"; printf '%-14s removed\n' "$label" ;;
    --copy)      rm -rf "$dest"; cp -R "$SRC" "$dest"; printf '%-14s copied  -> %s\n' "$label" "$dest" ;;
    *)           rm -rf "$dest"; ln -s "$SRC" "$dest"; printf '%-14s linked  -> %s\n' "$label" "$dest" ;;
  esac
done
