#!/usr/bin/env bash
# Download the pre-trained manifold artifacts.
#
# Hosted on Hugging Face:
#   https://huggingface.co/datasets/sauli-aisha/aisha-manifold
#
# Two pickle files are needed for the runtime:
#
#   manifold_clean.pkl   128 MB   per-word symplectic manifold,
#                                  loaded by aisha_respond.Responder
#                                  (which responder_pos imports).
#   pos_manifold.pkl      33 MB   POS-aligned 16-d coordinates,
#                                  loaded by kahler_pos_runtime.
#
# Both are required to run anything in this repo.  This script fetches
# them into aisha/data/processed/, which is the path the loaders look
# at by default.  Already-present files are skipped (idempotent).
#
# Usage:
#     ./scripts/download_artifacts.sh

set -e
HF_BASE="https://huggingface.co/datasets/sauli-aisha/aisha-manifold/resolve/main"
DEST="aisha/data/processed"

mkdir -p "$DEST"

curl_or_die() {
  local url="$1"; local out="$2"
  if [[ -f "$out" ]]; then
    echo "  ✓ $out (already present, skipping)"
    return
  fi
  echo "  → $out"
  curl -L --fail --progress-bar -o "$out" "$url"
}

echo "Fetching manifold artifacts (~161 MB total)…"
curl_or_die "$HF_BASE/manifold_clean.pkl" "$DEST/manifold_clean.pkl"
curl_or_die "$HF_BASE/pos_manifold.pkl"   "$DEST/pos_manifold.pkl"

echo
echo "Done.  Artifacts in $DEST/."
echo "Run a chat:   python aisha/chat_demo.py"
