#!/usr/bin/env bash
# Download the pre-trained large artifacts that don't fit in the repo.
#
# Hosted on Hugging Face:
#   https://huggingface.co/datasets/sauli-aisha/aisha-manifold
#
# Files fetched:
#   pos_manifold.pkl      33 MB   POS-aligned 16-d coordinates (runtime)
#   manifold_clean.pkl   128 MB   per-word symplectic manifold (training)
#
# Usage:  ./scripts/download_artifacts.sh           runtime only (33 MB)
#         ./scripts/download_artifacts.sh --train   also fetch training pickle

set -e
HF_BASE="https://huggingface.co/datasets/sauli-aisha/aisha-manifold/resolve/main"
DEST="aisha/data/processed"

mkdir -p "$DEST"

curl_or_die() {
  local url="$1"; local out="$2"
  echo "  → $out"
  curl -L --fail --progress-bar -o "$out" "$url"
}

echo "Fetching runtime artifact …"
curl_or_die "$HF_BASE/pos_manifold.pkl" "$DEST/pos_manifold.pkl"

if [[ "${1:-}" == "--train" ]]; then
  echo "Fetching training artifact (128 MB) …"
  curl_or_die "$HF_BASE/manifold_clean.pkl" "$DEST/manifold_clean.pkl"
fi

echo
echo "Done.  Artifacts saved to $DEST/."
