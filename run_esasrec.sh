#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
DATA_DIR="data/ml-20m"
RATINGS="$DATA_DIR/ratings.csv"
WORKDIR="./artifacts_100ep"

# ── venv ──────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -r requirements.txt

# ── data ──────────────────────────────────────────────────────────────
if [ ! -f "$RATINGS" ]; then
    echo "Downloading ML-20M dataset..."
    mkdir -p data
    wget -q --show-progress -O data/ml-20m.zip \
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    unzip -o data/ml-20m.zip "ml-20m/ratings.csv" -d data
    rm -f data/ml-20m.zip
fi

# ── train ─────────────────────────────────────────────────────────────
echo "Starting training..."
python3 -m esasrec.train \
    --ratings-path "$RATINGS" \
    --workdir "$WORKDIR" \
    --device cuda \
    --epochs 100 \
    --batch-size 128 \
    --amp \
    --eval-protocol realistic \
    --test-days 60 \
    --num-workers 4 \
    --seed 42 \
    --log-every 100 \
    --patience 50 \
    --test-every 1
