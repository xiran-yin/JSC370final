#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Running data + modeling pipeline..."
python3 scripts/final_pipeline.py

echo "[2/4] Rendering website pages..."
quarto render index.qmd
quarto render viz.qmd
quarto render slides.qmd

echo "[3/4] Rendering final report (HTML + PDF)..."
quarto render report.qmd --to html
quarto render report.qmd --to pdf

echo "[4/4] Rendering full project for consistency..."
quarto render

echo ""
echo "Build complete."
echo "- Website home: _site/index.html"
echo "- Report PDF: _site/report.pdf"
echo "- Slides: _site/slides.html"
echo "- Interactive viz: _site/viz.html"
