#!/usr/bin/env bash
set -euo pipefail

find . -name ".DS_Store" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
rm -rf __MACOSX .mplconfig
