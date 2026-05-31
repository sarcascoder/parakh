#!/usr/bin/env bash
# One-click: run the offline demo, then launch the review UI.
# Usage: ./run.sh           (demo + review UI)
#        ./run.sh demo      (just the text report)
set -e
cd "$(dirname "$0")"

PY=python3
command -v $PY >/dev/null 2>&1 || PY=python

echo "==> Parakh demo (offline, no model needed)"
PYTHONPATH=. $PY -m examples.invoices.run_demo

if [ "$1" != "demo" ]; then
  echo ""
  echo "==> Starting review UI at http://127.0.0.1:8000  (press Ctrl-C to stop)"
  PYTHONPATH=. $PY -m parakh.cli review
fi
