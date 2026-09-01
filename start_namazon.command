#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "NAmazon requires Python 3. Install Python, then run this launcher again."
  exit 1
fi
exec python3 run_namazon.py "$@"
