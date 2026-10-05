#!/usr/bin/env bash
# In-browser checks with headless Chrome: the 50-stimulus parity run and a scripted round.
# Usage: cd web && PATH=$HOME/.local/node/bin:$PATH ./browser_check.sh
set -euo pipefail
cd "$(dirname "$0")"
npm run build >/dev/null
PORT=${PORT:-8765}
python3 -m http.server "$PORT" --directory . >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 1
CHROME=${CHROME:-google-chrome}
run() { timeout 90 "$CHROME" --headless=new --no-sandbox --disable-gpu --virtual-time-budget=20000 --dump-dom "$1" 2>/dev/null; }
parity=$(run "http://localhost:$PORT/parity.html" | grep -o '<pre id="result">[^<]*' | sed 's/<pre id="result">//')
echo "parity: $parity"
echo "$parity" | grep -q '"ok":true' || { echo "FAIL: in-browser parity"; exit 1; }
for d in hit escape spook; do
  t=$(run "http://localhost:$PORT/index.html?demo=$d" | grep -o '<title>[^<]*' | sed 's/<title>//')
  echo "demo $d: $t"
done
