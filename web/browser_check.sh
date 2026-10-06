#!/usr/bin/env bash
# In-browser checks with headless Chrome against the real server (server/server.ts): parity, scripted rounds, Lab, a ranked run.
# Usage: cd web && PATH=$HOME/.local/node/bin:$PATH ./browser_check.sh
set -euo pipefail
cd "$(dirname "$0")"
npm run build >/dev/null
PORT=${PORT:-8765}
(cd .. && node server/server.ts --port "$PORT" --db "${TMPDIR:-/tmp}/swatter-check-$$.sqlite") >/dev/null 2>&1 &
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
t=$(run "http://localhost:$PORT/index.html?mode=lab&demo=lab" | grep -o '<title>[^<]*' | sed 's/<title>//')
echo "demo lab: $t"
note=$(run "http://localhost:$PORT/index.html" | grep -o 'id="mode-note">[^<]*' | sed 's/id="mode-note">//')
echo "classic: $note"
echo "$note" | grep -q "Ranked run" || { echo "FAIL: no ranked run from the server"; exit 1; }
rm -f "${TMPDIR:-/tmp}/swatter-check-$$.sqlite"
