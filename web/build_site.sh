#!/usr/bin/env bash
# Assemble the static site into web/site/: everything the game needs in a browser and nothing else.
# There is no server and no leaderboard; the page notices that and hides the board. Upload the folder anywhere that
# serves static files (Cloudflare Pages, Netlify, GitHub Pages).   Usage: cd web && ./build_site.sh
set -euo pipefail
cd "$(dirname "$0")"
npm run build >/dev/null
rm -rf site
mkdir -p site/dist/game site/brain
cp index.html site/
cp dist/game/main.js site/dist/game/
cp brain/manifest.json brain/brain.bin brain/engine.wasm site/brain/   # parity.json is only for tests
touch site/.nojekyll                                                  # GitHub Pages: serve files as they are
du -sh site | sed 's/^/site size: /'
find site -type f | sort
