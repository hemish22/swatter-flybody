// Leaderboard server: issues a run (a seed), takes the 20 cursor traces back, replays them with the same game code and
// WASM engine the browser plays (verify.ts), and stores the score only if the replay matches. Also serves web/ so the
// game and the API share an origin.  Run:  node server/server.ts [--port 8080] [--db server/leaderboard.sqlite]
// The server is off the gameplay path: nothing here is called while a round is being played.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { randomInt, randomUUID } from "node:crypto";
import { readFileSync, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";
import { FlyBrain, type Manifest } from "../web/engine/brain.ts";
import { rulesFor } from "../web/game/trace.ts";
import { CLASSIC_SWATS, verifyRun, type SwatSubmission } from "./verify.ts";

const WEB = fileURLToPath(new URL("../web/", import.meta.url));
const MAX_BODY = 4 * 1024 * 1024;
const RUN_TTL_MS = 2 * 60 * 60 * 1000;
const MIME: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".bin": "application/octet-stream", ".png": "image/png", ".css": "text/css" };

export async function loadBrain(): Promise<{ brain: FlyBrain; manifest: Manifest }> {
  const manifest: Manifest = JSON.parse(readFileSync(join(WEB, "brain/manifest.json"), "utf8"));
  const bin = readFileSync(join(WEB, "brain", manifest.binary));
  const wasm = readFileSync(join(WEB, "brain/engine.wasm"));
  return { brain: await FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), wasm), manifest };
}

export function cleanName(raw: unknown): string {
  const s = typeof raw === "string" ? raw.replace(/[^\p{L}\p{N} _.\-]/gu, "").trim().slice(0, 24) : "";
  return s || "anonymous";
}

export async function createApp(dbPath = ":memory:", now: () => number = Date.now): Promise<Server> {
  const { brain, manifest } = await loadBrain();
  const rules = rulesFor(manifest);
  const db = new DatabaseSync(dbPath);
  db.exec(`CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, seed INTEGER NOT NULL, issued_ms INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0);
           CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE, name TEXT NOT NULL, hits INTEGER NOT NULL, swats INTEGER NOT NULL, rules TEXT NOT NULL, at_ms INTEGER NOT NULL);`);

  const send = (res: ServerResponse, code: number, body: unknown) => {
    res.writeHead(code, { "content-type": "application/json", "cache-control": "no-store", "access-control-allow-origin": "*" });
    res.end(JSON.stringify(body));
  };
  const readBody = (req: IncomingMessage): Promise<string> => new Promise((ok, fail) => {
    let n = 0; const parts: Buffer[] = [];
    req.on("data", (c: Buffer) => { n += c.length; if (n > MAX_BODY) { fail(new Error("body too large")); req.destroy(); } else parts.push(c); });
    req.on("end", () => ok(Buffer.concat(parts).toString("utf8")));
    req.on("error", fail);
  });

  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://x");
      if (req.method === "OPTIONS") { res.writeHead(204, { "access-control-allow-origin": "*", "access-control-allow-headers": "content-type" }); return void res.end(); }

      if (url.pathname === "/api/run" && req.method === "GET") {
        const id = randomUUID(), seed = randomInt(1, 2 ** 31);
        db.prepare("INSERT INTO runs (id, seed, issued_ms) VALUES (?, ?, ?)").run(id, seed, now());
        return send(res, 200, { runId: id, seed, swats: CLASSIC_SWATS, rules });
      }

      if (url.pathname === "/api/submit" && req.method === "POST") {
        let body: { runId?: string; name?: string; rules?: string; swats?: SwatSubmission[] };
        try { body = JSON.parse(await readBody(req)); } catch { return send(res, 400, { accepted: false, reason: "unreadable body" }); }
        if (body.rules !== rules) return send(res, 409, { accepted: false, reason: "this page is running other game rules than the server; reload" });
        const run = db.prepare("SELECT seed, issued_ms, used FROM runs WHERE id = ?").get(String(body.runId)) as { seed: number; issued_ms: number; used: number } | undefined;
        if (!run) return send(res, 404, { accepted: false, reason: "unknown run" });
        if (run.used) return send(res, 409, { accepted: false, reason: "run already submitted" });
        if (now() - run.issued_ms > RUN_TTL_MS) return send(res, 410, { accepted: false, reason: "run expired" });
        db.prepare("UPDATE runs SET used = 1 WHERE id = ?").run(String(body.runId)); // a failed submission spends the run too
        const v = verifyRun(brain, run.seed, body.swats as SwatSubmission[]);
        if (!v.ok) return send(res, 422, { accepted: false, reason: v.reason, swat: v.swat });
        const name = cleanName(body.name);
        db.prepare("INSERT INTO scores (run_id, name, hits, swats, rules, at_ms) VALUES (?, ?, ?, ?, ?, ?)").run(String(body.runId), name, v.hits, CLASSIC_SWATS, rules, now());
        const better = db.prepare("SELECT COUNT(*) AS n FROM scores WHERE hits > ?").get(v.hits) as { n: number };
        return send(res, 200, { accepted: true, hits: v.hits, rank: better.n + 1 });
      }

      if (url.pathname === "/api/leaderboard" && req.method === "GET") {
        const limit = Math.max(1, Math.min(100, Number(url.searchParams.get("limit")) || 20));
        const rows = db.prepare("SELECT name, hits, swats, at_ms AS at FROM scores WHERE rules = ? ORDER BY hits DESC, at_ms ASC LIMIT ?").all(rules, limit);
        return send(res, 200, { rules, rows });
      }

      if (req.method === "GET" && !url.pathname.startsWith("/api/")) {
        const rel = normalize(decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname)).replace(/^([/\\])+/, "");
        const file = resolve(WEB, rel);
        if (!file.startsWith(resolve(WEB)) || rel.includes("node_modules") || !existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); return void res.end("not found"); }
        res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
        return void res.end(readFileSync(file));
      }
      send(res, 404, { error: "not found" });
    } catch (e) {
      send(res, 500, { error: String(e) });
    }
  });
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("server/server.ts")) {
  const arg = (k: string, d: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
  const port = Number(arg("--port", "8080")), db = arg("--db", "server/leaderboard.sqlite");
  createApp(db).then((s) => s.listen(port, () => console.log(`SWATTER server on http://localhost:${port}  (db ${db})`)));
}
