// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// The chat's server: the model opened inside this process with janas.node,
// the page and its scripts served, and each reply streamed to the browser
// as it is written (server-sent events). The page keeps the conversation and
// sends it whole; the engine reads again only what is new.
//
//     npm start -- --model path/to/model.jns [--port 8090] [--host 127.0.0.1]
//                  [--cache GiB] [--context tokens] [--mtp] [--addon janas.node]

import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { parseChatRequest, type ModelInfo, type ReplyCost, type StreamEvent } from "../shared/protocol.js";
import { encodeEvent } from "../shared/sse.js";
import { loadJanas, type ReplyStats } from "./janas.js";
import { Line } from "./line.js";

const { values: args } = parseArgs({
    options: {
        model: { type: "string" },
        port: { type: "string", default: "8090" },
        host: { type: "string", default: "127.0.0.1" },
        cache: { type: "string" },
        context: { type: "string", default: "8192" },
        mtp: { type: "boolean", default: false },
        addon: { type: "string" },
    },
});
if (!args.model) {
    console.error(
        "usage: npm start -- --model path/to/model.jns [--port 8090] [--host 127.0.0.1]\n" +
            "                    [--cache GiB] [--context tokens] [--mtp] [--addon janas.node]",
    );
    process.exit(2);
}

const janas = loadJanas(args.addon);
console.log(`Janas ${janas.version}, Node ${process.version}: opening ${args.model}`);
const opened = performance.now();
const model = await janas.open(args.model, {
    context: Number(args.context),
    mtp: args.mtp,
    ...(args.cache !== undefined ? { cacheGiB: Number(args.cache) } : {}),
});
console.log(`${model.name || model.path}: ${model.description}, open in ${seconds(performance.now() - opened)} s`);
const chat = await model.chat({ keep: 4 });
const line = new Line();
let applied: { thinking: boolean | null; temperature: number } | null = null;

const root = fileURLToPath(new URL("../../../", import.meta.url));
const info: ModelInfo = {
    name: model.name,
    description: model.description,
    reasons: model.reasons,
    defaultSystem: model.defaultSystem,
    mtp: model.mtp !== false,
    janas: janas.version,
    node: process.version,
};

function seconds(ms: number): string {
    return (ms / 1000).toFixed(1);
}

function cost(s: ReplyStats): ReplyCost {
    return {
        finish: s.finish,
        promptTokens: s.promptTokens,
        cachedTokens: s.cachedTokens,
        outputTokens: s.outputTokens,
        reasoningTokens: s.reasoningTokens,
        drafted: s.drafted,
        accepted: s.accepted,
        firstTokenSeconds: s.firstTokenSeconds,
        inputSeconds: s.inputSeconds,
        outputSeconds: s.outputSeconds,
        totalSeconds: s.totalSeconds,
    };
}

const HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:",
    "X-Content-Type-Options": "nosniff",
};

function json(res: http.ServerResponse, status: number, body: unknown): void {
    res.writeHead(status, { ...HEADERS, "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify(body));
}

const TYPES: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
};

/** a file of the page, never outside its directory */
async function file(res: http.ServerResponse, dir: string, name: string): Promise<void> {
    const base = join(root, dir);
    const path = normalize(join(base, name));
    const type = TYPES[extname(path)];
    if (!path.startsWith(base + sep) || !type) return json(res, 404, { error: "not found" });
    try {
        const body = await readFile(path);
        res.writeHead(200, { ...HEADERS, "Content-Type": type, "Cache-Control": "no-cache" });
        res.end(body);
    } catch {
        json(res, 404, { error: "not found" });
    }
}

function body(req: http.IncomingMessage, limit: number): Promise<string> {
    return new Promise((resolve, reject) => {
        const parts: Buffer[] = [];
        let size = 0;
        req.on("data", (b: Buffer) => {
            size += b.length;
            if (size > limit) {
                reject(new Error(`a request over ${limit} bytes`));
                req.destroy();
            } else parts.push(b);
        });
        req.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
        req.on("error", reject);
    });
}

async function reply(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    let parsed: unknown;
    try {
        parsed = JSON.parse(await body(req, 4 << 20));
    } catch (e) {
        return json(res, 400, { error: `not a request: ${(e as Error).message}` });
    }
    const request = parseChatRequest(parsed);
    if (typeof request === "string") return json(res, 400, { error: request });

    res.writeHead(200, {
        ...HEADERS,
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
    });
    const gone = new AbortController();
    res.on("close", () => gone.abort()); // the page stopped it, or went away
    const send = (e: StreamEvent): void => {
        if (!res.writableEnded && !res.destroyed) res.write(encodeEvent(e));
    };
    if (line.length > 0) send({ type: "queued", ahead: line.length });

    await line.run(async () => {
        if (gone.signal.aborted) return;
        const want = {
            thinking: model.reasons ? (request.options?.thinking ?? false) : null,
            temperature: request.options?.temperature ?? 0.7,
        };
        if (!applied || applied.thinking !== want.thinking || applied.temperature !== want.temperature) {
            await chat.set(want);
            applied = want;
        }
        const r = chat.load(request.messages, { signal: gone.signal });
        // the first prompt of a session waits while the NPU is made ready
        let writing = false;
        const watch = setInterval(() => {
            const s = model.status();
            if (!writing && s.npu.preparing)
                send({ type: "preparing", done: s.npu.done, total: s.npu.total, seconds: s.npu.seconds });
        }, 500);
        try {
            for await (const piece of r) {
                if (piece.type === "progress") send(piece);
                else {
                    writing = true;
                    send({ type: "text", text: piece.text, reasoning: piece.reasoning });
                }
            }
            const c = cost(await r.stats());
            send({ type: "done", cost: c });
            const rate = c.outputSeconds > 0 ? (c.outputTokens / c.outputSeconds).toFixed(1) : "-";
            console.log(
                `reply: ${c.promptTokens} prompt tokens (${c.cachedTokens} kept), ` +
                    `${c.outputTokens} written at ${rate} tok/s, ${c.finish ?? "-"}`,
            );
        } catch (e) {
            const err = e as { code?: string; message?: string };
            send({ type: "error", code: err.code ?? "ERROR", message: err.message ?? String(e) });
        } finally {
            clearInterval(watch);
        }
    });
    res.end();
}

async function route(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    if (req.method === "POST" && path === "/api/chat") return reply(req, res);
    if (req.method !== "GET") return json(res, 405, { error: "GET or POST /api/chat" });
    if (path === "/api/model") return json(res, 200, info);
    if (path === "/api/status") return json(res, 200, model.status());
    if (path === "/") return file(res, "public", "index.html");
    if (path === "/style.css") return file(res, "public", "style.css");
    if (path.startsWith("/app/")) return file(res, "dist/web", path.slice(5));
    return json(res, 404, { error: "not found" });
}

const server = http.createServer((req, res) => {
    route(req, res).catch((e: unknown) => {
        console.error(e);
        if (!res.headersSent) json(res, 500, { error: String(e) });
        else res.end();
    });
});
server.listen(Number(args.port), args.host, () => console.log(`the chat: http://${args.host}:${args.port}/`));

for (const signal of ["SIGINT", "SIGTERM"] as const)
    process.on(signal, () => {
        console.log("closing the model");
        server.close();
        model.close().then(() => process.exit(0));
    });
