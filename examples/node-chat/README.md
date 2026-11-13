# A chat in TypeScript on Janas-LLM in Node.js

The model runs inside the server's own Node.js process, through `janas.node`, the module for Node.js that Janas's `build.sh` makes beside its programs: no other server, no port of its own, no HTTP between the two. The browser gets each reply as the model writes it.

```
browser (TypeScript)  --POST /api/chat-->  Node.js server (TypeScript)  --chat.load()-->  janas.node  -->  Janas-LLM
         <--server-sent events, piece by piece--          <--for await (const piece of reply)--
```

- `src/shared/` is compiled for both sides: the protocol's types (`ChatRequest`, `StreamEvent`, ...), the check of a request, and the coding of the events on the wire.
- `src/server/` opens the model (`janas.open()`), keeps one chat, and answers `POST /api/chat` with the reply streamed: `chat.load(messages, { signal })` and `for await` over its pieces. A page that stops or goes away aborts the request, and the reply stops with it.
- `src/web/` is the page: the conversation kept there and sent whole each time (the engine reads again only what is new), the reasoning apart from the answer, and under every reply what it cost (tokens per second, the first token's time, how much of the prompt was already read).
- The types of the module are Janas's own `src/node/janas.d.ts`.

## What it needs

- Linux x86-64 and Node.js 20 or later.
- Janas built (`./build.sh` in the repository's root), which makes `bin/x86_64-linux/janas.node`.
- A model converted for Janas-LLM (`.jns`), such as `models/jns/qwen3-4b-q4km.jns`.

## Building and running

```sh
npm install            # TypeScript and Node's types, for building only
npm run build          # dist/node/ for the server, dist/web/ for the page
npm start -- --model ../../models/jns/qwen3-4b-q4km.jns
```

Then open `http://127.0.0.1:8090/`. The options: `--port` (8090), `--host` (127.0.0.1: this machine only), `--cache` (GiB for the expert cache; by default the free memory, less a margin), `--context` (tokens, 8192), `--mtp` (the model's MTP block, looked for beside it), `--addon` (another `janas.node`, or the `JANAS_NODE` variable). The engine's own variables apply as in Janas's programs (`JANAS_GPU=0`, `JANAS_NPU=0`, `JANAS_PROMPT_ON`, ...).

The model holds one conversation: requests from several pages wait in line, and the page says how many are before its own.

## The module in a few lines

```ts
import { createRequire } from "node:module";
import type * as Janas from "../../src/node/janas.js";

const janas = createRequire(import.meta.url)("./janas.node") as typeof Janas;
const model = await janas.open("qwen3-4b-q4km.jns");
const chat = await model.chat({ thinking: false });
for await (const piece of chat.send("Explain how bread is made at home."))
    if (piece.type === "text") process.stdout.write(piece.text);
await model.close();
```
