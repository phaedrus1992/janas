# Using it from a program

The library is shaped for foreign-function interfaces from the start: opaque handles, fixed-width types, nothing passed by value that is not a number, parameter blocks that carry their own size, error codes as `int32`, and reply text returned on request as whole UTF-8 characters. It is declared for **C** and for **FreeBASIC**, and it is a module for **Node.js** too, with its types for TypeScript.

## From C

[`include/janas/llm.h`](../include/janas/llm.h).

```c
janas_llm *llm;
janas_llm_chat *chat;
struct janas_llm_params p;
janas_llm_params_default(&p);
janas_llm_open("qwen3-next.jns", &p, &llm);
janas_llm_chat_create(llm, NULL, &chat);
janas_llm_chat_send(chat, "Hello! Introduce yourself.", -1);

char piece[256];
int32_t len;
while (janas_llm_chat_next(chat, piece, sizeof(piece), &len) == JANAS_LLM_OK)
    fwrite(piece, 1, (size_t)len, stdout);
```

```sh
cc yours.c -Iinclude -Lbin/x86_64-linux -ljanas_llm -o yours
```

A program that keeps the conversation itself - an HTTP client does, and sends it whole every time - hands it over with `janas_llm_chat_load`: the messages with their roles, the last one the user's. Whatever the new conversation shares with the one already computed is not read again, so the same messages plus a new one cost only the new one. `janas_llm_chat_prompt` continues raw text, with no chat format around it, and `janas_llm_default_system` gives the system message Janas's own programs use. `janas_llm_chat_stats` says why a reply ended and how much of its prompt was already computed. `janas_llm_chat_keep` keeps a few conversations computed besides the current one, for a program that switches between them: the one that shares most of the next request is copied back instead of being read again. `janas_llm_chat_prepare` reads the system message and the tools before the first message, a block at a time, so that a program can read them while its user writes; with `janas_llm_chat_keep_disk` a system message read once on a model is found on the disk by the next chat.

## From FreeBASIC and BASIC MODERN

[`include/janas/llm.bi`](../include/janas/llm.bi) declares the same library for FreeBASIC, and so for **BASIC MODERN**, the dialect implemented by Prabanta — the platform Janas is meant to sit inside. The same conversation as above:

```basic
#include once "janas/llm.bi"

dim as janas_llm ptr llm
dim as janas_llm_chat ptr chat
dim as janas_llm_params p
janas_llm_params_default(@p)
janas_llm_open("qwen3-next.jns", @p, @llm)
janas_llm_chat_create(llm, NULL, @chat)
janas_llm_chat_send(chat, "Hello! Introduce yourself.", -1)

dim as zstring * 256 piece
dim as long n
do
    '' the piece is not closed by a NUL: one byte is left for it
    dim as long r = janas_llm_chat_next(chat, @piece, sizeof(piece) - 1, @n)
    if r <> JANAS_LLM_OK then exit do
    piece[n] = 0
    print piece;
loop
```

[`tools/chat.bas`](../tools/chat.bas) is a whole chat in a hundred lines, sampling and counters included:

```sh
fbc tools/chat.bas -i include -p bin/x86_64-linux -x janas-chat-fb
LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
echo "Hello! Introduce yourself." | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
```

Two things to know. FreeBASIC does not tell names apart by their case, so the constant the C header calls `JANAS_LLM_ABI_VERSION` is `JANAS_LLM_ABI` there — it would otherwise collide with the function `janas_llm_abi_version()`. And its `line input` reads the terminal, not the standard input: to work in a pipe too, `chat.bas` asks `isatty` and reads a pipe through the `CONS` device.

## From Node.js

`janas.node` is the library as a module for Node.js, made by `./build.sh` on Linux beside the programs (`bin/x86_64-linux/janas.node`), with its types for TypeScript beside it (`janas.d.ts`). It is written in C against Node-API, version 8, whose functions it declares itself: building it needs nothing of Node's. It was tried with Node 20. Each model runs on a thread of its own, so JavaScript's thread never waits for it, and a reply is read as it is written:

```ts
import { createRequire } from "node:module";
import type * as Janas from "./janas.js"; // janas.d.ts

const janas = createRequire(import.meta.url)("./janas.node") as typeof Janas;
const model = await janas.open("qwen3-next.jns", { context: 8192 });
const chat = await model.chat({ thinking: false });
for await (const piece of chat.send("Hello! Introduce yourself."))
    if (piece.type === "text") process.stdout.write(piece.text);
await model.close();
```

A piece is the model's text (`reasoning` says whether it is its reasoning or its answer), or, while a long prompt is read, how much of it is (`type: "progress"`, `read` of `total` tokens). `chat.load(messages)` takes a whole conversation, as a server gets it from its client; what it shares with the conversation already computed is not read again. A reply stops when the loop is left (`break`, `return`, an exception), with `reply.stop()`, or with an `AbortSignal` given as `{ signal }` - the client of a request gone. `reply.text()` gives the whole answer, its reasoning left out, and `reply.stats()` the counters (tokens, seconds, how much of the prompt was already read, why it ended). A model holds one conversation, and one chat: replies asked for while another is written wait for it, in order. `model.status()` says, without waiting for the engine, how the expert cache fills and the NPU is made ready. Wrong arguments throw a `TypeError` that names them; what fails in the engine rejects with an `Error` whose `code` says what (`EOPEN`, `EBUSY`, `EFULL`, ...: the suffixes of the library's `JANAS_LLM_E...`). An idle model does not keep Node running, and a model left open is closed when Node leaves.

What has been run: `tests/test_node`, among the tests `./build.sh test` runs (the module without a model: its classes, its errors, engine threads started and ended), and `tests/node_check` on Qwen3-0.6B: the replies through Node the same text, byte for byte, as the library's own in C, streamed and read whole, after a reset and in a new chat; a reply stopped by leaving the loop, by a signal fired and by one aborted before; two replies asked at once; a prompt of several blocks; reasoning on and off; closing twice; a model left open when Node leaves. Both also under AddressSanitizer and ThreadSanitizer, the sanitizer's runtime preloaded into node. The module is not built for Windows yet.

[`examples/node-chat`](../examples/node-chat/README.md) is a chat in TypeScript on it, server and page: the model inside the server's process, its replies streamed to the browser as they are written, the types of what goes between the two shared by both sides.

## MCP servers from a program

[`include/janas/mcp.h`](../include/janas/mcp.h) is the MCP client the chat uses, as a library of its own, `libjanas_mcp`, which needs nothing of `libjanas_llm`; [`include/janas/mcp.bi`](../include/janas/mcp.bi) declares it for FreeBASIC. It fits the model's side of the tools: `janas_mcp_tools` gives a server's tools as the JSON `janas_llm_chat_tools` takes, the calls the model writes go to `janas_mcp_call`, and what `janas_mcp_result` returns goes back with `janas_llm_chat_send_results`.

```c
janas_mcp *files;
janas_mcp_open(NULL, "files", 0, &files); /* from ~/.config/janas/mcp.json */
char tools[65536];
int32_t len;
janas_mcp_tools(files, "files__", -1, tools, sizeof(tools), &len);
janas_llm_chat_tools(chat, tools, len);
/* ... a reply that ends with calls: for each, name without "files__" */
janas_mcp_call(files, name + 7, -1, args, -1, 60);
janas_mcp_result(files, answer, sizeof(answer), &len);
/* ... then janas_llm_chat_send_results(chat, n, answers, NULL) */
```

```sh
cc yours.c -Iinclude -Lbin/x86_64-linux -ljanas_llm -ljanas_mcp -o yours
```
