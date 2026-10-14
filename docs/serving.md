# Serving the model

## Janas as an MCP server

`janas-mcp` offers the model running here to the clients of the Model Context Protocol - Claude Code, editors, other agents - as tools: `generate` asks it a question (a prompt, and optionally a system message, a longest answer and a temperature) and returns its answer, and `embed` returns the vector of a text when an embedding model is given with `--embedding-model`. The client starts it and speaks to it over its standard input and output; it is configured like any other stdio server:

```json
{"mcpServers": {"janas": {"command": "/path/to/janas-mcp",
                          "args": ["/path/to/qwen3-4b.jns"]}}}
```

It answers both eras of the protocol: a request that names revision 2026-07-28 in its `_meta` is served statelessly, and a client that opens with `initialize` gets the handshake of the revision it asks for (2025-11-25, 2025-06-18, 2025-03-26 or 2024-11-05). A ping is answered while a generation runs, and `notifications/cancelled` stops it. Every call starts a conversation of its own; the reasoning of a model that reasons is never returned, only the answer. `janas-mcp --help` lists the options (`--ctx`, `--temp`, `--max`, `--think`, `--timeout` and the memory ones).

What has been run: a probe script speaking each era to it with Qwen3-0.6B and Qwen3-Embedding-0.6B, `janas-chat` with Qwen3-4B calling its two tools through `libjanas_mcp`, and Claude Code (2.1.282, revision 2026-07-28), which listed both tools and called them - a translation, a summary, an embedding - with Qwen3-4B and Qwen3-Embedding-0.6B behind them. To add it there: `claude mcp add janas -- /path/to/janas-mcp /path/to/model.jns`.

## Using it over HTTP

```sh
bin/x86_64-linux/janas-server qwen3-next.jns
```

`janas-server` puts the model behind an HTTP API that follows [OpenAI's](https://github.com/openai/openai-openapi), on `http://127.0.0.1:8080/v1`, so that a client written for that API can use a model running on this machine. It takes the model options `janas-chat` takes (`--ctx`, `--cache`, `--reserve`, `--mtp`, `--draft`, `--mode`, `--no-gpu`) and these:

| Option | |
|---|---|
| `--host ADDR` | the address to listen on; `127.0.0.1` (the default) is this machine only, `0.0.0.0` opens it to the network |
| `--port N` | 8080 by default |
| `--api-key KEY` | ask every client for this key, as `Authorization: Bearer KEY`; `JANAS_API_KEY` sets it too |
| `--name ID` | the model's name for clients; the file's name without `.jns` by default |
| `--think on\|off` | reasoning before replying, for models that do it, when a request does not say |
| `--reasoning-budget N` | the longest reasoning, in tokens, when a request does not say: there it is closed and the model answers (-1: no limit, the default; 0: no reasoning) |
| `--test` | serve a chat page for trying it out, on `/test` (below) |
| `--max-input N` | the longest input a request may have, in tokens, below the context: a longer one gets a 400 (`input_limit_exceeded`) whose error says `input_tokens`, `max_input_tokens`, `excess_tokens` and the parts, and nothing is ever cut to fit |
| `--warn-input N` | longer inputs are taken, and said on stderr with their parts |
| `--fim-prefix N`, `--fim-suffix N`, `--fim-max N` | completions with a `suffix`, an editor filling in code at its cursor: the tokens before the cursor read at most (2048; 0: all) and the lines after it (8; 0: all); `--fim-max N` the tokens written there at most, whatever the request asks (256; 0: as asked), since an editor asks for thousands and the requests of the next keystrokes wait for them. Over the budget the text before the cursor is cut at a line to three quarters of it, and the start of the window is kept across keystrokes, also when the editor slides its window, so that a keystroke reads only what it changed (see [Filling in code](../README.md#filling-in-code)) |
| `--metrics FILE`, `--progress S` | JSON lines of what every reply costs, and progress lines while a long prompt is read (see [Watching a long prompt](chat.md#watching-a-long-prompt)); `--verbose` prints progress every 30 s |
| `--no-mcp` | refuse tools of type `mcp` in `/v1/responses`: no connection leaves the server on a request's word (below) |
| `--keep N` | conversations kept computed besides the one in use (8; 0: none), so that clients taking turns, or a client's requests on the side for titles and tags, do not have their conversations read again from the start |
| `--keep-disk GIB` | disk for them below the memory, in `~/.cache/janas`, so that they outlive the server: a long system message and tools are read once (8; 0: none; never more than a quarter of the space left). Written only when a conversation leaves the memory and when the server stops |
| `--keep-memory GIB` | the memory for them; by default half of what the expert cache and the margin left to other programs leave free at start, at least 256 MiB |
| `--store DIR` | where stored completions, responses and conversations are kept, so that they outlive the server (`~/.local/share/janas/server`) |
| `--no-store` | keep them in memory only |
| `--queue N` | requests allowed to wait (16) |
| `--max-body MIB` | the largest request body (32) |
| `--connections N` | open connections at most (64) |
| `--verbose` | a line per request on the standard error |

What it answers:

| Request | |
|---|---|
| `GET /v1/models`, `GET /v1/models/{model}` | the models it runs, with a `description` of how they run here; `DELETE` answers that a model read from a file is not deleted through the API |
| `POST /v1/chat/completions` | a conversation, the reply whole or streamed (`"stream": true`, server-sent events), with tools, JSON output, log-probabilities and more than one reply (below) |
| `GET`, `POST`, `DELETE /v1/chat/completions/{id}`, `GET /v1/chat/completions`, `.../messages` | the completions a client asked to keep (`"store": true`): read, listed with their `metadata`, changed, deleted |
| `POST /v1/completions` | raw text to continue, no chat format around it; several prompts, token ids, `echo`, `logprobs`, `best_of`, `suffix` |
| `POST /infill` | llama.cpp's route for filling in code, which editors' extensions made for llama.cpp ask (llama.vscode): `input_prefix`, `prompt`, `input_suffix`, `n_predict` (0: the prompt only, read for the next request), `n_indent`, `t_max_predict_ms`, the answer as llama.cpp's (`content`, `timings`); `input_extra` is not used, and the most likely token is always taken |
| `POST /v1/responses` and the six operations under it | OpenAI's newer API: items in and out, `previous_response_id`, conversations, streamed events, `background` and `cancel`, `input_items`, `input_tokens`, `compact` |
| `/v1/conversations` and its seven operations | conversations the server keeps, and their items |
| `POST /v1/embeddings` | with `--embedding-model FILE`: the vectors of texts (below) |
| `POST /v1/moderations` | asked of the chat model (below) |
| `GET /health` | `{"status":"ok"}` |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "What is the capital of Sardinia?"}]}'
```

A client that lets you set the address of an OpenAI-compatible API should need nothing else: the base URL is `http://127.0.0.1:8080/v1`, and the key whatever you gave `--api-key` (anything, if you gave none). It has been tried here with `curl`, its own test page, the `openai` Python SDK and Open WebUI 0.11.4 (driven through its own API: streamed chat, titles, tags, follow-up suggestions, and a document indexed with Janas-LLM's embeddings and asked about), and by a user with AnythingLLM.

**Models that reason, and the limits a client sets.** Qwen3 and its successors reason before they answer, unless told not to. OpenAI's `max_completion_tokens` (and `max_output_tokens` in the Responses API) counts the reasoning too, and `janas-server` takes it so. The older `max_tokens` of a chat, which OpenAI does not take at all for the models that reason, is taken for the answer alone: a client that sends it means the answer, and counted with the reasoning a small limit was spent before a word of it was written (Open WebUI asks for a chat's title with 1000, and got none). The reasoning itself has a ceiling only when one is asked for: `reasoning_budget_tokens` (or `thinking_budget_tokens`, the names llama.cpp takes) in a request, `--reasoning-budget` for the requests that say nothing. A model still reasoning there has its reasoning closed for it, a sentence ("I have thought enough; now I give the answer.") and the closing mark written as if it had written them, and goes on with its answer, which the limits above still bound: -1 means no ceiling, 0 no reasoning at all. Without one a model that never closed its reasoning would write until the context is full; issue #21's did, its arithmetic broken by a defect since fixed.

**With Open WebUI**, start the server with `--think off`: the titles, tags and suggestions it asks for after every reply are then quick, and a model that reasons can still be asked to (`reasoning_effort`). Its built-in tools, when they are on, add about 5,000 tokens to every prompt; the server reads them once and keeps them computed, but the first reply of a conversation waits for them.

**The conversation is the client's.** OpenAI's API keeps no state: every request carries the whole conversation. `janas-server` keeps the one it computed last, and reads again only what a request adds to it: a client that sends the same messages plus a new one pays for the new one alone, and `usage.prompt_tokens_details.cached_tokens` says how much was already computed. The replies it wrote are remembered with their reasoning, so the answer a client sends back - without the reasoning, as clients do - still counts as the same turn. A conversation that differs earlier is read again from where it differs, and from the start on models with a recurrent state (Qwen3-Next, Qwen3.5 and 3.6).

**Reasoning** comes apart from the answer, in `reasoning_content`, whole or streamed. `"reasoning_effort": "none"` turns it off for a request, any other effort turns it on, and `"chat_template_kwargs": {"enable_thinking": false}` works too; `--think` decides for requests that say nothing.

**Beyond OpenAI's fields**, which clients ignore: `top_k` and `min_p` in a request, and in every reply's `usage` a `janas` object with the context used and its size and the seconds spent reading the prompt and writing the reply.

**Tools.** `tools`, `tool_choice` (`none`, `auto`, `required`, a function) and `parallel_tool_calls`, and the older `functions`. The tools are written into the system message the way the model's chat template writes them, and a call the model opens is held to a grammar of the functions and their parameters until it closes it, so its arguments are always a JSON object valid against the function's schema. The calls come back in `tool_calls`, whole in one delta of the stream, and the reply ends with `"finish_reason": "tool_calls"`; sent back with the tools' answers, they are found again among the replies the server remembers, so the next turn reads only the answers. Qwen models write calls in one of two ways, and the template says which: as JSON (Qwen3, Qwen3-Next) or as XML (Qwen3.5 and 3.6, Qwen3-Coder). **Both have been run here**: the JSON way on Qwen3-4B and Qwen3-0.6B, the XML way on Qwen3.6-35B-A3B (two calls at once, their answers sent back, 422 of 471 prompt tokens reused on the next turn) and on Qwen3-Coder-Next, which describes the tools in a way of its own (two calls at once with valid arguments, their answers used in the reply, 496 of 562 prompt tokens reused).

**MCP servers in the Responses API.** A tool of type `mcp` (`server_label`, `server_url`, `headers`, `authorization`, `allowed_tools`, `require_approval`) has the server reach that MCP server over HTTP or HTTPS, as OpenAI's does. The response begins with an `mcp_list_tools` item (once per conversation), the model sees the tools as `server_label__tool`, and a call it writes either becomes an `mcp_approval_request` that ends the response - `require_approval` is `always` by default, as at OpenAI - and runs when the next request answers it with an `mcp_approval_response`, or, where no approval is needed, runs at once as an `mcp_call` item with its output, and the model goes on in the same response (up to eight rounds). Streamed, the calls come as `response.mcp_call.*` events. `allowed_tools` and the approval filters take names or `{tool_names, read_only}`; `tool_choice` may name an MCP tool. Connectors and tunnels are not served. A request makes the server open connections to the address it names: `--no-mcp` turns that off for a server others can reach. Tried with Qwen3-4B against the test server of `tests/mcp_fake_server.c`: a call run at once, one approved on the next request, and the same streamed; and through the `openai` SDK against DeepWiki over HTTPS, as OpenAI's own examples do: its tool listed, called, and the model answering from what it returned.

**JSON output.** `response_format` `json_object` or `json_schema` (and `text.format` in the Responses API): the answer is held to a grammar token by token, so it is valid JSON when the model ends it (not when `max_tokens` cuts it). A schema is held to its types, the properties of an object in the schema's order with the required ones always there and no others, array items and bounds up to 32, string lengths up to 64, `enum`, `const`, `anyOf`, `oneOf`, `$ref` within the schema and `nullable`; `pattern`, `format` and numeric bounds are not enforced. The reasoning, where there is one, stays free.

**Log-probabilities.** `logprobs` and `top_logprobs` (up to 20) in chat, `logprobs` in completions, `top_logprobs` in responses. They are the model's own: where a grammar forced a token the model found unlikely, its log-probability says so.

**More than one reply** with `n` (and `best_of` in completions, which keeps the best by mean log-probability); `presence_penalty`, `frequency_penalty` and `logit_bias` as OpenAI defines them. The replies of `n` are written one after the other on the same prompt.

**Embeddings** come from a second, small model opened with `--embedding-model`, such as Qwen3-Embedding-0.6B: texts or token ids, one or many, `dimensions` to cut the vector shorter, floats or `base64`. Against llama.cpp on the same file the vectors of the same text agree to a cosine of 0.98 to 0.996, and the similarities between texts come out alike.

**Moderations** are asked of the chat model: at temperature 0 it answers OpenAI's thirteen categories with a JSON object held to a schema, and each score is the chance it gave `true` against `false` in that place. It is a general model following instructions, not a classifier trained for the task.

**Kept on disk.** Stored completions, responses and conversations are written, one file each, to `~/.local/share/janas/server` (or `--store DIR`), and read back when the server starts again; the oldest go past a thousand of each. The directory and its files are readable by their owner alone, since they hold what the clients said, and one server at a time uses a directory. `--no-store` keeps them in memory only, until the server stops. A response running in the background when the server is killed is found as failed at the next start. Images and sound in messages are refused with a 400 that says so.

**Every other operation of OpenAI's API** - there are 345 in the version it follows (2.3.0) - has a route that answers 501 and says why: not written yet, needs a model of another kind (sound, images, video), trains models, or belongs to the administration of OpenAI's service. The table is generated from the specification by `tools/openapi_routes`, which also counts what is done.

**One request at a time.** The model holds one sequence, so requests wait in line and run in the order they came; when the line is full the answer is 429. While a streamed request waits, the stream says where it stands in the line (`: queue N`, a comment clients skip) and the test page shows it. A client that closes the connection while a reply streams stops the reply.

**On the network.** It listens on this machine alone unless `--host` says otherwise, and then it warns if there is no key. There is no HTTPS yet: across a network you do not trust, put it behind something that adds it.

**The test page.** With `--test`, `http://127.0.0.1:8080/test` is *Janas-Chat Web*: a chat in the browser that talks to the API like any client, with the reasoning in grey, a line under each reply with the context used and the speed (last, lowest, highest and mean of the conversation), and the settings - system message, temperature, reasoning, key - behind *Settings*. It reads the marks a model writes as `janas-chat` does, and also quotes, tables, struck text and links; a link asks before it opens, because links an AI writes can be wrong or unsafe. *Markdown sample* shows a text of ours with every mark it knows, without asking the model. The page is inside the program, loads nothing from anywhere else and may talk only to this server.
