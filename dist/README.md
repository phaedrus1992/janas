# Janas for Linux x86-64

**Qwen3-Next-80B, a 48 GB model, on a laptop with 32 GB of memory: 37 tok/s with its own draft block, 24 without, where llama.cpp manages 2.8 on the same laptop.** Reading a prompt of 1,024 tokens: 62 tok/s against 16. Janas-LLM was faster than llama.cpp on every model compared on that laptop (Debian 13 Linux with kernel 6.12.107, Intel Core Ultra 9 185H, its integrated Arc GPU with Mesa 26.1.6, 32 GB of DDR5-5600 memory, an NVMe SSD), measured on 9-11 October 2026:

| Model | reading 1,024 tokens: Janas-LLM | llama.cpp | writing: Janas-LLM | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-Next-80B-A3B (48.4 GB, more than the memory) | 62 tok/s | 16 tok/s | 24 tok/s | 2.8 tok/s |
| Qwen3-Coder-Next (48.4 GB, more than the memory) | 66 tok/s | 14 tok/s | 22 tok/s | 2.2 tok/s |
| Qwen3.6-35B-A3B | 104 tok/s | 79 tok/s | 19 tok/s | 12 tok/s |
| Qwen3-30B-A3B | 135 tok/s | 122 tok/s | 32 tok/s | 15 tok/s |
| Gemma-4-26B-A4B-it | 84 tok/s | 76 tok/s | 18 tok/s | 12 tok/s |
| Qwen3.8-27B (dense, 17.6 GB) | 26.4 tok/s | 25.6 tok/s | 4.4 tok/s | 2.1 tok/s |
| Gemma-4-12B-it | 65 tok/s | 46 tok/s | 9.9 tok/s | 7.1 tok/s |
| Qwen3.5-9B | 111 tok/s | 78 tok/s | 14 tok/s | 8.7 tok/s |
| Gemma-4-E4B-it | 122 tok/s | 109 tok/s | 20 tok/s | 15 tok/s |
| Qwen3-4B | 193 tok/s | 164 tok/s | 27 tok/s | 18 tok/s |
| Gemma-4-E2B-it | 191 tok/s | 150 tok/s | 38 tok/s | 19 tok/s |
| Qwen3.5-2B | 441 tok/s | 323 tok/s | 52 tok/s | 28 tok/s |
| Qwen3.5-0.8B | 799 tok/s | 519 tok/s | 109 tok/s | 32 tok/s |
| Qwen3-0.6B | 939 tok/s | 670 tok/s | 130 tok/s | 45 tok/s |

The 3-bit Qwen3-30B of the catalog is the one exception: it reads a prompt slower than llama.cpp (100 tok/s against 112), its Q3_K weights not on the GPU's fast prompt yet. On another machine the figures will be other ones: `janas-bench` measures yours (below). How they were taken, and everything else: <https://github.com/prabanta-dev/janas>.

Janas-LLM, the inference engine of Janas, runs large language models on ordinary computers, with or without a GPU, mixtures of experts larger than the memory among them: the weights stay on disk, the experts each token needs are read from the SSD and the ones that keep coming back stay in memory. It is written in C and needs nothing but the system's C library; a GPU is used through its Vulkan driver when there is one.

## Start

```sh
./janas-get list                 # the models it knows, their size, the memory they want
./janas-get qwen3-4b             # downloads it, checked, into ~/.local/share/janas/models
./janas-chat qwen3-4b            # and talk to it
./janas-bench qwen3-4b           # how fast this machine runs it: a report to send back
```

`./janas-chat` with no model shows the list and downloads the one you choose, after asking. The models come already converted from Hugging Face ([prabanta-dev](https://huggingface.co/prabanta-dev)), each file checked against the fingerprints of `MODELS.md`.

## What is here

| | |
|---|---|
| `janas-chat` | a chat in the terminal: line editor, history, the reasoning shown apart, a context that slides instead of ending, the tools of MCP servers and of Janas's own services |
| `janas-server` | the model served over HTTP with an API that follows OpenAI's: chat completions (tools, JSON schemas), completions, responses, embeddings, and llama.cpp's route for filling in code in editors |
| `janas-get` | downloads a model of the catalog, checked; `--from-gguf` converts the original GGUF here instead |
| `janas-bench` | measures what this machine does with a model, every configuration the engine can use, and fills the profile it starts from next time |
| `janas-mcp` | the model as an MCP server for other clients (Claude Code, an editor): a question to it, the embedding of a text |
| `gguf2jns` | converts a GGUF file (Q4_K_M and the other types `MODELS.md` lists) to the `.jns` file Janas reads |
| `jns_planes` | cuts the experts of a converted mixture into bit planes, so that fewer bits can be read when the memory is short |
| `hf2jns_mtp` | makes a model's multi-token prediction block from its original checkpoint |
| `jns_check` | what a `.jns` file says about itself: architecture, sizes, types, licence |
| `vkcheck` | whether this computer's GPU can run Janas's GPU backend, and why not: what the backend needs with the driver's answer, then three short tests on the GPU |
| `services/` | Janas's services, which bring data of the world to a model: flights, weather, maps, prices, earthquakes, Wikipedia, GitHub, git, this computer. `janas-chat` starts them by itself; any other MCP client can use them. See `services/README.md` |
| `lib/`, `include/` | `libjanas_llm` and `libjanas_mcp` with their C headers and FreeBASIC bindings, for programs of your own; the programs above do not need them |

## Needs

- Linux on x86-64, with the C library of Ubuntu 22.04, Debian 12 or later.
- An NVMe SSD for the models larger than the memory.
- A GPU is optional: an integrated one with a Vulkan driver reads the prompts faster (measured on Intel's Arc; others untested).

It is an experiment, developed and measured on one machine: an Intel CPU with AVX2 and AVX-VNNI. On other processors, GPUs and disks it is untested; the report of `janas-bench` is the most useful thing to send back (<https://github.com/prabanta-dev/janas/issues>).

## Licence

Janas is under the GNU General Public License, version 3 or later: `LICENSE`. The models are not part of it and keep their own licences: `MODELS.md`.
