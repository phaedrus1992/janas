# Janas for Linux x86-64

**Qwen3-Next-80B, a 48 GB model, on a laptop with 32 GB of memory: 26 tok/s writing a reply, where llama.cpp manages 3.0, and 68 tok/s reading a prompt, against 18.** Janas-LLM read and wrote faster than llama.cpp on every model compared on that laptop (Debian 13 Linux with kernel 6.12.111, Intel Core Ultra 9 185H with its integrated Arc GPU, Mesa 26.1.6, and its NPU, 32 GB of DDR5-5600 memory, an NVMe SSD), each at its best, measured on 21 October 2026:

| Model | reading 1,024 tokens: Janas-LLM | llama.cpp | writing: Janas-LLM | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-Next-80B-A3B (48.4 GB, more than the memory) | 68.4 tok/s | 17.7 tok/s | 26.4 tok/s | 3.0 tok/s |
| Qwen3-Coder-Next (48.4 GB, more than the memory) | 71.4 tok/s | 14.0 tok/s | 26.5 tok/s | 2.9 tok/s |
| Qwen3.6-35B-A3B | 136 tok/s | 81.8 tok/s | 28.0 tok/s | 12.1 tok/s |
| Qwen3-30B-A3B | 133 tok/s | 122 tok/s† | 33.3 tok/s | 15.0 tok/s† |
| Qwen3-30B-A3B, bartowski's Q4_K_M (importance matrix) | 132 tok/s | 125 tok/s | 32.3 tok/s | 15.9 tok/s |
| Qwen3-30B-A3B, unsloth's UD-Q3_K_XL (3 bits) | 120 tok/s | 114 tok/s | 35.4 tok/s | 11.3 tok/s |
| Gemma-4-26B-A4B-it | 85.0 tok/s | 78.0 tok/s | 18.9 tok/s | 12.6 tok/s |
| Qwen3.8-27B (dense, 17.6 GB) | 31.4 tok/s | 25.5 tok/s | 4.4 tok/s | 2.1 tok/s |
| Gemma-4-12B-it | 65.7 tok/s | 47.6 tok/s | 10.0 tok/s | 7.4 tok/s |
| Qwen3.5-9B | 122 tok/s | 80.3 tok/s | 14.5 tok/s | 8.7 tok/s |
| Gemma-4-E4B-it | 245 tok/s | 110 tok/s | 19.8 tok/s | 15.3 tok/s |
| Qwen3-4B | 333 tok/s | 172 tok/s | 27.9 tok/s | 17.9 tok/s |
| Qwen3-4B, bartowski's Q4_K_M (importance matrix) | 333 tok/s | 173 tok/s | 27.8 tok/s | 18.2 tok/s |
| Gemma-4-E2B-it | 508 tok/s | 150 tok/s | 37.0 tok/s | 18.9 tok/s |
| Qwen3.5-2B | 729 tok/s | 314 tok/s | 53.3 tok/s | 27.5 tok/s |
| Qwen3.5-0.8B | 1,403 tok/s | 514 tok/s | 110 tok/s | 31.1 tok/s |
| Qwen3-0.6B | 1,421 tok/s | 710 tok/s | 133 tok/s | 46.0 tok/s |

Janas-LLM's figures are the best of the configurations `janas-bench` measures: the GPU reading a prompt's blocks, or on the smaller dense models (Qwen3-4B and smaller, Gemma-4-E2B and E4B) the NPU; llama.cpp's are its best on that laptop, with its Vulkan backend on the same GPU. † llama.cpp's figures of 11 October. On another machine the figures will be other ones: `janas-bench` measures yours (below). How they were taken, and everything else: <https://github.com/prabanta-dev/janas>.

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
- An NPU is optional too: the NPU of Intel's Core Ultra reads the prompts of the smaller dense models faster (measured on the first generation's, Meteor Lake), with Intel's NPU driver (linux-npu-driver), a Level Zero loader and the user in the `render` group.

It is an experiment, developed and measured on one machine: an Intel CPU with AVX2 and AVX-VNNI. On other processors, GPUs and disks it is untested; the report of `janas-bench` is the most useful thing to send back (<https://github.com/prabanta-dev/janas/issues>).

## Licence

Janas is under the GNU General Public License, version 3 or later: `LICENSE`. The models are not part of it and keep their own licences: `MODELS.md`.
