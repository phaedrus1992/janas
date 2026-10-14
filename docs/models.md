# Getting a model

**The short way: `janas-get`.** One command downloads a model already converted, from the project's own repositories on Hugging Face ([prabanta-dev](https://huggingface.co/prabanta-dev)), with its prediction file (an MTP block, Gemma 4's assistant, or for Qwen3-4B the small Qwen3-0.6B that guesses its tokens), which the chat then finds by itself, and checks the SHA-256 of every file against [MODELS.md](../MODELS.md). The models go to `~/.local/share/janas/models`, or wherever `JANAS_MODELS` or `--dir` says:

```sh
bin/x86_64-linux/janas-get list                  # the models it knows
bin/x86_64-linux/janas-get gemma-4-e4b           # into ~/.local/share/janas/models
bin/x86_64-linux/janas-chat ~/.local/share/janas/models/gemma-4-e4b-it-q4km.jns
```

A download stopped halfway resumes where it stopped, and a file already there is checked rather than fetched again. `--from-gguf` takes the other way: the original GGUF the catalog names, downloaded and converted on your machine to the very same fingerprints (the GGUF goes once converted, `--keep` keeps it); `janas-get` takes it by itself where a converted file is not on Hugging Face. Any other file: `janas-get hf:<owner>/<repo>/<file.jns|file.gguf>`, checked against the SHA-256 Hugging Face lists for it. The memory column of `janas-get list` is the smallest machine each model is meant for. The rest of this section is the same by hand.

Janas-LLM reads its own format, `.jns`, converted from a **Q4_K_M** GGUF (Q3_K, IQ3_S, IQ4_NL, IQ4_XS, Q4_0, Q4_1 and Q5_1 convert too, and so do unsloth's "UD" files made of them and the "MXFP4_MOE" files, whose experts are MXFP4; IQ2 and IQ1 types, not yet). Keep both in `models/` next to the sources — the tools take a path, so anywhere works, but that is where these examples put them, and where the project keeps its own.

| Model | GGUF converted here | GGUF | `.jns` | Memory it likes |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B-Instruct** — most of this engine was measured on it | [Qwen](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3.6-35B-A3B** — the easiest to start with | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) | 22.3 GB | 22.3 GB | 16-32 GB |
| **Qwen3-Coder-Next** | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3-Coder-30B-A3B-Instruct** — for code completion in the editor ([docs/code-completion.md](code-completion.md)) | [unsloth](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18.6 GB | 18.6 GB | 32 GB, to keep it whole in memory |
| **Qwen3-30B-A3B** | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B with an importance matrix** (`qwen3-30b-a3b-imatrix`) — bartowski's Q4_K_M: the nearest to Qwen's own Q8_0 of the three here (400 tokens: code 388, Italian 367, against 376 and 335 for the Q4_K_M above), 8-10% slower held in memory: it keeps more matrices in Q6_K | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B in three bits** (`qwen3-30b-a3b-q3`) — unsloth's dynamic UD-Q3_K_XL: against Qwen's own Q8_0 it agrees more often than the Q4_K_M above (400 tokens: code 386 against 376, Italian 360 against 335), with 27% fewer bytes of experts; with the expert cache held at 2 GiB on this 32 GB machine, 19.8 tokens/s against 13.6; held whole in memory, as fast (32.0-32.2 tokens/s against 32.1-32.2) | [unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) | 13.8 GB | 13.8 GB | 8-16 GB |
| **Qwen3.5-9B** — dense; its prediction block from the checkpoint ([MODELS.md](../MODELS.md#the-fingerprints)) | [unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5.7 GB | 5.7 GB | 16 GB |
| **Qwen3.5-2B** — dense, small and quick; a prediction block too | [unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | 1.3 GB | 1.3 GB | 8 GB |
| **Qwen3-4B** — dense, and the smallest here that answers well | [Qwen](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-4B with an importance matrix** (`qwen3-4b-imatrix`) — bartowski's Q4_K_M, the same weight types and size as Qwen's: against Qwen's own Q8_0 (400 tokens) code 381 against 364, Italian 351 against 329, at the same speed | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-0.6B** — not to talk to: to draft for a dense one (`--draft`) | [unsloth](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) | 0.40 GB | 0.40 GB | with the model above |
| **Qwen3-Embedding-0.6B** — not to talk to: for embeddings (`janas-server --embedding-model`); published in Q8_0 and converted as it is | [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | 0.64 GB | 0.64 GB | beside the chat model |

Qwen3 has other dense models - 1.7B, 8B, 14B, 32B - which declare the same architecture as the 4B and so should convert and run the same way. **None of them has been tried here**, and until one is, that is a reading of the file and not a claim. (Qwen3-30B-A3B is not one of them: the number is close but it is a mixture of experts, with an architecture of its own.)

Any other Q4_K_M of the same models converts and runs just as well, including the "dynamic" ones that give different tensors different types: the engine reads Q4_K, Q5_K, Q6_K and Q8_0, and refuses a file carrying anything else rather than guessing. The four above are the files the numbers in this repository were measured on, and the ones [MODELS.md](../MODELS.md) gives fingerprints for, so that a converted file can be checked against the one measured here without downloading anything.

All four models are published under the Apache 2.0 licence by Alibaba Cloud, and quantized into GGUF by them or by the people above. Their licence is theirs, not this project's: [MODELS.md](../MODELS.md) says what that means if you pass a converted file on to somebody else.

**Download.** Nothing needs installing — one file, resumable:

```sh
mkdir -p models/gguf && cd models/gguf
curl -L -C - -O \
  https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
cd ../..
```

With `pip install huggingface_hub` you can instead fetch a whole quantization, which is handier when it is split into several files, as Qwen3-Coder-Next is:

```sh
huggingface-cli download Qwen/Qwen3-Coder-Next-GGUF \
    --include "Qwen3-Coder-Next-Q4_K_M/*" --local-dir models/gguf
```

**Convert**, with the converter the build made (nothing else to install). With a model split into several files, give the first one and the rest is found:

```sh
bin/x86_64-linux/gguf2jns models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    models/qwen3.6-35b-a3b-flat.jns
```

It says what it is doing, and refuses a model it does not know with a clear message rather than half a conversion.

**Cut the experts into bit planes.** Worth it on any machine that cannot hold all the experts in memory, which is the case this engine is built for: the down matrix is stored as three planes of two bits, so the engine can read four bits of it, or two, instead of six, and decide which when the model is opened. Nothing is lost — all three planes give the weights the file came with, bit for bit. This is the form the models here are kept in:

```sh
bin/x86_64-linux/jns_planes models/qwen3.6-35b-a3b-flat.jns \
    models/qwen3.6-35b-a3b.jns
rm models/qwen3.6-35b-a3b-flat.jns
```

Afterwards:

```sh
bin/x86_64-linux/jns_check models/qwen3.6-35b-a3b.jns            # what is inside
bin/x86_64-linux/jns_check models/qwen3.6-35b-a3b.jns --verify   # and the checksums
```

Both steps are deterministic, so the file can be checked against the fingerprints in [MODELS.md](../MODELS.md).

Room needed: three copies of the model at the worst moment — the GGUF, the converted file and the one with the planes — so about 145 GB for the largest. The flat file goes as soon as the planes are written, and the GGUF can go too, though keeping it saves the download the day a new version of the converter is worth running.

**The multi-token prediction block** (Qwen3-Next and the dense Qwen3.5; optional, faster replies). Its own model carries it, GGUF files leave it out, so it comes from the original checkpoint. `hf2jns_mtp` can read just the block's tensors from Hugging Face, with range requests, instead of the shards that hold them:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next.jns \
    hf:Qwen/Qwen3-Next-80B-A3B-Instruct models/qwen3-next-mtp.jns
```

For the dense Qwen3.5-9B that is 487 MB instead of 14 GB of shards, in under a minute. It also takes the shards themselves, when you have them on disk (`hf2jns_mtp <model.jns> <shard>... <out.jns>`). Qwen3.6-35B-A3B needs none of this: its GGUF keeps the block.

**Gemma 4** comes from unsloth's GGUF files, converted in one step, and so does its assistant (the `mtp-gemma-4-*.gguf` file beside the model, under 0.5 GB), which serves as the multi-token prediction file:

```sh
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    --include "gemma-4-E4B-it-Q4_K_M.gguf" "mtp-gemma-4-E4B-it.gguf" \
    --local-dir models/gguf
bin/x86_64-linux/gguf2jns models/gguf/gemma-4-E4B-it-Q4_K_M.gguf \
    models/gemma-4-e4b.jns
bin/x86_64-linux/gguf2jns models/gguf/mtp-gemma-4-E4B-it.gguf \
    models/gemma-4-e4b-mtp.jns
```

The same for E2B, the 12B (`gemma-4-12b-it-Q4_K_M.gguf`) and the 26B (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`); the fingerprints are in [MODELS.md](../MODELS.md#the-fingerprints). E2B's per-layer table, half of its file, is read from the file a token at a time and never loaded: E2B ran here in 1.5 GB of memory, E4B in 3.1 GB.

**And chat:**

```sh
bin/x86_64-linux/janas-chat models/qwen3-next.jns --stats
bin/x86_64-linux/janas-chat models/gemma-4-e4b.jns --stats
```

The chat finds the multi-token prediction file by itself: of the `.jns` files beside the model, the one whose header fits it (an MTP block of the same architecture and width, or Gemma 4's assistant for a model that wide and with its vocabulary), and says which it took. `--mtp <file>` names one; `--no-mtp` goes without. `janas-server` does the same, and programs can ask for it with `janas_llm_find_mtp()`.

The first start reads the model's resident weights (a couple of GB) and then fills the expert cache in the background, while you read and type. The second reply of a session is the one that shows the machine's real speed.

[INSTALL.md](../INSTALL.md) has the details, including the other distributions and what to do when the GPU is not found.
