# Tools

| | |
|---|---|
| `tools/gguf2jns` | GGUF → the `.jns` format the engine reads |
| `tools/hf2gguf` | a Llama 3 checkpoint from Hugging Face (safetensors), with a PEFT LoRA adapter merged into it if given, → a bf16 GGUF that llama.cpp's `llama-quantize` quantizes and `gguf2jns` converts |
| `tools/hf2jns_mtp` | the multi-token prediction block, from the original checkpoint: its shards, or just the block's tensors from Hugging Face (`hf:<owner>/<repo>`); or from a GGUF file of the block alone (unsloth's `mtp-*.gguf` of Qwen3.8-Flash-Next), copied as it is |
| `tools/jns_planes` | rewrites a model with the experts' down matrix in bit planes, so a machine short of memory can read part of it |
| `jns_check` | checks a model file, with `--verify` every expert's checksum |
| `tools/openapi_routes` | `janas-server`'s route table, from OpenAI's specification |
| `tools/gen_unicode` | the tokenizer's Unicode tables, from the Unicode Character Database |
| `tools/gen_geo` | the services' tables of airports, cities and seas, from OurAirports, GeoNames and Natural Earth |
| `mcp_fake_server` | an MCP server over stdio in either era of the protocol, for trying a client without anybody else's server |
| `llm_eval` | compares the engine's logits against a reference dump |
| `vocab_check` | the output head restricted to a vocabulary (`janas_llm_chat_vocab`) against the whole head: the same logits to the bit |
| `bench_kernels`, `bench_gemm`, `bench_attn`, `bench_long`, ... | the pieces measured on their own |

They are built into `bin/x86_64-linux/` along with everything else.

## janas-nsm

Explications in the Natural Semantic Metalanguage (NSM: a word's meaning paraphrased with 65 semantic primes - someone, something, do, happen, want, know, good, before... - and a few simpler words) written by a model trained for it, such as DeepNSM ([arXiv 2505.11764](https://arxiv.org/abs/2505.11764)), and counted as the paper counts them.

```
janas-nsm <model.jns> <word> ["example sentence" ...]   one word
janas-nsm <model.jns> --batch words.jsonl --out out.jsonl   "word" and "examples" a line
janas-nsm --score explications.jsonl                     "word" and "explication" a line, no model
```

Each explication gets its words, the share of primes, of function words and of the other words ("molecules"), the paper's legality, 10 (primes - molecules) / words, and whether it is circular (a form of the word itself in it); a batch ends with the means and their standard errors, and the speed. `--lexicon paper` (the default) counts with the list the paper's scores were computed with, single words; `--lexicon full` with Goddard and Wierzbicka's 65 primes, their inflected forms and phrases ("a long time", "don't want"). On the paper's 1,341 published records the counts give its figures (circularity on all 1,341, primes and molecules on 1,337).

`--strict` lets the model write only the primes, the function words, a list of molecules (`--molecules FILE`, one word a line; a default list of body parts, physical qualities, nature, people, everyday actions and things) and punctuation: some 3,800 tokens of Llama 3's 128,257, whose rows alone the output head computes (`janas_llm_chat_vocab` in the library, `Chat.Vocab` in FreeBASIC). DeepNSM-1B Q8_0 on the laptop's CPU, the paper's 149 benchmark words, five seeds each way: writing 57.9 to 72.1 tokens/s, legality 5.33 to 5.48 (+0.15, 95% interval +0.08 to +0.22). The explications tell their word from its neighbours less well, though: judged by another model (Gemma 4 E4B, which ranks the word among four WordNet neighbours from the explication), 81.9% to 75.8% of the words ranked first, -6.0 points (-10.1 to -2.7; one seed, one judge), lost on concrete things and body parts (leg, chair, candle, elbow). The default writes freely.

DeepNSM's weights, code and data are CC BY-NC-SA 4.0: Janas does not ship them; `tools/hf2gguf` converts the model from its Hugging Face repository and Llama-3.2-1B's (Meta's licence), for research.
