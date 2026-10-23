# Tools

| | |
|---|---|
| `tools/gguf2jns` | GGUF → the `.jns` format the engine reads |
| `tools/hf2jns_mtp` | the multi-token prediction block, from the original checkpoint: its shards, or just the block's tensors from Hugging Face (`hf:<owner>/<repo>`); or from a GGUF file of the block alone (unsloth's `mtp-*.gguf` of Qwen3.8-Flash-Next), copied as it is |
| `tools/jns_planes` | rewrites a model with the experts' down matrix in bit planes, so a machine short of memory can read part of it |
| `jns_check` | checks a model file, with `--verify` every expert's checksum |
| `tools/openapi_routes` | `janas-server`'s route table, from OpenAI's specification |
| `tools/gen_unicode` | the tokenizer's Unicode tables, from the Unicode Character Database |
| `tools/gen_geo` | the services' tables of airports, cities and seas, from OurAirports, GeoNames and Natural Earth |
| `mcp_fake_server` | an MCP server over stdio in either era of the protocol, for trying a client without anybody else's server |
| `llm_eval` | compares the engine's logits against a reference dump |
| `bench_kernels`, `bench_gemm`, `bench_attn`, `bench_long`, ... | the pieces measured on their own |

They are built into `bin/x86_64-linux/` along with everything else.
