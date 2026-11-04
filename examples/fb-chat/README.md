# A chat in FreeBASIC on Janas-LLM

The smallest chat on Janas-LLM, written twice:

- `chat.bas`, procedure by procedure, with `include/janas/llm.bi`: the C library's functions as they are, for FreeBASIC and for BASIC MODERN, the dialect Prabanta implements.
- `chat_oop.bas`, with the classes of `include/janas/janas_oop.bi`: a `Janas.Model`, its `Janas.Chat`, and a handler that extends `Janas.Printer` to print the reply as it is written and its counters at the end. Nothing is closed by hand: the objects close what they hold when the program ends.

Both read the user's lines from a terminal, or from a pipe.

## Building and running

Janas built first (`./build.sh` in the repository's root), which makes the shared libraries in `bin/x86_64-linux/`; then, from the root, with FreeBASIC 1.10:

```sh
fbc examples/fb-chat/chat_oop.bas -i include -p bin/x86_64-linux -x janas-chat-oop
LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-oop models/jns/qwen3-4b-q4km.jns
echo "Ciao" | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-oop models/jns/qwen3-4b-q4km.jns
```

and the same for `chat.bas`. The classes are described in `docs/library.md`, "From FreeBASIC and BASIC MODERN".
