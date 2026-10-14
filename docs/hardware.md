# Your hardware

## Measuring your machine

```sh
bin/x86_64-linux/janas-bench qwen3-4b
```

It takes the model as `janas-chat` does: a name of the models Janas knows (downloaded, after asking, if it is not there yet), a `.jns` file, or nothing, and then the list to choose from; the model's prediction file comes with it as in the chat (`--no-mtp`: without). Close what you can before running it: it measures the machine you give it, and a browser with thirty tabs is part of the machine. It tries the configurations the machine offers — how many threads, which cores, with and without the GPU — on real passes of the model, keeps the fastest, writes them into `~/.cache/janas` so the next start begins from them, and leaves a report in the current directory. With the fast prompt the GPU reads a prompt's blocks in every configuration, so a row without "+ GPU" is not the CPU alone; the report says so under its table. On the development laptop that report says, among other things (10 October 2026, ten models), that a single token and a prompt go fastest on the performance cores' 12 threads, while a draft check - a few tokens at once - goes fastest on all the usable cores: the kinds of pass want different threads, which is why each is tuned on its own.

## Helping it run on more hardware

This is where help is worth most. The engine decides everything from what it measures, but it has only ever measured **one** machine, so the decisions are tuned to one shape of CPU, one disk and one GPU.

**The easy way: one command.** From a fresh clone, on Linux, with a C compiler and `curl` and nothing else:

```sh
git clone https://github.com/prabanta-dev/janas && cd janas
./tools/janas-try.sh
```

It builds Janas and runs its tests, downloads a model from Hugging Face (checking its SHA-256), converts it (checking the result against the fingerprints of [MODELS.md](../MODELS.md)), checks that the model's answer at temperature 0 is the very text every other machine gets, measures the speed with `janas-bench`, tries `janas-server`, and writes a report with no host name, user name or paths in it. It asks before every long step and shows the report before anything is sent. **Run it with the machine to itself** (close the browser, builds, other models): before measuring it checks how idle the CPU is and, if it is busy, offers to wait; the report says how idle it was, in general terms only (a load average and a percentage, never which programs ran). If you agree, it opens the issue with `gh`, or gives you a link to a filled [test report form](https://github.com/prabanta-dev/janas/issues/new?template=04-test-report.yml). Four levels: `quick` (Qwen3-4B, 2.5 GB to download), `medium` (Qwen3.6-35B-A3B, 22 GB), `full` (Qwen3-Next-80B-A3B with its MTP block, about 52 GB) and `gemma` (Gemma-4-E4B with its assistant, 5.1 GB, runs in about 3 GB of memory); it offers those your machine can hold, and deletes at the end only what it downloaded, if you say so. `--help` lists the options.

**What to send, by hand.** Run the benchmark and attach the report it writes (a `janas-bench-<date>.txt` in the current directory):

```sh
bin/x86_64-linux/janas-bench <model>
```

`<model>` is a name from the list `janas-bench` shows when given none (`qwen3-4b`), or a `.jns` file. The report already carries the CPU's name, the core layout, the GPU if there is one, and the speed of every configuration tried. Add:

- the **distribution** and kernel (`uname -a`), and the **RAM** (`free -g`);
- the **disk** the model sits on (NVMe? SATA? over USB?);
- the **model** you used and its file size;
- what you saw: numbers that look wrong, a reply that made no sense, a crash, a machine that went unresponsive — with what you were doing at the time.

The [hardware report form](https://github.com/prabanta-dev/janas/issues/new?template=01-hardware-report.yml) asks for exactly that, in that order. There are two more forms, for [something that went wrong](https://github.com/prabanta-dev/janas/issues/new?template=02-something-went-wrong.yml) and for [a model that will not convert](https://github.com/prabanta-dev/janas/issues/new?template=03-a-model-will-not-work.yml).

**Whether a GPU can run the GPU backend**, and why not: `bin/x86_64-linux/vkcheck` lists what the backend needs with the answer of the GPU's driver, the decision the engine would take, and three short tests on the device (zero-copy, the int8 dot product, the float operations that keep the GPU's results the CPU's to the bit). It builds alone where Janas does not build yet, macOS with MoltenVK included: `cc -O2 tools/vkcheck.c -o vkcheck -ldl -lm` (on macOS without `-ldl -lm`, with `-I/opt/homebrew/include`). Its output is welcome in a hardware report.

**What is most wanted, in order.**

1. **CPUs without AVX-VNNI**, and **AMD** — the arithmetic has a path for them and it is the least exercised of all.
2. **Discrete GPUs** (NVIDIA, AMD, Intel Arc). The GPU support is correct where it was checked - bit for bit on a reply's tokens, and on a prompt as close to the references as the CPU - and it pays where it was measured: on the development laptop's integrated GPU, with the fast prompt, Qwen3-4B reads a prompt 2.6 times faster than without the GPU (85 to 220 tokens/s; before the fast prompt, 83 to 107), and on an RTX 5080 (issue #15) Qwen3-4B read 13% faster and, with the GPU on single tokens too, replied 11-16% faster. From the release of 10 October to that of 13 October a discrete card's results stayed in its own memory, where the CPU never read them (issue #17, corrupted output on a Quadro RTX 5000); fixed since, and checked here on the path such a card takes, not yet on one. Every other card is unknown, and the fast prompt has not been seen working on a discrete one yet (`JANAS_GPU_FAST_PROMPT=0` turns it off if it misbehaves). **Careful:** a GPU driver reset takes the desktop with it. Start with `janas-bench`, not with a long chat.
3. **Machines with 16 GB or less.** The engine is supposed to trade quality for memory on its own — fewer bits per weight, light experts not in memory left out — and the thresholds for that were measured on 32 GB.
4. **Slower disks.** Everything assumes an NVMe SSD; on a SATA disk the engine should still work and simply wait more, but nobody has watched it do so.
5. **Windows on other machines.** The Windows build has run on one laptop, the development one (Windows 11, build 26300), where its tests pass and it is as fast as on Linux; `janas-bench`'s report from any other is welcome.
6. **Other models** of the supported families. A file that will not convert, or converts and then answers nonsense, is a useful bug.

**What testers found so far.** On a Radeon 780M (Ryzen 7 255, issue #16) the GPU read Qwen3-4B's prompt 5.2 times faster than the CPU alone (477 against 92 tokens/s, the second with `JANAS_GPU_FAST_PROMPT=0`), and subgroups of 32 (`JANAS_GPU_SG=32`) instead of the 64 the engine chose cost 14% (405 against 470 tokens/s): the default is right there. Its "+ GPU" rows wrote a reply at 10.5 tokens/s against 17.3 on the CPU alone: the GPU was kept awake for the single tokens of a reply, which an integrated GPU does not take, and awake it only took the power the cores need. Since 14 October 2026 it is kept awake only for the passes it takes; on the development laptop's Arc the "+ GPU" rows wrote 2-5% slower than the CPU alone before, and as fast after (Qwen3-4B, two alternated runs each). On a Quadro RTX 5000 (issue #17) the defect above.

**What not to bother with yet:** macOS (not supported), and models outside the Qwen3, Qwen3.5, Qwen3.6, Qwen3-Next and Gemma 4 families (they are refused with a message).

Patches are welcome too, but a measurement from a machine nobody here can buy is worth more than most patches.
