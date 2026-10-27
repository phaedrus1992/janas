# Reports from other machines

What testers measured with `janas-bench` on their own machines, as they reported it in the issues of this repository. Every figure is theirs, from the version of Janas they ran at the time (the date says which): the engine has changed since, so rows of different dates are not a comparison between machines. Prompt and reply are tokens per second; "with MTP" is a chat reply with the drafts of the model's prediction block, greedy / sampled where the report gave both. Rows up to 26 September give the best configuration of each column; from 27 September, the configuration the engine itself chose, as a chat runs.

How to add a row: [Helping it run on more hardware](hardware.md#helping-it-run-on-more-hardware).

| Issue | Date | CPU | System | GPU | RAM | Model | Prompt | Reply | With MTP | Answer |
|---|---|---|---|---|---|---|---|---|---|---|
| [#1](https://github.com/prabanta-dev/janas/issues/1) | 20 Sep | Intel Core i7-1365U | Linux | not used | 30.7 GiB | Qwen3.6-35B-A3B | 23.1 | 10.8 | - | - |
| [#2](https://github.com/prabanta-dev/janas/issues/2) | 22 Sep | AMD Ryzen AI 9 HX 370 | Linux | not used | 23.1 GiB | Qwen3.6-35B-A3B | 44.8 | 16.0 | 24.6 | - |
| [#8](https://github.com/prabanta-dev/janas/issues/8) | 25 Sep | Intel Core Ultra 9 185H | Linux | Arc (integrated) | 30.8 GiB | Qwen3-4B | 89.7 | 26.1 | - | ok |
| [#8](https://github.com/prabanta-dev/janas/issues/8#issuecomment-5844758197) | 26 Sep | Intel Core i7-1365U | Linux | none | 30.7 GiB | Qwen3-4B | 40.3 | 10.8 | - | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3-4B | 341.5 | 21.1 | - | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3.6-35B-A3B | 317.4 | 22.8 | 49.5 | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3-Next-80B-A3B | 143.4 | 16.7 | 20.7 | - |
| [#12](https://github.com/prabanta-dev/janas/issues/12) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3-4B | 366.4 | 21.5 | - | ok |
| [#13](https://github.com/prabanta-dev/janas/issues/13) | 27 Sep | Intel Core i9-14900HX | Linux | RTX 4070 Laptop | 62.5 GiB | Qwen3-Next-80B-A3B | 122.0 | 25.5 | 33.2 | ok |
| [#13](https://github.com/prabanta-dev/janas/issues/13) | 27 Sep | Intel Core i9-14900HX | Linux | not used | 62.5 GiB | Qwen3-235B-A22B (142 GB) | 8.2 | 3.0 | - | - |
| [#14](https://github.com/prabanta-dev/janas/issues/14) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3.6-35B-A3B | 338.9 | 22.1 | 43.3 / 44.0 | ok |
| [#15](https://github.com/prabanta-dev/janas/issues/15) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3.6-35B-A3B | 374.6 | 23.0 | 45.5 / 45.5 | ok |
| [#16](https://github.com/prabanta-dev/janas/issues/16) | 13 Oct | AMD Ryzen 7 255 | Linux | Radeon 780M (integrated) | 31.2 GiB | Qwen3-4B | 477.4 | 20.5 | - | - |
| [#16](https://github.com/prabanta-dev/janas/issues/16) | 13 Oct | AMD Ryzen 7 255 | Linux | Radeon 780M (integrated) | 31.2 GiB | Qwen3-Next-80B-A3B | 61.6 | 16.2 | 23.7 / 25.8 | - |
| [#23](https://github.com/prabanta-dev/janas/issues/23) | 13 Oct | AMD Ryzen 9 5900HX | Linux | not built | 30.8 GiB | Qwen3-Next-80B-A3B | 28.2 | 9.5 | 16.6 / 20.7 | see note |
| [#37](https://github.com/prabanta-dev/janas/issues/37) | 17 Oct | Intel Core i5-1135G7 | Linux | Iris Xe (integrated) | 15.4 GiB | Qwen3-4B | 180.9 | 16.0 | - | ok |
| [#39](https://github.com/prabanta-dev/janas/issues/39) | 17 Oct | Intel Core i5-1135G7 | Linux | Iris Xe (integrated) | 15.4 GiB | Qwen3-Next-80B-A3B | 15.7 | 7.3 | 11.0 / 11.2 | ok |
| [#40](https://github.com/prabanta-dev/janas/issues/40) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3-4B | 30.9 | 25.0 | - | - |
| [#41](https://github.com/prabanta-dev/janas/issues/41) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3.5-9B | 23.7 | 15.2 | 21.0 / 21.2 | - |
| [#42](https://github.com/prabanta-dev/janas/issues/42) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3-Next-80B-A3B | 26.3 | 13.7 | 23.5 / 27.2 | - |

## Notes

- **Answer**: "ok" where `janas-try` found the model's answer at temperature 0 to be the very text every other machine gets, "-" where the report did not check it.
- **#13**: Qwen3-235B-A22B, 142 GB, on 62.5 GiB of RAM, with an expert cache of 38.7 GiB: 86.9% of the experts found in memory, the rest read from the NVMe disk.
- **#15**: with `JANAS_GPU_MIN_N=1` the RTX 5080 also took a reply's single tokens and Qwen3-4B replied at 25.9 tok/s instead of 22.4; since then the engine tries this on any discrete GPU and keeps it where it measures faster.
- **#16**: the integrated GPU read the prompt 5.2 times faster than the CPU alone (477 against 92 tok/s with `JANAS_GPU_FAST_PROMPT=0`).
- **#23**: the answer differed on every machine, not only on this one: since 1 October `janas-chat` wrote the catalog of its services into the system message, and `janas-try` expected the answer from before. Fixed on 14 October.
- **#39**: a 48 GB model on 15.4 GiB of RAM, an expert cache of 7.0 GiB.
- **#40-#42**: the prompt reads far slower than on the other machines; being looked into with the tester.
