# Janas diagnostics

Two programs that tell whether a GPU can run Janas-LLM's GPU backend, and whether it computes what the CPU does, without installing Janas or downloading a model. A few megabytes; Linux x86-64 (Ubuntu 22.04, Debian 12 and later) and Windows 10 and 11 x86-64.

## vkcheck

Lists what the GPU backend needs with the answer of the GPU's driver, the decision the engine would take, and three short tests on the device: zero-copy, the int8 dot product, the float operations that keep the GPU's results the CPU's to the bit. A few seconds.

## gpu_check

Runs every kind of product the GPU takes on the GPU Janas would use - each weight type, rows that end inside a group, several vector counts and strides, the BF16 products - and compares each with the CPU's, bit for bit. Each product runs twice: all its rows on the GPU, and shared with the CPU as the engine does. A card with memory of its own gets the weights copied there, as in the engine. Under a minute on most machines.

Nothing stops at the first error. For every wrong product the report says:

- which shader computed it;
- how far its results are from the CPU's, in units in the last place: a few mean the driver rounds a float operation differently, many mean wrong values;
- the first wrong row, with both values in hexadecimal.

Then gpu_check isolates the culprits by itself:

- it runs each wrong product again with the other shaders the engine could take for it;
- it runs it again with every other subgroup size the device allows.

The last line says `ok` or `FAILED`.

Run it in a terminal from this folder, `./gpu_check` on Linux and `.\gpu_check` in PowerShell on Windows. It opens the GPU even where `JANAS_GPU=0` keeps the engine off it. Other variables:

- `GPU_CHECK_TYPES=Q6_K,Q8_0` checks those types only;
- `JANAS_GPU_DEVICE=discrete` or `integrated` picks the GPU when there are both.

## Sending the results

If either program reports a problem, paste its whole output into an issue at <https://github.com/prabanta-dev/janas/issues>, with the GPU's driver version if the output does not show it.

The SHA-256 of every file is in SHA256SUMS.
