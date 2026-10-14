# vkcheck on the M1 Max

The maintainer named `tools/vkcheck.c` as the gate for the macOS GPU path (upstream issue #20, 2026-10-13).
The gate answers whether the engine would take a GPU on this machine, and through which library.

## Machine

- Model: MacBookPro18,4
- Chip: Apple M1 Max (integrated GPU)
- macOS: Darwin 27.0.0

## Build

The build used the maintainer's command:

```sh
cc -O2 -I/opt/homebrew/include tools/vkcheck.c -o vkcheck
```

Exit 0.
The Vulkan headers come from the `vulkan-headers` Homebrew bottle (1.4.363.0).

## Run

MoltenVK 1.4.2 (the `molten-vk` Homebrew bottle) does not link `libMoltenVK.dylib` into `/opt/homebrew/lib`.
The run therefore named the library, which the tool accepts as its first argument:

```sh
./vkcheck /opt/homebrew/opt/molten-vk/lib/libMoltenVK.dylib
```

Exit 1.

## Output

```text
vkcheck: /opt/homebrew/opt/molten-vk/lib/libMoltenVK.dylib, instance Vulkan 1.4
GPU 0: Apple M1 Max (integrated), Vulkan 1.3.357
  Vulkan 1.3 or later: yes
  driver: MoltenVK 1.4.2
  buffer device addresses: yes
  integer dot product: yes
  subgroup operations in compute shaders: basic yes, ballot yes, arithmetic yes, clustered yes, shuffle yes
  subgroup size: 32, varying, not selectable
  host memory imported in place (zero-copy): yes, aligned to 16384 bytes
  a portability implementation (VK_KHR_portability_subset)
  device-local memory: 68.7 GB, of which the CPU can map 68.7 GB
  Janas-LLM would not take it: subgroup size varying and not selectable

Janas-LLM would run on the CPU alone: no GPU it takes
```

## What the gate says

- MoltenVK imports host memory in place (zero-copy), with 16384-byte alignment.
  The design's old concern about host-memory import on MoltenVK does not hold.
- The integer dot product works.
- Every subgroup operation works.
- The subgroup size is 32, and it varies.
  It is not selectable.
  The engine requires a selectable subgroup size.
  The gate therefore answers: Janas-LLM would not take this GPU through MoltenVK.

The maintainer set the rule: native Metal is weighed on these numbers if MoltenVK cannot do it.
The output above is the evidence for that call.
The decision itself belongs to the maintainer.
