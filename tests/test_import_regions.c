/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_import_regions.c - the regions in which the GPU imports an expert
 * cache's arena (model_load.c janas_m_import_next): each starts on a 2 MB
 * boundary, overlapping the one before, and must still be at most the 4 GiB
 * the GPU takes from its first byte, end on a slot, and leave no gap. From
 * 13 to 14 October 2026 the overlap was not counted, and Qwen3.6-35B-A3B's
 * MXFP4 experts (slots of 1,671,168 bytes) got a second region of 4 GiB and
 * 1.9 MB, which the GPU refused. No GPU needed.
 */
#include <stdio.h>

#include "llm/model_impl.h"

static int failures;

static void check(uint64_t unit, uint64_t bytes, int huge)
{
    const uint64_t mb2 = (uint64_t)2 << 20;
    uint64_t at = 0;
    int n = 0;
    while (at < bytes) {
        uint64_t from, end = janas_m_import_next(at, bytes, unit, huge, &from);
        int bad = from > at || end <= at || end > bytes ||
                  end - from > JANAS_M_REGION_MAX ||
                  (huge && from % mb2 != 0) || (!huge && from != at) ||
                  (end != bytes && (end - at) % unit != 0);
        if (bad) {
            printf("FAIL: unit %llu, %llu bytes, huge %d: region %d "
                   "[%llu, %llu) from %llu\n",
                   (unsigned long long)unit, (unsigned long long)bytes, huge, n,
                   (unsigned long long)at, (unsigned long long)end,
                   (unsigned long long)from);
            failures++;
            return;
        }
        at = end;
        n++;
    }
}

int main(void)
{
    /* the slots of Qwen3.6-35B-A3B MXFP4_MOE, of its Q4_K_M at six bits,
       of a page, of 2 MB and around it, of a dense layer's feed-forward */
    static const uint64_t units[] = {1671168,   2039808, 4096,
                                     2097152,   2101248, 2093056,
                                     157286400, 1179648, 3014656};
    for (size_t u = 0; u < sizeof(units) / sizeof(units[0]); u++)
        for (uint64_t slots = 1; slots <= 40000; slots = slots * 3 + 7)
            for (int huge = 0; huge < 2; huge++)
                check(units[u], slots * units[u], huge);
    /* the arena of the 35B's MXFP4 experts, every slot of its 40 layers */
    check(1671168, (uint64_t)40 * 256 * 1671168, 1);
    printf("test_import_regions: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
