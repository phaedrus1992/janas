/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * spv_fma.c - a shader's SPIR-V with every fma() decorated NoContraction,
 * written as the C array build.sh includes (glslangValidator --vn's form).
 *
 * Usage: spv_fma <in.spv> <array name> <out.h>
 *
 * GLSL.std.450's Fma may be done as a multiply and an add, two roundings,
 * unless it is decorated NoContraction; glslang decorates a precise
 * variable's multiplies and adds but not its fma() calls, and the Arc's
 * driver did split some of them: the experts' activation (expert_act.comp)
 * came out a bit off the CPU's in a few hundredths of its values, enough
 * to move a block's scale in one expert in eighty (24 Oct 2026). The
 * shaders write fma() where the CPU fuses, so every one of them is meant
 * as one operation.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    OP_EXT_INST_IMPORT = 11,
    OP_EXT_INST = 12,
    OP_DECORATE = 71,
    OP_TYPE_FIRST = 19, /* OpTypeVoid .. OpTypeForwardPointer: after the */
    OP_TYPE_LAST = 39,  /* annotations, so the new ones go before them */
    GLSL_FMA = 50,
    DEC_NO_CONTRACTION = 42,
};

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: spv_fma <in.spv> <array name> <out.h>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = bytes > 0 ? (size_t)bytes / 4 : 0;
    uint32_t *w = malloc((n ? n : 1) * 4);
    if (!w || fread(w, 4, n, f) != n || n < 5 || w[0] != 0x07230203u) {
        fprintf(stderr, "%s: not SPIR-V\n", argv[1]);
        return 1;
    }
    fclose(f);
    /* the GLSL.std.450 set, the Fma results, those already decorated, and
       where the types start */
    uint32_t glsl = 0, *fma = malloc(n * 4), nf = 0;
    uint8_t *done = calloc(w[3], 1); /* w[3]: the ids' bound */
    size_t types = 0;
    if (!fma || !done)
        return 1;
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (wc == 0 || i + wc > n) {
            fprintf(stderr, "%s: bad instruction at word %zu\n", argv[1], i);
            return 1;
        }
        if (op == OP_EXT_INST_IMPORT && wc >= 3 &&
            strncmp((const char *)&w[i + 2], "GLSL.std.450", 13) == 0)
            glsl = w[i + 1];
        if (op == OP_EXT_INST && wc >= 5 && w[i + 3] == glsl &&
            w[i + 4] == GLSL_FMA)
            fma[nf++] = w[i + 2];
        if (op == OP_DECORATE && wc >= 3 && w[i + 2] == DEC_NO_CONTRACTION &&
            w[i + 1] < w[3])
            done[w[i + 1]] = 1;
        if (!types && op >= OP_TYPE_FIRST && op <= OP_TYPE_LAST)
            types = i;
        i += wc;
    }
    if (nf && !types) {
        fprintf(stderr, "%s: no types\n", argv[1]);
        return 1;
    }
    FILE *o = fopen(argv[3], "w");
    if (!o) {
        perror(argv[3]);
        return 1;
    }
    fprintf(o,
            "#pragma once\n/* %s, its %u fma() NoContraction (spv_fma) "
            "*/\nconst uint32_t %s[] = {\n",
            argv[1], nf, argv[2]);
    size_t col = 0;
#define PUT(x) fprintf(o, "%s0x%08x,", col++ % 8 ? "" : "\n\t", (unsigned)(x))
    for (size_t i = 0; i < n; i++) {
        if (i == types)
            for (uint32_t k = 0; k < nf; k++)
                if (!done[fma[k]]) {
                    done[fma[k]] = 1;
                    PUT((3u << 16) | OP_DECORATE);
                    PUT(fma[k]);
                    PUT(DEC_NO_CONTRACTION);
                }
        PUT(w[i]);
    }
    fprintf(o, "\n};\n");
    if (fclose(o) != 0) {
        perror(argv[3]);
        return 1;
    }
    free(w);
    free(fma);
    free(done);
    return 0;
}
