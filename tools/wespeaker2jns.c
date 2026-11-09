/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * wespeaker2jns.c - WeSpeaker's ResNet34 speaker embedding model in ONNX
 * (Wespeaker/wespeaker-voxceleb-resnet34-LM, CC-BY-4.0: its batch norms
 * folded into the convolutions) into a JNS file: the 36 convolutions in
 * the graph's order, each weight in half precision laid out as the
 * engine's tiled products take it (a column per patch element: row, column
 * of the kernel, input channel; a row per output channel), its bias, its
 * kernel and stride; the pooled statistics' projection to 256 and the
 * embeddings' mean subtracted at the end.
 *
 *   wespeaker2jns voxceleb_resnet34_LM.onnx out.jns
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/quant.h"
#include "llm/jns_write.h"
#include "onnx_pb.h"

static void die(const char *msg, const char *arg)
{
    fprintf(stderr, "wespeaker2jns: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(1);
}

struct tensor {
    char name[128];
    uint64_t dims[4];
    int n_dims;
    const float *data;
    uint64_t count;
};

static struct tensor T[256];
static int n_t;

static void read_tensor(struct pb m)
{
    if (n_t == (int)(sizeof T / sizeof *T))
        die("too many tensors", NULL);
    struct tensor *t = &T[n_t];
    memset(t, 0, sizeof *t);
    int num, wt, type = 0, r;
    uint64_t v;
    struct pb s = {0, 0}, raw = {0, 0};
    while ((r = field(&m, &num, &wt, &v, &s)) == 1) {
        if (num == 1 && wt == 0 && t->n_dims < 4)
            t->dims[t->n_dims++] = v;
        else if (num == 1 && wt == 2)
            while (s.p < s.end && t->n_dims < 4 && varint(&s, &v) == 0)
                t->dims[t->n_dims++] = v;
        else if (num == 2 && wt == 0)
            type = (int)v;
        else if (num == 8 && wt == 2) {
            size_t k = (size_t)(s.end - s.p);
            if (k >= sizeof t->name)
                k = sizeof t->name - 1;
            memcpy(t->name, s.p, k);
        } else if (num == 9 && wt == 2)
            raw = s;
    }
    if (r < 0)
        die("a tensor that is not protobuf", NULL);
    uint64_t count = 1;
    for (int i = 0; i < t->n_dims; i++)
        count *= t->dims[i];
    if (type != 1 || !raw.p || (uint64_t)(raw.end - raw.p) != 4 * count)
        return;
    t->data = (const float *)raw.p; /* little-endian floats, as here */
    t->count = count;
    n_t++;
}

static const struct tensor *find(const struct pb *name)
{
    for (int i = 0; i < n_t; i++)
        if (str_eq(name, T[i].name))
            return &T[i];
    return NULL;
}

static const struct tensor *find_name(const char *name)
{
    for (int i = 0; i < n_t; i++)
        if (!strcmp(name, T[i].name))
            return &T[i];
    return NULL;
}

struct conv {
    struct pb w, b;
    int kernel, stride, pad;
};

static struct conv C[64];
static int n_c;
static struct pb gemm_w, gemm_b;

static void read_node(struct pb m)
{
    int num, wt;
    uint64_t v;
    struct pb s = {0, 0}, ins[4], op = {0, 0};
    int n_ins = 0, kernel = 0, stride = 1, pad = 0;
    while (field(&m, &num, &wt, &v, &s) == 1) {
        if (num == 1 && wt == 2 && n_ins < 4)
            ins[n_ins++] = s;
        else if (num == 4 && wt == 2)
            op = s;
        else if (num == 5 && wt == 2) {
            int an, aw;
            uint64_t av;
            struct pb as, name = {0, 0}, ints = {0, 0};
            int64_t first = -1;
            while (field(&s, &an, &aw, &av, &as) == 1) {
                if (an == 1 && aw == 2)
                    name = as;
                else if (an == 8 && aw == 0 && first < 0)
                    first = (int64_t)av;
                else if (an == 8 && aw == 2)
                    ints = as;
            }
            if (ints.p && first < 0 && varint(&ints, &av) == 0)
                first = (int64_t)av;
            if (name.p && str_eq(&name, "kernel_shape"))
                kernel = (int)first;
            else if (name.p && str_eq(&name, "strides"))
                stride = (int)first;
            else if (name.p && str_eq(&name, "pads"))
                pad = (int)first;
        }
    }
    if (op.p && str_eq(&op, "Conv") && n_ins == 3) {
        if (n_c == (int)(sizeof C / sizeof *C))
            die("too many convolutions", NULL);
        C[n_c++] = (struct conv){ins[1], ins[2], kernel, stride, pad};
    } else if (op.p && str_eq(&op, "Gemm") && n_ins == 3) {
        gemm_w = ins[1];
        gemm_b = ins[2];
    }
}

static struct janas_jw W;

static void put_f32(const char *name, const float *d, uint64_t n)
{
    float *c = malloc(4 * n);
    if (!c)
        die("out of memory", NULL);
    memcpy(c, d, 4 * n);
    uint64_t dims[1] = {n};
    if (janas_jw_tensor(&W, name, 0, 1, dims, c, 4 * n) != 0)
        die("out of memory", NULL);
}

/* rows x cols in half precision, transposed: wt[col][row] */
static void put_f16t(const char *name, const float *w, uint64_t rows,
                     uint64_t cols, int kernel, uint64_t cin)
{
    uint16_t *h = malloc(2 * rows * cols);
    if (!h)
        die("out of memory", NULL);
    for (uint64_t o = 0; o < rows; o++)
        for (uint64_t ci = 0; ci < cin; ci++)
            for (int ky = 0; ky < kernel; ky++)
                for (int kx = 0; kx < kernel; kx++) {
                    /* ONNX: [out][in][ky][kx]; ours: column (ky, kx, in) */
                    uint64_t col = ((uint64_t)ky * kernel + kx) * cin + ci;
                    h[col * rows + o] = janas_fp32_to_fp16(
                        w[((o * cin + ci) * kernel + ky) * kernel + kx]);
                }
    uint64_t dims[2] = {rows, cols};
    if (janas_jw_tensor(&W, name, 1, 2, dims, h, 2 * rows * cols) != 0)
        die("out of memory", NULL);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: wespeaker2jns model.onnx out.jns\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f)
        die("cannot open", argv[1]);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n > 0 ? (size_t)n : 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n)
        die("cannot read", argv[1]);
    fclose(f);
    struct pb m = {buf, buf + n}, s, g = {0, 0};
    int num, wt, r;
    uint64_t v;
    while ((r = field(&m, &num, &wt, &v, &s)) == 1)
        if (num == 7 && wt == 2)
            g = s;
    if (r < 0 || !g.p)
        die("not an ONNX file", argv[1]);
    /* initializers first: the nodes name them */
    struct pb gg = g;
    while (field(&gg, &num, &wt, &v, &s) == 1)
        if (num == 5 && wt == 2)
            read_tensor(s);
    gg = g;
    while (field(&gg, &num, &wt, &v, &s) == 1)
        if (num == 1 && wt == 2)
            read_node(s);
    if (n_c != 36 || !gemm_w.p)
        die("not WeSpeaker's ResNet34 (36 convolutions and a projection)",
            argv[1]);
    int32_t kernels[36], strides[36];
    for (int i = 0; i < n_c; i++) {
        const struct tensor *w = find(&C[i].w), *b = find(&C[i].b);
        if (!w || !b || w->n_dims != 4 || w->dims[2] != (uint64_t)C[i].kernel ||
            w->dims[3] != (uint64_t)C[i].kernel ||
            C[i].pad != (C[i].kernel - 1) / 2)
            die("a convolution's weights are not as expected", NULL);
        uint64_t cout = w->dims[0], cin = w->dims[1];
        char name[32];
        snprintf(name, sizeof name, "spk.conv%d.w", i);
        put_f16t(name, w->data, cout, cin * C[i].kernel * C[i].kernel,
                 C[i].kernel, cin);
        snprintf(name, sizeof name, "spk.conv%d.b", i);
        put_f32(name, b->data, cout);
        kernels[i] = C[i].kernel;
        strides[i] = C[i].stride;
    }
    const struct tensor *pw = find(&gemm_w), *pb = find(&gemm_b);
    const struct tensor *mean = find_name("mean_vec");
    if (!pw || !pb || pw->n_dims != 2 || pw->dims[0] != 256 ||
        pw->dims[1] != 5120 || pb->count != 256 || !mean || mean->count != 256)
        die("the projection is not 5120 to 256", NULL);
    put_f16t("spk.proj.w", pw->data, 256, 5120, 1, 5120);
    put_f32("spk.proj.b", pb->data, 256);
    put_f32("spk.mean", mean->data, 256);
    janas_jw_ints(&W, "spk.kernels", 36, kernels);
    janas_jw_ints(&W, "spk.strides", 36, strides);
    janas_jw_u32(&W, "spk.mels", 80);
    janas_jw_u32(&W, "spk.dim", 256);
    janas_jw_str(&W, "general.name", "WeSpeaker ResNet34-LM (VoxCeleb2)");
    janas_jw_str(&W, "general.license", "CC-BY-4.0");
    char err[300];
    uint64_t bytes;
    if (janas_jw_write(&W, argv[2], "wespeaker_resnet34", 256, &bytes, err,
                       sizeof err) != 0)
        die(err, NULL);
    printf("wespeaker2jns: %s, %d tensors, %llu bytes\n", argv[2], W.n,
           (unsigned long long)bytes);
    janas_jw_free(&W);
    free(buf);
    return 0;
}
