/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_ir.c - the OpenVINO IR writer of npu_ir.h.
 */
#include "npu_ir.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put(struct npu_ir *g, const char *fmt, ...)
{
    if (g->failed)
        return;
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(g->xml + g->len, g->cap - g->len, fmt, ap);
        va_end(ap);
        if (n < 0) {
            g->failed = 1;
            return;
        }
        if ((size_t)n < g->cap - g->len) {
            g->len += (size_t)n;
            return;
        }
        size_t cap = (g->cap + (size_t)n) * 2;
        char *x = realloc(g->xml, cap);
        if (!x) {
            g->failed = 1;
            return;
        }
        g->xml = x, g->cap = cap;
    }
}

static const char *precision(const char *et)
{
    return !strcmp(et, "f32")   ? "FP32"
           : !strcmp(et, "f16") ? "FP16"
           : !strcmp(et, "i8")  ? "I8"
                                : "I64";
}

static void port(struct npu_ir *g, int id, int l, const char *names)
{
    put(g, "<port id=\"%d\" precision=\"%s\"", id, precision(g->et[l]));
    if (names)
        put(g, " names=\"%s\"", names);
    put(g, ">");
    for (int d = 0; d < g->nd[l]; d++)
        put(g, "<dim>%d</dim>", g->dims[l][d]);
    put(g, "</port>");
}

/* a layer of output type et and dims, inputs a[ni] */
static int layer(struct npu_ir *g, const char *type, const char *name,
                 const char *data, const char *et, int nd, const int *dims,
                 int ni, const int *a, const char *names)
{
    if (g->failed || g->nl >= NPU_IR_LAYERS || nd > NPU_IR_DIMS ||
        g->ne + ni > NPU_IR_LAYERS * 2) {
        g->failed = 1;
        return -1;
    }
    for (int i = 0; i < ni; i++)
        if (a[i] < 0 || a[i] >= g->nl) {
            g->failed = 1;
            return -1;
        }
    int l = g->nl++;
    g->et[l] = et, g->nd[l] = nd, g->ins[l] = ni;
    memcpy(g->dims[l], dims, (size_t)nd * sizeof(int));
    put(g, "<layer id=\"%d\" name=\"%s\" type=\"%s\" version=\"opset1\">%s", l,
        name, type, data ? data : "");
    if (ni) {
        put(g, "<input>");
        for (int i = 0; i < ni; i++) {
            port(g, i, a[i], NULL);
            g->edge[g->ne][0] = a[i], g->edge[g->ne][1] = l;
            g->edge[g->ne++][2] = i;
        }
        put(g, "</input>");
    }
    if (strcmp(type, "Result")) {
        put(g, "<output>");
        port(g, ni, l, names);
        put(g, "</output>");
    }
    put(g, "</layer>\n");
    return l;
}

static void shape(char *s, size_t len, int nd, const int *dims)
{
    size_t o = 0;
    s[0] = 0;
    for (int d = 0; d < nd && o < len; d++)
        o += (size_t)snprintf(s + o, len - o, d ? ",%d" : "%d", dims[d]);
}

void npu_ir_begin(struct npu_ir *g, const char *name)
{
    memset(g, 0, sizeof(*g));
    put(g,
        "<?xml version=\"1.0\"?>\n<net name=\"%s\" version=\"11\">\n"
        "<layers>\n",
        name);
}

int npu_ir_param(struct npu_ir *g, const char *name, const char *et, int nd,
                 const int *dims)
{
    char data[160], s[64];
    shape(s, sizeof s, nd, dims);
    snprintf(data, sizeof data, "<data shape=\"%s\" element_type=\"%s\"/>", s,
             et);
    return layer(g, "Parameter", name, data, et, nd, dims, 0, NULL, name);
}

int npu_ir_const(struct npu_ir *g, const char *name, const char *et, int nd,
                 const int *dims, const void *p, size_t bytes)
{
    if (g->failed)
        return -1;
    uint8_t *b = realloc(g->bin, g->nbin + bytes);
    if (!b) {
        g->failed = 1;
        return -1;
    }
    g->bin = b;
    memcpy(g->bin + g->nbin, p, bytes);
    char data[200], s[64];
    shape(s, sizeof s, nd, dims);
    snprintf(data, sizeof data,
             "<data element_type=\"%s\" shape=\"%s\" offset=\"%zu\" "
             "size=\"%zu\"/>",
             et, s, g->nbin, bytes);
    g->nbin += bytes;
    return layer(g, "Const", name, data, et, nd, dims, 0, NULL, NULL);
}

int npu_ir_convert(struct npu_ir *g, const char *name, int a, const char *et)
{
    if (a < 0 || a >= g->nl)
        return g->failed = 1, -1;
    char data[64];
    snprintf(data, sizeof data, "<data destination_type=\"%s\"/>", et);
    return layer(g, "Convert", name, data, et, g->nd[a], g->dims[a], 1, &a,
                 NULL);
}

int npu_ir_multiply(struct npu_ir *g, const char *name, int a, int b)
{
    if (a < 0 || a >= g->nl)
        return g->failed = 1, -1;
    int in[2] = {a, b};
    return layer(g, "Multiply", name, "<data auto_broadcast=\"numpy\"/>",
                 g->et[a], g->nd[a], g->dims[a], 2, in, NULL);
}

int npu_ir_add(struct npu_ir *g, const char *name, int a, int b)
{
    if (a < 0 || a >= g->nl)
        return g->failed = 1, -1;
    int in[2] = {a, b};
    return layer(g, "Add", name, "<data auto_broadcast=\"numpy\"/>", g->et[a],
                 g->nd[a], g->dims[a], 2, in, NULL);
}

int npu_ir_binary(struct npu_ir *g, const char *name, const char *type, int a,
                  int b)
{
    if (a < 0 || a >= g->nl || b < 0 || b >= g->nl)
        return g->failed = 1, -1;
    int nd = g->nd[a] > g->nd[b] ? g->nd[a] : g->nd[b], dims[NPU_IR_DIMS];
    for (int i = 0; i < nd; i++) { /* aligned to the right, as numpy */
        int ia = i - (nd - g->nd[a]), ib = i - (nd - g->nd[b]);
        int da = ia >= 0 ? g->dims[a][ia] : 1,
            db = ib >= 0 ? g->dims[b][ib] : 1;
        dims[i] = da > db ? da : db;
    }
    int in[2] = {a, b};
    return layer(g, type, name, "<data auto_broadcast=\"numpy\"/>", g->et[a],
                 nd, dims, 2, in, NULL);
}

int npu_ir_reduce_mean(struct npu_ir *g, const char *name, int a, int axis)
{
    return npu_ir_reduce(g, name, "ReduceMean", a, axis);
}

int npu_ir_reduce(struct npu_ir *g, const char *name, const char *type, int a,
                  int axis)
{
    if (a < 0 || a >= g->nl || axis < 0 || axis >= g->nd[a])
        return g->failed = 1, -1;
    int64_t ax = axis;
    int one = 1, dims[NPU_IR_DIMS];
    char cn[64];
    snprintf(cn, sizeof cn, "%s_axes", name);
    int c = npu_ir_const(g, cn, "i64", 1, &one, &ax, sizeof ax);
    memcpy(dims, g->dims[a], sizeof dims);
    dims[axis] = 1;
    int in[2] = {a, c};
    return layer(g, type, name, "<data keep_dims=\"true\"/>", g->et[a],
                 g->nd[a], dims, 2, in, NULL);
}

int npu_ir_fake_quantize(struct npu_ir *g, const char *name, int a, float lo,
                         float hi, int levels)
{
    if (a < 0 || a >= g->nl)
        return g->failed = 1, -1;
    char n[4][48], data[64];
    static const char *part[4] = {"il", "ih", "ol", "oh"};
    float v[4] = {lo, hi, lo, hi};
    int in[5] = {a};
    for (int i = 0; i < 4; i++) {
        snprintf(n[i], sizeof n[i], "%s_%s", name, part[i]);
        in[i + 1] = npu_ir_scalar(g, n[i], v[i]);
    }
    snprintf(data, sizeof data,
             "<data levels=\"%d\" auto_broadcast=\"numpy\"/>", levels);
    return layer(g, "FakeQuantize", name, data, g->et[a], g->nd[a], g->dims[a],
                 5, in, NULL);
}

int npu_ir_unary(struct npu_ir *g, const char *name, const char *type, int a)
{
    if (a < 0 || a >= g->nl)
        return g->failed = 1, -1;
    return layer(g, type, name, NULL, g->et[a], g->nd[a], g->dims[a], 1, &a,
                 NULL);
}

/* FP16 by hand (round to nearest even) */
static uint16_t half(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000u, e = (b >> 23) & 0xff,
             mant = b & 0x7fffffu;
    int ee = (int)e - 127 + 15;
    if (e == 0xff || ee >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (ee <= 0)
        return (uint16_t)sign; /* the constants here are not that small */
    uint32_t v = ((uint32_t)ee << 10) | (mant >> 13), rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (v & 1)))
        v++;
    return (uint16_t)(sign | v);
}

int npu_ir_scalar(struct npu_ir *g, const char *name, float v)
{
    uint16_t h = half(v);
    int one = 1;
    return npu_ir_const(g, name, "f16", 1, &one, &h, 2);
}

int npu_ir_matmul_tb(struct npu_ir *g, const char *name, int a, int b,
                     const char *names)
{
    if (a < 0 || a >= g->nl || b < 0 || b >= g->nl || g->nd[a] != 2 ||
        g->nd[b] != 2 || g->dims[a][1] != g->dims[b][1])
        return g->failed = 1, -1;
    int in[2] = {a, b}, dims[2] = {g->dims[a][0], g->dims[b][0]};
    return layer(g, "MatMul", name,
                 "<data transpose_a=\"false\" transpose_b=\"true\"/>", g->et[a],
                 2, dims, 2, in, names);
}

void npu_ir_result(struct npu_ir *g, int a)
{
    if (a < 0 || a >= g->nl) {
        g->failed = 1;
        return;
    }
    char name[32];
    snprintf(name, sizeof name, "result_%d", g->nres++);
    layer(g, "Result", name, NULL, g->et[a], g->nd[a], g->dims[a], 1, &a, NULL);
}

int npu_ir_weights_i8(struct npu_ir *g, const char *name, const int8_t *q,
                      const uint16_t *s, int rows, int cols)
{
    char nq[48], ns[48], nc[48];
    int wd[2] = {rows, cols}, sd[2] = {rows, 1};
    snprintf(nq, sizeof nq, "%s_q", name);
    snprintf(ns, sizeof ns, "%s_s", name);
    snprintf(nc, sizeof nc, "%s_c", name);
    int lq = npu_ir_const(g, nq, "i8", 2, wd, q, (size_t)rows * cols);
    int ls = npu_ir_const(g, ns, "f16", 2, sd, s, (size_t)rows * 2);
    return npu_ir_multiply(g, name, npu_ir_convert(g, nc, lq, "f16"), ls);
}

int npu_ir_rms_norm(struct npu_ir *g, const char *name, int x, const float *w,
                    int d, float eps)
{
    char n[6][48];
    static const char *part[6] = {"sq", "ms", "eps", "rt", "xn", "w"};
    for (int i = 0; i < 6; i++)
        snprintf(n[i], sizeof n[i], "%s_%s", name, part[i]);
    int one = 1;
    int ms = npu_ir_reduce_mean(g, n[1],
                                npu_ir_binary(g, n[0], "Multiply", x, x), 1);
    int le = npu_ir_const(g, n[2], "f32", 1, &one, &eps, sizeof eps);
    int rt =
        npu_ir_unary(g, n[3], "Sqrt", npu_ir_binary(g, n[2], "Add", ms, le));
    int xn = npu_ir_binary(g, n[4], "Divide", x, rt);
    int lw = npu_ir_const(g, n[5], "f32", 1, &d, w, sizeof(float) * (size_t)d);
    return npu_ir_binary(g, name, "Multiply", xn, lw);
}

int npu_ir_ffn(struct npu_ir *g, const char *name, int x, int gate, int up,
               int down, int act)
{
    char n[18][48];
    static const char *part[18] = {"g",   "u",   "sig",  "silu", "g2",   "g3",
                                   "c3",  "gc3", "gin",  "k",    "gk",   "gt",
                                   "one", "g1t", "half", "h",    "gelu", "gh"};
    for (int i = 0; i < 18; i++)
        snprintf(n[i], sizeof n[i], "%s_%s", name, part[i]);
    int lg = npu_ir_matmul_tb(g, n[0], x, gate, NULL);
    int lu = npu_ir_matmul_tb(g, n[1], x, up, NULL);
    int la;
    if (act == 0) /* g sigmoid(g) */
        la = npu_ir_multiply(g, n[3], lg, npu_ir_unary(g, n[2], "Sigmoid", lg));
    else { /* 0.5 g (1 + tanh(sqrt(2/pi) (g + 0.044715 g^3))) */
        int g3 = npu_ir_multiply(g, n[5], npu_ir_multiply(g, n[4], lg, lg), lg);
        int in = npu_ir_add(
            g, n[8], lg,
            npu_ir_multiply(g, n[7], g3, npu_ir_scalar(g, n[6], 0.044715f)));
        int t = npu_ir_unary(
            g, n[11], "Tanh",
            npu_ir_multiply(g, n[10], in,
                            npu_ir_scalar(g, n[9], 0.7978845608f)));
        int one = npu_ir_add(g, n[13], t, npu_ir_scalar(g, n[12], 1.0f));
        la = npu_ir_multiply(
            g, n[16],
            npu_ir_multiply(g, n[17], lg, npu_ir_scalar(g, n[14], 0.5f)), one);
    }
    return npu_ir_matmul_tb(g, name, npu_ir_multiply(g, n[15], la, lu), down,
                            NULL);
}

int npu_ir_end(struct npu_ir *g)
{
    put(g, "</layers>\n<edges>\n");
    for (int e = 0; e < g->ne; e++) {
        int f = g->edge[e][0];
        put(g,
            "<edge from-layer=\"%d\" from-port=\"%d\" to-layer=\"%d\" "
            "to-port=\"%d\"/>\n",
            f, g->ins[f], g->edge[e][1], g->edge[e][2]);
    }
    put(g, "</edges>\n</net>\n");
    return g->failed ? -1 : 0;
}

void npu_ir_free(struct npu_ir *g)
{
    free(g->xml);
    free(g->bin);
    memset(g, 0, sizeof(*g));
}

uint8_t *npu_ir_model(const struct npu_ir *g, uint16_t major, uint16_t minor,
                      size_t *bytes)
{
    uint64_t sx = g->len, sb = g->nbin;
    uint32_t parts = 2;
    size_t n = 4 + sizeof parts + 8 + sx + 8 + sb, at = 0;
    uint8_t *b = malloc(n);
    if (!b)
        return NULL;
    memcpy(b + at, &major, 2), at += 2;
    memcpy(b + at, &minor, 2), at += 2;
    memcpy(b + at, &parts, sizeof parts), at += sizeof parts;
    memcpy(b + at, &sx, 8), at += 8;
    memcpy(b + at, g->xml, sx), at += sx;
    memcpy(b + at, &sb, 8), at += 8;
    if (sb)
        memcpy(b + at, g->bin, sb);
    *bytes = n;
    return b;
}
