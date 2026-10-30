/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * silero2jns.c - Silero VAD v5/v6's ONNX model (onnx-community/silero-vad,
 * MIT) into a JNS file: the weights of its 16 kHz branch in single
 * precision as the model holds them - the STFT's basis, the four
 * convolutions of the encoder, the LSTM's input and recurrent matrices and
 * biases (ONNX's gate order: input, output, forget, cell), the output
 * convolution. The ONNX file is read with a small protobuf reader of its
 * own: the graph of the "If" on the rate, its then branch, the LSTM node's
 * inputs naming its weights.
 *
 *   silero2jns model.onnx out.jns
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/jns_write.h"
#include "onnx_pb.h"

static void die(const char *msg, const char *arg)
{
    fprintf(stderr, "silero2jns: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(1);
}

/* ---- the tensors of a graph ---- */

struct tensor {
    char name[256];
    uint64_t dims[4];
    int n_dims;
    float *data;
    uint64_t count;
};

static struct tensor T[512];
static int n_t;

static void read_tensor(struct pb m)
{
    if (n_t == (int)(sizeof T / sizeof *T))
        die("too many tensors", NULL);
    struct tensor *t = &T[n_t];
    memset(t, 0, sizeof *t);
    int num, wt, type = 0;
    uint64_t v;
    struct pb s = {0, 0}, raw = {0, 0}, fl = {0, 0};
    int r;
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
        else if (num == 4 && wt == 2)
            fl = s;
    }
    if (r < 0)
        die("a tensor that is not protobuf", NULL);
    if (type != 1) /* FLOAT only: the weights */
        return;
    uint64_t count = 1;
    for (int i = 0; i < t->n_dims; i++)
        count *= t->dims[i];
    struct pb src = raw.p ? raw : fl;
    if (!src.p || (uint64_t)(src.end - src.p) != 4 * count)
        return;
    t->data = malloc(4 * count);
    if (!t->data)
        die("out of memory", NULL);
    memcpy(t->data, src.p, 4 * count); /* little-endian, as here */
    t->count = count;
    n_t++;
}

static struct pb lstm_in[8];
static int n_lstm_in, found_lstm;

static void read_graph(struct pb g, int keep);

static void read_node(struct pb m, int keep)
{
    int num, wt;
    uint64_t v;
    struct pb s, ins[8];
    int n_ins = 0, is_lstm = 0;
    while (field(&m, &num, &wt, &v, &s) == 1) {
        if (num == 1 && wt == 2 && n_ins < 8)
            ins[n_ins++] = s;
        else if (num == 4 && wt == 2)
            is_lstm = str_eq(&s, "LSTM");
        else if (num == 5 && wt == 2) {
            /* an attribute: a then_branch holds the 16 kHz model */
            int an, aw;
            uint64_t av;
            struct pb as, name = {0, 0}, graph = {0, 0};
            while (field(&s, &an, &aw, &av, &as) == 1) {
                if (an == 1 && aw == 2)
                    name = as;
                else if (an == 6 && aw == 2)
                    graph = as;
            }
            if (graph.p)
                read_graph(graph,
                           keep || (name.p && str_eq(&name, "then_branch")));
        }
    }
    if (is_lstm && keep && !found_lstm) {
        memcpy(lstm_in, ins, sizeof(struct pb) * (size_t)n_ins);
        n_lstm_in = n_ins;
        found_lstm = 1;
    }
}

static void read_graph(struct pb g, int keep)
{
    int num, wt;
    uint64_t v;
    struct pb s = {0, 0};
    while (field(&g, &num, &wt, &v, &s) == 1) {
        if (num == 1 && wt == 2)
            read_node(s, keep);
        else if (num == 5 && wt == 2 && keep)
            read_tensor(s);
    }
}

static struct tensor *find(const char *suffix, uint64_t count)
{
    size_t n = strlen(suffix);
    for (int i = 0; i < n_t; i++) {
        size_t k = strlen(T[i].name);
        if (k >= n && !strcmp(T[i].name + k - n, suffix) &&
            (!count || T[i].count == count))
            return &T[i];
    }
    return NULL;
}

static struct tensor *by_input(int i)
{
    if (i >= n_lstm_in)
        return NULL;
    for (int k = 0; k < n_t; k++)
        if (str_eq(&lstm_in[i], T[k].name))
            return &T[k];
    return NULL;
}

static struct janas_jw W;

static void put(const char *name, const struct tensor *t)
{
    if (!t)
        die("a tensor is missing", name);
    uint64_t dims[1] = {t->count};
    float *d = malloc(4 * t->count);
    if (!d)
        die("out of memory", NULL);
    memcpy(d, t->data, 4 * t->count);
    if (janas_jw_tensor(&W, name, 0, 1, dims, d, 4 * t->count) != 0)
        die("out of memory", NULL);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: silero2jns model.onnx out.jns\n");
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
    struct pb m = {buf, buf + n};
    int num, wt, r;
    uint64_t v;
    struct pb s = {0, 0};
    while ((r = field(&m, &num, &wt, &v, &s)) == 1)
        if (num == 7 && wt == 2) /* ModelProto.graph */
            read_graph(s, 0);
    if (r < 0)
        die("not an ONNX file", argv[1]);
    if (!found_lstm || n_lstm_in < 4)
        die("no LSTM in a then branch: not Silero VAD v5", argv[1]);
    /* the 16 kHz branch's weights by their names, the LSTM's by its node */
    put("vad.stft", find("stft.forward_basis_buffer", 258 * 256));
    put("vad.enc0.w", find("encoder.0.reparam_conv.weight", 128 * 129 * 3));
    put("vad.enc0.b", find("encoder.0.reparam_conv.bias", 128));
    put("vad.enc1.w", find("encoder.1.reparam_conv.weight", 64 * 128 * 3));
    put("vad.enc1.b", find("encoder.1.reparam_conv.bias", 64));
    put("vad.enc2.w", find("encoder.2.reparam_conv.weight", 64 * 64 * 3));
    put("vad.enc2.b", find("encoder.2.reparam_conv.bias", 64));
    put("vad.enc3.w", find("encoder.3.reparam_conv.weight", 128 * 64 * 3));
    put("vad.enc3.b", find("encoder.3.reparam_conv.bias", 128));
    struct tensor *lw = by_input(1), *lr = by_input(2), *lb = by_input(3);
    if (!lw || !lr || !lb || lw->count != 512 * 128 || lr->count != 512 * 128 ||
        lb->count != 1024)
        die("the LSTM's weights are not 512 x 128", NULL);
    put("vad.lstm.w", lw);
    put("vad.lstm.r", lr);
    put("vad.lstm.b", lb);
    put("vad.out.w", find("decoder.decoder.2.weight", 128));
    put("vad.out.b", find("decoder.decoder.2.bias", 1));
    janas_jw_u32(&W, "vad.window", 512);
    janas_jw_u32(&W, "vad.context", 64);
    janas_jw_u32(&W, "vad.rate", 16000);
    janas_jw_str(&W, "general.name", "Silero VAD");
    janas_jw_str(&W, "general.license", "MIT");
    char err[300];
    uint64_t bytes;
    if (janas_jw_write(&W, argv[2], "silero_vad", 128, &bytes, err,
                       sizeof err) != 0)
        die(err, NULL);
    printf("silero2jns: %s, %d tensors, %llu bytes\n", argv[2], W.n,
           (unsigned long long)bytes);
    janas_jw_free(&W);
    for (int i = 0; i < n_t; i++)
        free(T[i].data);
    free(buf);
    return 0;
}
