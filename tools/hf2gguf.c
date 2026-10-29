/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf2gguf.c - a Llama 3 checkpoint from Hugging Face (safetensors), with a
 * LoRA adapter merged into it if one is given, into a GGUF file of
 * llama.cpp's "llama" architecture, its matrices in bf16. From there
 * llama.cpp's own llama-quantize makes the quantized file both engines read,
 * and gguf2jns the JNS: the same weights in Janas-LLM and in llama.cpp.
 *
 * Usage: hf2gguf <model dir> [<adapter dir>] <out.gguf>
 *
 * The model directory holds config.json, tokenizer.json and the weights in
 * one or more *.safetensors files (bf16). The adapter directory holds a PEFT
 * adapter: adapter_config.json (r, lora_alpha, target_modules) and
 * adapter_model.safetensors - lora_A and lora_B of the projections it names
 * (f32 or bf16), and the token embeddings and output head when it trained
 * them whole (DeepNSM's, grown by one token for its padding), which then
 * replace the model's; its tokenizer.json, which has that token, replaces
 * the model's too.
 *
 * What is written follows llama.cpp's convert_hf_to_gguf.py for Llama:
 *  - W + (lora_alpha / r) B A for each adapted projection, summed in float
 *    from the bf16 weight and rounded back to bf16 (to nearest, ties to
 *    even), as PEFT's merge_and_unload does on a bf16 model - the rounding
 *    of the sum is its, not of the product, which PEFT takes in bf16 too:
 *    the merged weights differ from PEFT's in the last bit at most, where
 *    the product's own rounding falls the other way;
 *  - the query and key rows permuted head by head, [2][half] to [half][2],
 *    for llama.cpp's rotation of adjacent pairs (gguf2jns turns them back
 *    for Janas-LLM's rotation of halves);
 *  - rope_freqs.weight, Llama 3's divisors of the rotary angles
 *    (rope_scaling of type llama3), in float;
 *  - the norms in f32; the output head the token embeddings' when the model
 *    ties them and the adapter has no head of its own;
 *  - the tokenizer as GPT-2's byte-level BPE with Llama 3's pre-tokenizer
 *    (tokenizer.ggml.pre "llama-bpe"): the vocabulary and the added tokens,
 *    special ones as control tokens, the merges, BOS added.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "llm/json.h"

static _Noreturn void die(const char *msg, const char *arg)
{
    fprintf(stderr, "hf2gguf: %s%s%s\n", msg, arg ? ": " : "", arg ? arg : "");
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory", NULL);
    return p;
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot open", path);
    fseeko(f, 0, SEEK_END);
    off_t n = ftello(f);
    fseeko(f, 0, SEEK_SET);
    char *s = xmalloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n)
        die("cannot read", path);
    fclose(f);
    s[n] = 0;
    *len = (size_t)n;
    return s;
}

static struct janas_json_doc *read_json(const char *dir, const char *name)
{
    char path[4096], err[256];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    size_t n;
    char *s = read_file(path, &n);
    struct janas_json_doc *d = janas_json_parse(s, n, err, sizeof(err));
    free(s);
    if (!d)
        die(err, path);
    return d;
}

static double num(const struct janas_json *o, const char *key)
{
    const struct janas_json *v = janas_json_get(o, key);
    if (!v || v->type != JANAS_JSON_NUMBER)
        die("missing number in config.json", key);
    return janas_json_num(v, 0);
}

/* ---- safetensors ---- */

enum { ST_F32, ST_BF16, ST_F16 };

struct st_tensor {
    char name[160];
    int dtype, dims;
    uint64_t shape[4];
    uint64_t begin, end; /* absolute file offsets */
    FILE *f;
};

static struct st_tensor *tensors;
static int n_tensors, cap_tensors;

static void st_scan(const char *path)
{
    FILE *f = fopen(path, "rb");
    uint64_t n;
    if (!f || fread(&n, 8, 1, f) != 1 || n > (1u << 30))
        die("cannot read", path);
    char *js = xmalloc(n + 1), err[256];
    if (fread(js, 1, n, f) != n)
        die("cannot read the header of", path);
    struct janas_json_doc *d = janas_json_parse(js, n, err, sizeof(err));
    if (!d)
        die(err, path);
    for (const struct janas_json *t = janas_json_root(d)->child; t;
         t = t->next) {
        if (!strncmp(t->key, "__metadata__", t->key_n))
            continue;
        if (n_tensors == cap_tensors) {
            cap_tensors = cap_tensors ? 2 * cap_tensors : 512;
            tensors = realloc(tensors, sizeof(*tensors) * cap_tensors);
            if (!tensors)
                die("out of memory", NULL);
        }
        struct st_tensor *s = &tensors[n_tensors++];
        memset(s, 0, sizeof(*s));
        snprintf(s->name, sizeof(s->name), "%.*s", (int)t->key_n, t->key);
        const struct janas_json *dt = janas_json_get(t, "dtype");
        s->dtype = janas_json_is(dt, "F32")    ? ST_F32
                   : janas_json_is(dt, "BF16") ? ST_BF16
                   : janas_json_is(dt, "F16")  ? ST_F16
                                               : -1;
        if (s->dtype < 0)
            die("unsupported dtype of", s->name);
        for (const struct janas_json *x = janas_json_get(t, "shape")->child;
             x && s->dims < 4; x = x->next)
            s->shape[s->dims++] = (uint64_t)janas_json_num(x, 0);
        const struct janas_json *of = janas_json_get(t, "data_offsets");
        s->begin = 8 + n + (uint64_t)janas_json_num(of->child, 0);
        s->end = 8 + n + (uint64_t)janas_json_num(of->child->next, 0);
        uint64_t count = 1;
        for (int i = 0; i < s->dims; i++)
            count *= s->shape[i];
        if (s->end - s->begin != count * (s->dtype == ST_F32 ? 4 : 2))
            die("size and shape disagree", s->name);
        s->f = f;
    }
    janas_json_free(d);
    free(js);
}

static const struct st_tensor *st_get(const char *name)
{
    for (int i = 0; i < n_tensors; i++)
        if (!strcmp(tensors[i].name, name))
            return &tensors[i];
    return NULL;
}

static uint64_t st_count(const struct st_tensor *t)
{
    uint64_t c = 1;
    for (int i = 0; i < t->dims; i++)
        c *= t->shape[i];
    return c;
}

static float bf16_to_f32(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

static uint16_t f32_to_bf16(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffff) > 0x7f800000)
        return (uint16_t)((u >> 16) | 64); /* NaN stays NaN */
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static float f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    float x;
    if (e == 0) {
        x = ldexpf((float)m, -24);
        return s ? -x : x;
    }
    uint32_t u = e == 31 ? s | 0x7f800000 | (m << 13)
                         : s | ((e + 112) << 23) | (m << 13);
    memcpy(&x, &u, 4);
    return x;
}

static float *st_load(const struct st_tensor *t)
{
    uint64_t n = st_count(t);
    size_t es = t->dtype == ST_F32 ? 4 : 2;
    uint8_t *raw = xmalloc(n * es);
    if (fseeko(t->f, (off_t)t->begin, SEEK_SET) || fread(raw, es, n, t->f) != n)
        die("cannot read", t->name);
    float *x = xmalloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        if (t->dtype == ST_F32) {
            memcpy(&x[i], raw + 4 * i, 4);
            continue;
        }
        memcpy(&h, raw + 2 * i, 2);
        x[i] = t->dtype == ST_BF16 ? bf16_to_f32(h) : f16_to_f32(h);
    }
    free(raw);
    return x;
}

/* ---- the model ---- */

struct hp {
    uint32_t d, ff, n_layer, n_head, n_kv, hd, ctx, vocab;
    float eps, rope_base;
    int tied;
    /* rope_scaling of type llama3, else factor 0 */
    double rs_factor, rs_low, rs_high, rs_orig;
};

static struct hp read_config(const char *dir)
{
    struct janas_json_doc *d = read_json(dir, "config.json");
    const struct janas_json *c = janas_json_root(d);
    if (!janas_json_is(janas_json_get(c, "model_type"), "llama"))
        die("not a Llama model (model_type)", dir);
    struct hp h = {
        .d = (uint32_t)num(c, "hidden_size"),
        .ff = (uint32_t)num(c, "intermediate_size"),
        .n_layer = (uint32_t)num(c, "num_hidden_layers"),
        .n_head = (uint32_t)num(c, "num_attention_heads"),
        .n_kv = (uint32_t)num(c, "num_key_value_heads"),
        .ctx = (uint32_t)num(c, "max_position_embeddings"),
        .vocab = (uint32_t)num(c, "vocab_size"),
        .eps = (float)num(c, "rms_norm_eps"),
        .rope_base = (float)num(c, "rope_theta"),
    };
    const struct janas_json *v = janas_json_get(c, "head_dim");
    h.hd = v ? (uint32_t)janas_json_num(v, 0) : h.d / h.n_head;
    v = janas_json_get(c, "tie_word_embeddings");
    h.tied = v && v->type == JANAS_JSON_TRUE;
    const struct janas_json *rs = janas_json_get(c, "rope_scaling");
    if (rs && rs->type == JANAS_JSON_OBJECT) {
        if (!janas_json_is(janas_json_get(rs, "rope_type"), "llama3"))
            die("rope_scaling of a type other than llama3", dir);
        h.rs_factor = num(rs, "factor");
        h.rs_low = num(rs, "low_freq_factor");
        h.rs_high = num(rs, "high_freq_factor");
        h.rs_orig = num(rs, "original_max_position_embeddings");
    }
    janas_json_free(d);
    return h;
}

/* Llama 3's divisors of the angles, as convert_hf_to_gguf.py computes them
   (its frequencies in float32). */
static float *rope_factors(const struct hp *h)
{
    uint32_t n = h->hd / 2;
    float *f = xmalloc(n * sizeof(float));
    double low_wl = h->rs_orig / h->rs_low, high_wl = h->rs_orig / h->rs_high;
    for (uint32_t i = 0; i < n; i++) {
        float e = (float)(2 * i) / (float)h->hd;
        float freq = 1.0f / powf(h->rope_base, e);
        double wl = 2 * M_PI / (double)freq;
        if (wl < high_wl)
            f[i] = 1;
        else if (wl > low_wl)
            f[i] = (float)h->rs_factor;
        else {
            double s = (h->rs_orig / wl - h->rs_low) / (h->rs_high - h->rs_low);
            f[i] = (float)(1 / ((1 - s) / h->rs_factor + s));
        }
    }
    return f;
}

/* ---- the adapter ---- */

static int have_lora;
static float lora_scale;
static char lora_prefix[] = "base_model.model.";

/* W (rows x cols, float) += scale B A, B rows x r, A r x cols. */
static void merge(float *w, const char *hf_name, uint64_t rows, uint64_t cols)
{
    char na[256], nb[256], base[200];
    snprintf(base, sizeof(base), "%.*s", (int)(strlen(hf_name) - 7),
             hf_name); /* without ".weight" */
    snprintf(na, sizeof(na), "%s%s.lora_A.weight", lora_prefix, base);
    snprintf(nb, sizeof(nb), "%s%s.lora_B.weight", lora_prefix, base);
    const struct st_tensor *ta = st_get(na), *tb = st_get(nb);
    if (!ta && !tb)
        return;
    if (!ta || !tb || ta->dims != 2 || tb->dims != 2 || ta->shape[1] != cols ||
        tb->shape[0] != rows || ta->shape[0] != tb->shape[1])
        die("lora_A and lora_B do not fit", base);
    uint64_t r = ta->shape[0];
    float *a = st_load(ta), *b = st_load(tb);
    for (uint64_t i = 0; i < rows; i++) {
        float *wi = w + i * cols;
        for (uint64_t k = 0; k < r; k++) {
            float s = lora_scale * b[i * r + k];
            const float *ak = a + k * cols;
            for (uint64_t j = 0; j < cols; j++)
                wi[j] += s * ak[j];
        }
    }
    free(a);
    free(b);
}

/* ---- GGUF ---- */

enum { G_U32 = 4, G_I32 = 5, G_F32 = 6, G_BOOL = 7, G_STR = 8, G_ARR = 9 };
enum { GT_F32 = 0, GT_BF16 = 30 };
#define GGUF_ALIGN 32

static FILE *out;

static void w8(const void *p, size_t n)
{
    if (fwrite(p, 1, n, out) != n)
        die("write failed", NULL);
}

static void wstrn(const char *s, uint64_t n)
{
    w8(&n, 8);
    w8(s, n);
}

static void wstr(const char *s)
{
    wstrn(s, strlen(s));
}

static void kv_head(const char *k, uint32_t type)
{
    wstr(k);
    w8(&type, 4);
}

static void kv_u32(const char *k, uint32_t v)
{
    kv_head(k, G_U32);
    w8(&v, 4);
}

static void kv_f32(const char *k, float v)
{
    kv_head(k, G_F32);
    w8(&v, 4);
}

static void kv_bool(const char *k, int v)
{
    uint8_t b = (uint8_t)(v != 0);
    kv_head(k, G_BOOL);
    w8(&b, 1);
}

static void kv_str(const char *k, const char *v)
{
    kv_head(k, G_STR);
    wstr(v);
}

static void kv_arr(const char *k, uint32_t type, uint64_t n)
{
    kv_head(k, G_ARR);
    w8(&type, 4);
    w8(&n, 8);
}

/* ---- the tokenizer ---- */

struct vocab {
    uint32_t n;
    const char **tok; /* decoded strings, in the JSON document */
    size_t *len;
    int32_t *type;
    const struct janas_json *merges;
    uint64_t n_merges;
    int32_t bos, eos, pad;
};

static int32_t token_id(const struct vocab *v, const char *s)
{
    for (uint32_t i = 0; i < v->n; i++)
        if (v->tok[i] && v->len[i] == strlen(s) &&
            !memcmp(v->tok[i], s, v->len[i]))
            return (int32_t)i;
    return -1;
}

static struct vocab read_vocab(const struct janas_json *root, uint32_t rows)
{
    struct vocab v = {.n = rows, .bos = -1, .eos = -1, .pad = -1};
    const struct janas_json *model = janas_json_get(root, "model");
    if (!janas_json_is(janas_json_get(model, "type"), "BPE"))
        die("tokenizer.json: not a BPE model", NULL);
    v.tok = calloc(rows, sizeof(*v.tok));
    v.len = calloc(rows, sizeof(*v.len));
    v.type = calloc(rows, sizeof(*v.type));
    if (!v.tok || !v.len || !v.type)
        die("out of memory", NULL);
    for (const struct janas_json *t = janas_json_get(model, "vocab")->child; t;
         t = t->next) {
        double id = janas_json_num(t, -1);
        if (id < 0 || id >= rows)
            die("tokenizer.json: a token id beyond the embeddings", NULL);
        v.tok[(uint32_t)id] = t->key;
        v.len[(uint32_t)id] = t->key_n;
        v.type[(uint32_t)id] = 1; /* normal */
    }
    const struct janas_json *added = janas_json_get(root, "added_tokens");
    for (const struct janas_json *t = added ? added->child : NULL; t;
         t = t->next) {
        double id = janas_json_num(janas_json_get(t, "id"), -1);
        const struct janas_json *c = janas_json_get(t, "content");
        if (id < 0 || id >= rows || !c)
            die("tokenizer.json: a bad added token", NULL);
        const struct janas_json *sp = janas_json_get(t, "special");
        v.tok[(uint32_t)id] = c->s;
        v.len[(uint32_t)id] = c->n;
        v.type[(uint32_t)id] = sp && sp->type == JANAS_JSON_TRUE ? 3 : 4;
    }
    for (uint32_t i = 0; i < rows; i++)
        if (!v.tok[i])
            die("tokenizer.json: the vocabulary has holes", NULL);
    v.merges = janas_json_get(model, "merges");
    v.n_merges = v.merges ? v.merges->n : 0;
    v.bos = token_id(&v, "<|begin_of_text|>");
    v.eos = token_id(&v, "<|end_of_text|>");
    v.pad = token_id(&v, "[PAD]");
    if (v.bos < 0 || v.eos < 0)
        die("tokenizer.json: no <|begin_of_text|> or <|end_of_text|>", NULL);
    return v;
}

/* A merge as "a b", from either form tokenizer.json writes. */
static void write_merge(const struct janas_json *m)
{
    if (m->type == JANAS_JSON_STRING) {
        wstrn(m->s, m->n);
        return;
    }
    const struct janas_json *a = m->child, *b = a ? a->next : NULL;
    if (!a || !b || a->type != JANAS_JSON_STRING ||
        b->type != JANAS_JSON_STRING)
        die("tokenizer.json: a bad merge", NULL);
    uint64_t n = a->n + 1 + b->n;
    w8(&n, 8);
    w8(a->s, a->n);
    w8(" ", 1);
    w8(b->s, b->n);
}

/* ---- the tensors written ---- */

enum { SRC_NORM, SRC_MATRIX, SRC_ROPE };

struct gt {
    char name[64];
    char hf[160];     /* the source tensor */
    int src, permute; /* permute: the heads of the query (n_head) or key */
    uint32_t heads;
    uint64_t ne[2]; /* ggml order: ne[0] the row length */
    uint64_t offset;
};

static struct gt gts[1024];
static int n_gts;

static void add(const char *name, const char *hf, int src, uint32_t heads)
{
    struct gt *g = &gts[n_gts++];
    memset(g, 0, sizeof(*g));
    snprintf(g->name, sizeof(g->name), "%s", name);
    snprintf(g->hf, sizeof(g->hf), "%s", hf);
    g->src = src;
    g->permute = heads != 0;
    g->heads = heads;
}

static uint64_t gt_bytes(const struct gt *g)
{
    uint64_t n = g->ne[0] * (g->ne[1] ? g->ne[1] : 1);
    return n * (g->src == SRC_MATRIX ? 2 : 4);
}

/* The source tensor of g: the adapter's whole tables first. */
static const struct st_tensor *source(const struct gt *g)
{
    char nm[256];
    snprintf(nm, sizeof(nm), "%s%s", lora_prefix, g->hf);
    const struct st_tensor *t = have_lora ? st_get(nm) : NULL;
    return t ? t : st_get(g->hf);
}

static void pad_to_align(uint64_t at)
{
    static const char zero[GGUF_ALIGN];
    w8(zero, (size_t)((GGUF_ALIGN - at % GGUF_ALIGN) % GGUF_ALIGN));
}

static void write_tensor(const struct gt *g, const struct hp *h)
{
    uint64_t n = g->ne[0] * (g->ne[1] ? g->ne[1] : 1);
    if (g->src == SRC_ROPE) {
        float *f = rope_factors(h);
        w8(f, n * 4);
        free(f);
        return;
    }
    const struct st_tensor *t = source(g);
    if (st_count(t) != n)
        die("unexpected shape of", g->hf);
    float *x = st_load(t);
    if (g->src == SRC_NORM) {
        w8(x, n * 4);
        free(x);
        return;
    }
    uint64_t rows = g->ne[1], cols = g->ne[0];
    if (have_lora)
        merge(x, g->hf, rows, cols);
    uint16_t *y = xmalloc(n * 2);
    if (g->permute) {
        /* rows of head hh: [2][half] -> [half][2] */
        uint64_t hd = rows / g->heads, half = hd / 2;
        for (uint64_t hh = 0; hh < g->heads; hh++)
            for (uint64_t j = 0; j < 2; j++)
                for (uint64_t i = 0; i < half; i++) {
                    const float *src = x + (hh * hd + j * half + i) * cols;
                    uint16_t *dst = y + (hh * hd + i * 2 + j) * cols;
                    for (uint64_t c = 0; c < cols; c++)
                        dst[c] = f32_to_bf16(src[c]);
                }
    } else
        for (uint64_t i = 0; i < n; i++)
            y[i] = f32_to_bf16(x[i]);
    w8(y, n * 2);
    free(y);
    free(x);
}

static void scan_dir(const char *dir, int adapter)
{
    DIR *d = opendir(dir);
    if (!d)
        die("cannot open", dir);
    struct dirent *e;
    int found = 0;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors"))
            continue;
        if (adapter && strcmp(e->d_name, "adapter_model.safetensors"))
            continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        st_scan(path);
        found++;
    }
    closedir(d);
    if (!found)
        die("no safetensors in", dir);
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr,
                "usage: hf2gguf <model dir> [<adapter dir>] <out.gguf>\n");
        return 2;
    }
    const char *mdir = argv[1], *adir = argc == 4 ? argv[2] : NULL;
    const char *opath = argv[argc - 1];
    struct hp h = read_config(mdir);
    scan_dir(mdir, 0);
    if (adir) {
        struct janas_json_doc *ac = read_json(adir, "adapter_config.json");
        const struct janas_json *c = janas_json_root(ac);
        if (!janas_json_is(janas_json_get(c, "peft_type"), "LORA"))
            die("not a LoRA adapter", adir);
        const struct janas_json *rs = janas_json_get(c, "use_rslora");
        const struct janas_json *dora = janas_json_get(c, "use_dora");
        if ((rs && rs->type == JANAS_JSON_TRUE) ||
            (dora && dora->type == JANAS_JSON_TRUE))
            die("rsLoRA and DoRA are not supported", adir);
        lora_scale = (float)(num(c, "lora_alpha") / num(c, "r"));
        janas_json_free(ac);
        scan_dir(adir, 1);
        have_lora = 1;
    }
    const char *tdir = adir ? adir : mdir;
    struct janas_json_doc *tj = read_json(tdir, "tokenizer.json");

    /* the shapes: the embeddings' rows are the vocabulary */
    const struct st_tensor *emb =
        source(&(struct gt){.hf = "model.embed_tokens.weight"});
    if (!emb || emb->dims != 2 || emb->shape[1] != h.d)
        die("no model.embed_tokens.weight of the model's width", NULL);
    uint32_t rows = (uint32_t)emb->shape[0];
    struct vocab v = read_vocab(janas_json_root(tj), rows);

    add("token_embd.weight", "model.embed_tokens.weight", SRC_MATRIX, 0);
    gts[n_gts - 1].ne[0] = h.d, gts[n_gts - 1].ne[1] = rows;
    const char *head = "lm_head.weight";
    char nm[256];
    snprintf(nm, sizeof(nm), "%s%s", lora_prefix, head);
    if (!st_get(head) && !(have_lora && st_get(nm))) {
        if (!h.tied)
            die("no lm_head.weight and the embeddings are not tied", NULL);
        head = "model.embed_tokens.weight";
    }
    add("output.weight", head, SRC_MATRIX, 0);
    gts[n_gts - 1].ne[0] = h.d, gts[n_gts - 1].ne[1] = rows;
    add("output_norm.weight", "model.norm.weight", SRC_NORM, 0);
    gts[n_gts - 1].ne[0] = h.d;
    if (h.rs_factor > 0) {
        add("rope_freqs.weight", "", SRC_ROPE, 0);
        gts[n_gts - 1].ne[0] = h.hd / 2;
    }
    for (uint32_t l = 0; l < h.n_layer; l++) {
        static const struct {
            const char *gg, *hf;
            int src, which; /* which: 1 q, 2 k */
            int in, out;    /* 0 d, 1 ff, 2 q dim, 3 kv dim */
        } part[] = {
            {"attn_norm", "input_layernorm", SRC_NORM, 0, 0, -1},
            {"attn_q", "self_attn.q_proj", SRC_MATRIX, 1, 0, 2},
            {"attn_k", "self_attn.k_proj", SRC_MATRIX, 2, 0, 3},
            {"attn_v", "self_attn.v_proj", SRC_MATRIX, 0, 0, 3},
            {"attn_output", "self_attn.o_proj", SRC_MATRIX, 0, 2, 0},
            {"ffn_norm", "post_attention_layernorm", SRC_NORM, 0, 0, -1},
            {"ffn_gate", "mlp.gate_proj", SRC_MATRIX, 0, 0, 1},
            {"ffn_up", "mlp.up_proj", SRC_MATRIX, 0, 0, 1},
            {"ffn_down", "mlp.down_proj", SRC_MATRIX, 0, 1, 0},
        };
        uint64_t dims[4] = {h.d, h.ff, (uint64_t)h.n_head * h.hd,
                            (uint64_t)h.n_kv * h.hd};
        for (size_t p = 0; p < sizeof(part) / sizeof(part[0]); p++) {
            char gg[64], hf[160];
            snprintf(gg, sizeof(gg), "blk.%u.%s.weight", l, part[p].gg);
            snprintf(hf, sizeof(hf), "model.layers.%u.%s.weight", l,
                     part[p].hf);
            add(gg, hf, part[p].src,
                part[p].which == 1   ? h.n_head
                : part[p].which == 2 ? h.n_kv
                                     : 0);
            struct gt *g = &gts[n_gts - 1];
            g->ne[0] = dims[part[p].in];
            g->ne[1] = part[p].out < 0 ? 0 : dims[part[p].out];
        }
    }
    for (int i = 0; i < n_gts; i++)
        if (gts[i].src != SRC_ROPE && !source(&gts[i]))
            die("tensor missing", gts[i].hf);

    out = fopen(opath, "wb");
    if (!out)
        die("cannot create", opath);
    uint32_t ver = 3;
    uint64_t ntens = (uint64_t)n_gts, nkv = 0;
    w8("GGUF", 4);
    w8(&ver, 4);
    w8(&ntens, 8);
    long nkv_at = ftell(out);
    w8(&nkv, 8);
#define KV(call) (call, nkv++)
    KV(kv_str("general.architecture", "llama"));
    KV(kv_str("general.name", adir ? adir : mdir));
    KV(kv_u32("general.file_type", 32)); /* MOSTLY_BF16 */
    KV(kv_u32("general.quantization_version", 2));
    KV(kv_u32("llama.vocab_size", rows));
    KV(kv_u32("llama.context_length", h.ctx));
    KV(kv_u32("llama.embedding_length", h.d));
    KV(kv_u32("llama.block_count", h.n_layer));
    KV(kv_u32("llama.feed_forward_length", h.ff));
    KV(kv_u32("llama.attention.head_count", h.n_head));
    KV(kv_u32("llama.attention.head_count_kv", h.n_kv));
    KV(kv_u32("llama.attention.key_length", h.hd));
    KV(kv_u32("llama.attention.value_length", h.hd));
    KV(kv_f32("llama.attention.layer_norm_rms_epsilon", h.eps));
    KV(kv_f32("llama.rope.freq_base", h.rope_base));
    KV(kv_u32("llama.rope.dimension_count", h.hd));
    KV(kv_str("tokenizer.ggml.model", "gpt2"));
    KV(kv_str("tokenizer.ggml.pre", "llama-bpe"));
    kv_arr("tokenizer.ggml.tokens", G_STR, rows), nkv++;
    for (uint32_t i = 0; i < rows; i++)
        wstrn(v.tok[i], v.len[i]);
    kv_arr("tokenizer.ggml.token_type", G_I32, rows), nkv++;
    w8(v.type, rows * 4);
    kv_arr("tokenizer.ggml.merges", G_STR, v.n_merges), nkv++;
    for (const struct janas_json *m = v.merges ? v.merges->child : NULL; m;
         m = m->next)
        write_merge(m);
    KV(kv_u32("tokenizer.ggml.bos_token_id", (uint32_t)v.bos));
    KV(kv_u32("tokenizer.ggml.eos_token_id", (uint32_t)v.eos));
    if (v.pad >= 0)
        KV(kv_u32("tokenizer.ggml.padding_token_id", (uint32_t)v.pad));
    KV(kv_bool("tokenizer.ggml.add_bos_token", 1));
#undef KV
    uint64_t off = 0;
    for (int i = 0; i < n_gts; i++) {
        struct gt *g = &gts[i];
        uint32_t nd = g->ne[1] ? 2 : 1;
        uint32_t ty = g->src == SRC_MATRIX ? GT_BF16 : GT_F32;
        g->offset = off;
        wstr(g->name);
        w8(&nd, 4);
        w8(g->ne, 8 * nd);
        w8(&ty, 4);
        w8(&off, 8);
        off += (gt_bytes(g) + GGUF_ALIGN - 1) / GGUF_ALIGN * GGUF_ALIGN;
    }
    pad_to_align((uint64_t)ftell(out));
    for (int i = 0; i < n_gts; i++) {
        write_tensor(&gts[i], &h);
        pad_to_align(gt_bytes(&gts[i]));
        if (i % 16 == 0)
            fprintf(stderr, "\rhf2gguf: %d/%d tensors", i + 1, n_gts);
    }
    fseek(out, nkv_at, SEEK_SET);
    w8(&nkv, 8);
    if (fclose(out))
        die("cannot write", opath);
    fprintf(stderr, "\rhf2gguf: %s, %d tensors, %u tokens, %llu merges%s\n",
            opath, n_gts, rows, (unsigned long long)v.n_merges,
            have_lora ? ", adapter merged" : "");
    janas_json_free(tj);
    return 0;
}
