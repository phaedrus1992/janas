/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-nsm - a word's explication in the Natural Semantic Metalanguage
 * (NSM), written by a model trained for it (DeepNSM, arXiv 2505.11764:
 * Llama 3 with a LoRA adapter, converted with tools/hf2gguf), and what the
 * explication is made of: its primes, function words and other words
 * ("molecules"), DeepNSM's legality, whether it is circular (score.h).
 *
 *   janas-nsm <model.jns> <word> [example sentence...]   one word
 *   janas-nsm <model.jns> --batch <in.jsonl>            a file of words
 *   janas-nsm --score <in.jsonl>                        explications given
 *
 * A word's lines in a batch are JSON objects with "word" and "examples"
 * (DeepNSM's dataset as it is); --score reads "word" and "explication".
 * A batch writes one JSON line per explication (--out, else the standard
 * output) and a summary on the standard error.
 *
 * The prompt is the model card's ("Word: ...\nExamples:\n...\nParaphrase:");
 * --prompt paper is the one the paper's numbers were obtained with (its
 * evaluation script's, indentation included). The sampling is DeepNSM's
 * (temperature 0.7, top-k 50, top-p 0.95, at most 512 tokens).
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"
#include "llm/json.h"
#include "score.h"
#include "vocab.h"

struct opts {
    const char *model, *batch, *score, *out, *prompt, *molecules;
    int strict;
    enum nsm_lex lex;
    int samples, max, quiet;
    uint64_t seed;
    float temp, top_p;
    int top_k;
};

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-nsm <model.jns> <word> [example sentence...] "
            "[options]\n"
            "       janas-nsm <model.jns> --batch <in.jsonl> [options]\n"
            "       janas-nsm --score <in.jsonl> [options]\n"
            "A word's explication in NSM primes, by a model trained for it\n"
            "(DeepNSM), and what it is made of: primes, function words and\n"
            "other words (molecules), legality 10 (primes - molecules) / "
            "words,\n"
            "circularity (the word itself in its explication).\n"
            "  --batch <file>   JSON lines with \"word\" and \"examples\"\n"
            "  --score <file>   JSON lines with \"word\" and \"explication\": "
            "count\n"
            "                   them, no model\n"
            "  --out <file>     a batch's JSON lines (default: standard "
            "output)\n"
            "  -n <count>       explications a word (default 1)\n"
            "  --seed <n>       the sampling's seed (default 1)\n"
            "  --prompt <card|paper>  the model card's prompt (default) or "
            "the\n"
            "                   one the paper's numbers were obtained with\n"
            "  --lexicon <paper|full> primes as DeepNSM's scores counted "
            "them\n"
            "                   (default), or the 65 with their forms and "
            "phrases\n"
            "  --strict         the model may write only the primes' forms, "
            "function\n"
            "                   words, punctuation and the molecules: its "
            "output head\n"
            "                   computes those tokens alone\n"
            "  --molecules <file>  the molecules --strict allows, one word a "
            "line\n"
            "                   (default: a list of everyday words; an empty "
            "file: none)\n"
            "  --temp <t>  --top-k <k>  --top-p <p>  --max <tokens>\n"
            "                   sampling (default 0.7, 50, 0.95, 512)\n"
            "  --quiet          no progress on the standard error\n");
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    if (b) {
        b[n] = 0;
        *len = (size_t)n;
    }
    return b;
}

/* --- the summary of a batch --- */

struct tally {
    size_t n;
    double leg, leg2, primes, molecules, circular;
    double out_tokens, out_s, in_tokens, in_s;
};

static void tally_add(struct tally *t, const struct nsm_score *s)
{
    t->n++;
    t->leg += s->legality;
    t->leg2 += s->legality * s->legality;
    t->primes += s->primes_ratio;
    t->molecules += s->molecules_ratio;
    t->circular += s->circular;
}

static void tally_print(const struct tally *t)
{
    if (!t->n)
        return;
    double n = (double)t->n, m = t->leg / n;
    double sd = t->n > 1 ? sqrt((t->leg2 - n * m * m) / (n - 1)) : 0;
    fprintf(stderr,
            "%zu explications: legality %.2f +- %.2f (standard error), "
            "primes %.1f%%, molecules %.1f%%, circular %.1f%%\n",
            t->n, m, sd / sqrt(n), 100 * t->primes / n, 100 * t->molecules / n,
            100 * t->circular / n);
    if (t->out_s > 0)
        fprintf(stderr,
                "writing %.0f tokens in %.1f s, %.1f tokens/s; reading the "
                "prompts %.0f tokens in %.1f s, %.0f tokens/s\n",
                t->out_tokens, t->out_s, t->out_tokens / t->out_s, t->in_tokens,
                t->in_s, t->in_s > 0 ? t->in_tokens / t->in_s : 0);
}

static void put_score(struct janas_buf *b, const struct nsm_score *s)
{
    janas_buf_printf(b,
                     ", \"legality\": %.4f, \"primes\": %.4f, "
                     "\"molecules\": %.4f, \"words\": %zu, \"circular\": %s",
                     s->legality, s->primes_ratio, s->molecules_ratio, s->words,
                     s->circular ? "true" : "false");
}

/* --- the model --- */

struct gen {
    janas_llm *m;
    janas_llm_chat *c;
    struct janas_llm_chat_params p;
};

/* The words of a file, one a line, NULL-terminated (malloc'd, into *text). */
static const char **read_words(const char *path, char **text)
{
    size_t len;
    *text = read_file(path, &len);
    if (!*text) {
        fprintf(stderr, "janas-nsm: %s: %s\n", path, strerror(errno));
        exit(1);
    }
    size_t n = 0;
    for (size_t i = 0; i < len; i++)
        n += (*text)[i] == '\n';
    const char **w = calloc(n + 2, sizeof(*w));
    size_t k = 0;
    for (char *p = strtok(*text, "\r\n"); p && w; p = strtok(NULL, "\r\n"))
        if (*p)
            w[k++] = p;
    return w;
}

/* --strict: the replies' vocabulary */
static void restrict_vocab(struct gen *g, const struct opts *o)
{
    char *text = NULL;
    const char **mol = o->molecules ? read_words(o->molecules, &text) : NULL;
    int32_t *ids = NULL;
    long n = nsm_vocab_tokens(g->m, mol, &ids);
    if (n <= 0 || janas_llm_chat_vocab(g->c, (int32_t)n, ids) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-nsm: --strict: %s\n",
                n <= 0 ? "no vocabulary" : janas_llm_last_error());
        exit(1);
    }
    if (!o->quiet)
        fprintf(stderr, "janas-nsm: --strict, %ld tokens of the vocabulary\n",
                n);
    free(ids);
    free(mol);
    free(text);
}

static void gen_open(struct gen *g, const struct opts *o)
{
    struct janas_llm_params lp;
    janas_llm_params_default(&lp);
    lp.n_ctx = 4096;
    if (!o->quiet)
        fprintf(stderr, "janas-nsm: loading %s ...\n", o->model);
    if (janas_llm_open(o->model, &lp, &g->m) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-nsm: %s\n", janas_llm_last_error());
        exit(1);
    }
    janas_llm_chat_params_default(&g->p);
    g->p.temperature = o->temp;
    g->p.top_k = o->top_k;
    g->p.top_p = o->top_p;
    g->p.min_p = 0;
    g->p.max_reply = o->max;
    g->p.speculate = 0;
    if (janas_llm_chat_create(g->m, &g->p, &g->c) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-nsm: %s\n", janas_llm_last_error());
        exit(1);
    }
    if (o->strict)
        restrict_vocab(g, o);
}

/* The prompt for a word and its examples. */
static void build_prompt(struct janas_buf *b, const struct opts *o,
                         const char *word, const char *const *ex, int n_ex)
{
    int paper = !strcmp(o->prompt, "paper");
    /* nsm_evaluation.py's f-string keeps its source's indentation */
    const char *ind = paper ? "                " : "";
    janas_buf_printf(b, "Word: %s\n%sExamples:\n%s", word, ind, ind);
    for (int i = 0; i < n_ex; i++)
        janas_buf_printf(b, "%s%s", i ? "\n" : "", ex[i]);
    janas_buf_printf(b, "\n%sParaphrase:%s%s", ind, paper ? "\n" : "", ind);
}

/* One explication into out; its tokens and times into t. */
static int explain(struct gen *g, const char *prompt, size_t n, uint64_t seed,
                   struct janas_buf *out, struct tally *t)
{
    g->p.seed = seed;
    if (janas_llm_chat_reset(g->c) != JANAS_LLM_OK ||
        janas_llm_chat_set_params(g->c, &g->p) != JANAS_LLM_OK ||
        janas_llm_chat_prompt(g->c, prompt, (int32_t)n) != JANAS_LLM_OK)
        return -1;
    out->n = 0;
    char piece[256];
    for (;;) {
        int32_t len = 0;
        int32_t rc = janas_llm_chat_next(g->c, piece, sizeof(piece), &len);
        if (rc == JANAS_LLM_DONE)
            break;
        if (rc != JANAS_LLM_OK)
            return -1;
        janas_buf_put(out, piece, (size_t)len);
    }
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    janas_llm_chat_stats(g->c, &st);
    t->out_tokens += st.output_tokens;
    t->out_s += st.output_seconds;
    t->in_tokens += st.input_tokens;
    t->in_s += st.input_seconds;
    /* the model writes a space after "Paraphrase:", and a newline last */
    size_t a = 0;
    while (a < out->n && (out->p[a] == ' ' || out->p[a] == '\n'))
        a++;
    while (out->n > a &&
           (out->p[out->n - 1] == ' ' || out->p[out->n - 1] == '\n'))
        out->n--;
    memmove(out->p, out->p + a, out->n - a);
    out->n -= a;
    janas_buf_put(out, "", 1);
    out->n--;
    return 0;
}

/* --- the three ways to run --- */

static int one_word(const struct opts *o, const char *word,
                    const char *const *ex, int n_ex)
{
    struct gen g;
    gen_open(&g, o);
    struct janas_buf pr = {0}, out = {0};
    build_prompt(&pr, o, word, ex, n_ex);
    struct tally t = {0};
    for (int k = 0; k < o->samples; k++) {
        if (explain(&g, pr.p, pr.n, o->seed * 1000003u + (uint64_t)k, &out,
                    &t) != 0) {
            fprintf(stderr, "janas-nsm: %s\n", janas_llm_last_error());
            return 1;
        }
        struct nsm_score s;
        nsm_score_text(o->lex, out.p, out.n, word, &s);
        tally_add(&t, &s);
        printf("%s%.*s\n\n[legality %.2f, primes %.0f%%, molecules %.0f%%, "
               "%zu words%s]\n",
               k ? "\n" : "", (int)out.n, out.p, s.legality,
               100 * s.primes_ratio, 100 * s.molecules_ratio, s.words,
               s.circular ? ", circular" : "");
    }
    if (o->samples > 1 || !o->quiet)
        tally_print(&t);
    janas_buf_free(&pr);
    janas_buf_free(&out);
    janas_llm_chat_destroy(g.c);
    janas_llm_close(g.m);
    return 0;
}

/* Calls f on each JSON line of the file. */
static int each_line(const char *path,
                     int (*f)(const struct janas_json *, size_t, void *),
                     void *ctx)
{
    size_t len;
    char *text = read_file(path, &len);
    if (!text) {
        fprintf(stderr, "janas-nsm: %s: %s\n", path, strerror(errno));
        return 1;
    }
    size_t line = 0;
    for (char *p = text; *p;) {
        char *e = strchr(p, '\n');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        line++;
        if (n) {
            char err[128];
            struct janas_json_doc *d = janas_json_parse(p, n, err, sizeof(err));
            if (!d) {
                fprintf(stderr, "janas-nsm: %s:%zu: %s\n", path, line, err);
                free(text);
                return 1;
            }
            int rc = f(janas_json_root(d), line, ctx);
            janas_json_free(d);
            if (rc) {
                free(text);
                return rc;
            }
        }
        p += n + (e != NULL);
    }
    free(text);
    return 0;
}

struct run {
    const struct opts *o;
    struct gen g;
    FILE *out;
    struct tally t;
    size_t index;
};

static int score_line(const struct janas_json *v, size_t line, void *ctx)
{
    struct run *r = ctx;
    const char *w = janas_json_str(janas_json_get(v, "word"));
    const struct janas_json *e = janas_json_get(v, "explication");
    if (!e)
        e = janas_json_get(v, "text");
    if (!w || !janas_json_str(e)) {
        fprintf(stderr,
                "janas-nsm: line %zu: no \"word\" and \"explication\"\n", line);
        return 1;
    }
    struct nsm_score s;
    nsm_score_text(r->o->lex, e->s, e->n, w, &s);
    tally_add(&r->t, &s);
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"word\": ");
    janas_json_write_str(&b, w, strlen(w));
    put_score(&b, &s);
    janas_buf_puts(&b, "}\n");
    fwrite(b.p, 1, b.n, r->out);
    janas_buf_free(&b);
    return 0;
}

static int batch_line(const struct janas_json *v, size_t line, void *ctx)
{
    struct run *r = ctx;
    const char *w = janas_json_str(janas_json_get(v, "word"));
    const struct janas_json *ex = janas_json_get(v, "examples");
    if (!w || !ex || ex->type != JANAS_JSON_ARRAY) {
        fprintf(stderr, "janas-nsm: line %zu: no \"word\" and \"examples\"\n",
                line);
        return 1;
    }
    const char *exs[64];
    int n_ex = 0;
    for (const struct janas_json *x = ex->child; x && n_ex < 64; x = x->next)
        if (janas_json_str(x))
            exs[n_ex++] = x->s;
    struct janas_buf pr = {0}, out = {0}, b = {0};
    build_prompt(&pr, r->o, w, exs, n_ex);
    for (int k = 0; k < r->o->samples; k++) {
        double s0 = r->t.out_s, t0 = r->t.out_tokens;
        uint64_t seed = (r->o->seed * 1000003u + r->index) * 31u + (uint64_t)k;
        if (explain(&r->g, pr.p, pr.n, seed, &out, &r->t) != 0) {
            fprintf(stderr, "janas-nsm: line %zu: %s\n", line,
                    janas_llm_last_error());
            return 1;
        }
        struct nsm_score s;
        nsm_score_text(r->o->lex, out.p, out.n, w, &s);
        tally_add(&r->t, &s);
        b.n = 0;
        janas_buf_puts(&b, "{\"word\": ");
        janas_json_write_str(&b, w, strlen(w));
        janas_buf_printf(&b, ", \"sample\": %d, \"explication\": ", k);
        janas_json_write_str(&b, out.p, out.n);
        put_score(&b, &s);
        janas_buf_printf(&b, ", \"tokens\": %.0f, \"seconds\": %.3f}\n",
                         r->t.out_tokens - t0, r->t.out_s - s0);
        fwrite(b.p, 1, b.n, r->out);
        fflush(r->out);
    }
    r->index++;
    if (!r->o->quiet && r->index % 10 == 0)
        fprintf(stderr, "janas-nsm: %zu words, %.1f tokens/s\n", r->index,
                r->t.out_s > 0 ? r->t.out_tokens / r->t.out_s : 0);
    janas_buf_free(&pr);
    janas_buf_free(&out);
    janas_buf_free(&b);
    return 0;
}

static int run_file(const struct opts *o, int with_model)
{
    struct run r = {.o = o, .out = stdout};
    if (o->out && !(r.out = fopen(o->out, "w"))) {
        fprintf(stderr, "janas-nsm: %s: %s\n", o->out, strerror(errno));
        return 1;
    }
    if (with_model)
        gen_open(&r.g, o);
    int rc = each_line(with_model ? o->batch : o->score,
                       with_model ? batch_line : score_line, &r);
    tally_print(&r.t);
    if (r.out != stdout)
        fclose(r.out);
    if (with_model) {
        janas_llm_chat_destroy(r.g.c);
        janas_llm_close(r.g.m);
    }
    return rc;
}

int main(int argc, char **argv)
{
    struct opts o = {.prompt = "card",
                     .lex = NSM_LEX_PAPER,
                     .samples = 1,
                     .max = 512,
                     .seed = 1,
                     .temp = 0.7f,
                     .top_p = 0.95f,
                     .top_k = 50};
    const char *pos[66];
    int n_pos = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define TAKES(name) (!strcmp(a, name) && v && ++i)
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage();
            return 0;
        } else if (TAKES("--batch"))
            o.batch = v;
        else if (TAKES("--score"))
            o.score = v;
        else if (TAKES("--out"))
            o.out = v;
        else if (TAKES("-n"))
            o.samples = atoi(v);
        else if (TAKES("--seed"))
            o.seed = strtoull(v, NULL, 10);
        else if (TAKES("--prompt"))
            o.prompt = v;
        else if (TAKES("--lexicon"))
            o.lex = !strcmp(v, "full") ? NSM_LEX_FULL : NSM_LEX_PAPER;
        else if (TAKES("--temp"))
            o.temp = (float)atof(v);
        else if (TAKES("--top-k"))
            o.top_k = atoi(v);
        else if (TAKES("--top-p"))
            o.top_p = (float)atof(v);
        else if (TAKES("--max"))
            o.max = atoi(v);
        else if (TAKES("--molecules"))
            o.molecules = v;
        else if (!strcmp(a, "--strict"))
            o.strict = 1;
        else if (!strcmp(a, "--quiet"))
            o.quiet = 1;
        else if (a[0] == '-' && a[1]) {
            fprintf(stderr, "janas-nsm: unknown option %s\n", a);
            usage();
            return 2;
        } else if (n_pos < 66)
            pos[n_pos++] = a;
#undef TAKES
    }
    if (strcmp(o.prompt, "card") && strcmp(o.prompt, "paper")) {
        fprintf(stderr, "janas-nsm: --prompt card or paper\n");
        return 2;
    }
    if (o.samples < 1 || o.max < 1) {
        usage();
        return 2;
    }
    if (o.score)
        return run_file(&o, 0);
    if (n_pos < 1) {
        usage();
        return 2;
    }
    o.model = pos[0];
    if (o.batch)
        return run_file(&o, 1);
    if (n_pos < 2) {
        usage();
        return 2;
    }
    return one_word(&o, pos[1], pos + 2, n_pos - 2);
}
