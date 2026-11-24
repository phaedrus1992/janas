/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio.c - POST /audio/transcriptions and /audio/translations: a recording
 * (multipart/form-data, its file a WAV) turned into text by the speech
 * model the server was started with (--speech-model, a Whisper model), as
 * OpenAI's whisper-1 answers: json, text, srt, vtt or verbose_json with
 * the segments and their times. The model runs in the request's thread,
 * one recording at a time, beside the chat model's engine.
 */
#include "llm/json.h"
#include "multipart.h"
#include "server.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <microhttpd.h>

int srv_speech_start(struct srv *s, char *err, size_t err_len)
{
    pthread_mutex_init(&s->speech_mu, NULL);
    if (!s->cfg.speech_path)
        return 0;
    struct janas_llm_speech_params p;
    janas_llm_speech_params_default(&p);
    p.vad_model = s->cfg.vad_path;
    p.speaker_model = s->cfg.speaker_path;
    if (janas_llm_speech_open(s->cfg.speech_path, &p, &s->speech) !=
        JANAS_LLM_OK) {
        snprintf(err, err_len, "%s", janas_llm_last_error());
        return -1;
    }
    return 0;
}

void srv_speech_stop(struct srv *s)
{
    janas_llm_speech_close(s->speech);
    s->speech = NULL;
    pthread_mutex_destroy(&s->speech_mu);
}

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

/* The text of a form field, NUL-terminated into buf: 1, 0 when absent, -1
   when it does not fit or is a file. */
static int field(const struct srv_part *parts, int n, const char *name,
                 char *buf, size_t cap)
{
    const struct srv_part *p = srv_multipart_get(parts, n, name);
    if (!p)
        return 0;
    if (p->n >= cap || p->has_filename)
        return -1;
    memcpy(buf, p->data, p->n);
    buf[p->n] = 0;
    return 1;
}

/* seconds as 00:00:00,000 (srt) or 00:00:00.000 (vtt) */
static void stamp(struct janas_buf *b, double t, char sep)
{
    long ms = lround(t * 1000);
    janas_buf_printf(b, "%02ld:%02ld:%02ld%c%03ld", ms / 3600000,
                     ms / 60000 % 60, ms / 1000 % 60, sep, ms % 1000);
}

/* segment i's text, malloc'd */
static char *seg_text(const janas_llm_speech *sp, int32_t i, double *t0,
                      double *t1)
{
    int32_t len = 0;
    janas_llm_speech_segment(sp, i, t0, t1, NULL, 0, &len);
    char *t = malloc((size_t)len + 1);
    if (t && janas_llm_speech_segment(sp, i, NULL, NULL, t, len + 1, &len) !=
                 JANAS_LLM_OK) {
        free(t);
        t = NULL;
    }
    return t;
}

static char *whole_text(const janas_llm_speech *sp)
{
    int32_t len = 0;
    janas_llm_speech_text(sp, NULL, 0, &len);
    char *t = malloc((size_t)len + 1);
    if (t && janas_llm_speech_text(sp, t, len + 1, &len) != JANAS_LLM_OK) {
        free(t);
        t = NULL;
    }
    return t;
}

/* srt and vtt: a cue a segment */
static int reply_cues(struct srv_req *r, const janas_llm_speech *sp, int vtt)
{
    struct janas_buf b = {0};
    if (vtt)
        janas_buf_puts(&b, "WEBVTT\n\n");
    for (int32_t i = 0; i < janas_llm_speech_segments(sp); i++) {
        double t0, t1;
        char *t = seg_text(sp, i, &t0, &t1);
        if (!t) {
            janas_buf_free(&b);
            return srv_reply_error(r, 500, "server_error", NULL,
                                   "out of memory");
        }
        if (!vtt)
            janas_buf_printf(&b, "%d\n", i + 1);
        stamp(&b, t0, vtt ? '.' : ',');
        janas_buf_puts(&b, " --> ");
        stamp(&b, t1, vtt ? '.' : ',');
        janas_buf_printf(&b, "\n%s\n\n", t[0] == ' ' ? t + 1 : t);
        free(t);
    }
    if (b.oom) {
        janas_buf_free(&b);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    return srv_reply_bytes(r, 200, b.p ? b.p : calloc(1, 1), b.n,
                           vtt ? "text/vtt; charset=utf-8"
                               : "application/x-subrip; charset=utf-8");
}

/* OpenAI's diarized_json: the turns as transcript.text.segment items */
static void add_turns(yyjson_mut_doc *d, yyjson_mut_val *arr,
                      const janas_llm_speech *sp)
{
    for (int32_t i = 0; i < janas_llm_speech_turns(sp); i++) {
        double t0, t1;
        char who[128];
        int32_t wl = 0, tl = 0;
        janas_llm_speech_turn(sp, i, &t0, &t1, who, sizeof who, &wl, NULL, 0,
                              &tl);
        char *t = malloc((size_t)tl + 1);
        if (!t)
            return;
        janas_llm_speech_turn(sp, i, NULL, NULL, NULL, 0, NULL, t, tl + 1, &tl);
        char id[32];
        snprintf(id, sizeof id, "seg_%d", i);
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_str(d, o, "type", "transcript.text.segment");
        yyjson_mut_obj_add_strcpy(d, o, "id", id);
        yyjson_mut_obj_add_real(d, o, "start", t0);
        yyjson_mut_obj_add_real(d, o, "end", t1);
        yyjson_mut_obj_add_strcpy(d, o, "text", t);
        yyjson_mut_obj_add_strcpy(d, o, "speaker", who);
        free(t);
    }
}

static int reply_diarized(struct srv_req *r, const janas_llm_speech *sp)
{
    char *text = whole_text(sp);
    if (!text)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    double dur = janas_llm_speech_duration(sp);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "task", "transcribe");
    yyjson_mut_obj_add_real(d, o, "duration", dur);
    yyjson_mut_obj_add_strcpy(d, o, "text", text);
    free(text);
    add_turns(d, yyjson_mut_obj_add_arr(d, o, "segments"), sp);
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
    yyjson_mut_obj_add_str(d, u, "type", "duration");
    yyjson_mut_obj_add_int(d, u, "seconds", (int64_t)ceil(dur));
    size_t len;
    char *json = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, len);
}

/* granularities: 1 segments, 2 words */
static void add_logprobs(yyjson_mut_doc *d, yyjson_mut_val *arr,
                         const janas_llm_speech *sp, int32_t i);

static int reply_json(struct srv_req *r, const janas_llm_speech *sp,
                      int verbose, int translate, int gran, int logprobs)
{
    char *text = whole_text(sp);
    if (!text)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    double dur = janas_llm_speech_duration(sp);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    if (verbose) {
        char code[16];
        int32_t n;
        janas_llm_speech_language(sp, code, sizeof code, &n);
        yyjson_mut_obj_add_str(d, o, "task",
                               translate ? "translate" : "transcribe");
        /* OpenAI names the language in English, the translation's too */
        yyjson_mut_obj_add_strcpy(d, o, "language",
                                  translate ? "english"
                                  : janas_llm_speech_language_name(code)
                                      ? janas_llm_speech_language_name(code)
                                      : code);
        yyjson_mut_obj_add_real(d, o, "duration", dur);
    }
    yyjson_mut_obj_add_strcpy(d, o, "text", text);
    free(text);
    if (logprobs) {
        yyjson_mut_val *a = yyjson_mut_obj_add_arr(d, o, "logprobs");
        for (int32_t i = 0; i < janas_llm_speech_segments(sp); i++)
            add_logprobs(d, a, sp, i);
    }
    if (verbose && (gran & 2)) {
        /* OpenAI's words: without their leading space */
        yyjson_mut_val *ws = yyjson_mut_obj_add_arr(d, o, "words");
        for (int32_t i = 0; i < janas_llm_speech_words(sp); i++) {
            double t0, t1;
            int32_t len = 0;
            janas_llm_speech_word(sp, i, &t0, &t1, NULL, NULL, 0, &len);
            char *t = malloc((size_t)len + 1);
            if (!t)
                break;
            janas_llm_speech_word(sp, i, NULL, NULL, NULL, t, len + 1, &len);
            const char *b = t;
            while (*b == ' ')
                b++;
            yyjson_mut_val *w = yyjson_mut_arr_add_obj(d, ws);
            yyjson_mut_obj_add_strcpy(d, w, "word", b);
            yyjson_mut_obj_add_real(d, w, "start", t0);
            yyjson_mut_obj_add_real(d, w, "end", t1);
            free(t);
        }
    }
    if (verbose && (gran & 1)) {
        yyjson_mut_val *segs = yyjson_mut_obj_add_arr(d, o, "segments");
        for (int32_t i = 0; i < janas_llm_speech_segments(sp); i++) {
            double t0, t1;
            float lp, ns, temp, cr;
            int32_t ids[512], nt = 0;
            char *t = seg_text(sp, i, &t0, &t1);
            janas_llm_speech_segment_info(sp, i, &lp, &ns, &temp, &cr, ids, 512,
                                          &nt);
            yyjson_mut_val *s = yyjson_mut_arr_add_obj(d, segs);
            yyjson_mut_obj_add_int(d, s, "id", i);
            /* the 10 ms frame it starts at */
            yyjson_mut_obj_add_int(d, s, "seek", (int64_t)lround(t0 * 100));
            yyjson_mut_obj_add_real(d, s, "start", t0);
            yyjson_mut_obj_add_real(d, s, "end", t1);
            yyjson_mut_obj_add_strcpy(d, s, "text", t ? t : "");
            yyjson_mut_val *a = yyjson_mut_obj_add_arr(d, s, "tokens");
            for (int32_t k = 0; k < nt && k < 512; k++)
                yyjson_mut_arr_add_int(d, a, ids[k]);
            yyjson_mut_obj_add_real(d, s, "temperature", temp);
            yyjson_mut_obj_add_real(d, s, "avg_logprob", lp);
            yyjson_mut_obj_add_real(d, s, "compression_ratio", cr);
            yyjson_mut_obj_add_real(d, s, "no_speech_prob", ns);
            free(t);
        }
    }
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
    yyjson_mut_obj_add_str(d, u, "type", "duration");
    yyjson_mut_obj_add_int(d, u, "seconds", (int64_t)ceil(dur));
    size_t len;
    char *json = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, len);
}

/* the log-probabilities of segment i's text tokens, as OpenAI's include[]
   = logprobs gives them: each token's text, log-probability and bytes */
static void add_logprobs(yyjson_mut_doc *d, yyjson_mut_val *arr,
                         const janas_llm_speech *sp, int32_t i)
{
    int32_t ids[512], nt = 0, nl = 0;
    float lp[512];
    janas_llm_speech_segment_info(sp, i, NULL, NULL, NULL, NULL, ids, 512, &nt);
    janas_llm_speech_segment_logprobs(sp, i, lp, 512, &nl);
    for (int32_t k = 0; k < nt && k < nl && k < 512; k++) {
        if (!janas_llm_speech_token_is_text(sp, ids[k]))
            continue;
        char tb[256];
        int32_t tl = 0;
        janas_llm_speech_token(sp, ids[k], tb, sizeof tb, &tl);
        if (tl > (int32_t)sizeof tb - 1)
            tl = (int32_t)sizeof tb - 1;
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_strncpy(d, o, "token", tb, (size_t)tl);
        yyjson_mut_obj_add_real(d, o, "logprob", lp[k]);
        yyjson_mut_val *by = yyjson_mut_obj_add_arr(d, o, "bytes");
        for (int32_t j = 0; j < tl; j++)
            yyjson_mut_arr_add_int(d, by, (unsigned char)tb[j]);
    }
}

/* ---- the stream: a thread transcribes, the connection reads its events
   (transcript.text.delta, a segment at a time; transcript.text.done) ---- */

struct sstream {
    struct srv *srv;
    uint8_t *data;
    size_t n;
    char *prompt, lang[64];
    struct janas_llm_transcribe_params p;
    int logprobs, diarize;
    /* the known speakers: names and recordings, the stream's own */
    char *names[8];
    unsigned char *refs[8];
    const char *name_p[8];
    const void *ref_p[8];
    int64_t ref_n[8];
    pthread_t th;
    int started;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char *buf;
    size_t len, cap, off;
    int finished;
    volatile int32_t stop;
};

static void ss_event(struct sstream *x, yyjson_mut_doc *d)
{
    size_t n;
    char *json = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    if (!json)
        return;
    pthread_mutex_lock(&x->mu);
    size_t need = x->len + n + 8;
    if (need > x->cap) {
        size_t nc = need * 2;
        char *b = realloc(x->buf, nc);
        if (b) {
            x->buf = b;
            x->cap = nc;
        }
    }
    if (need <= x->cap) {
        memcpy(x->buf + x->len, "data: ", 6);
        memcpy(x->buf + x->len + 6, json, n);
        memcpy(x->buf + x->len + 6 + n, "\n\n", 2);
        x->len += n + 8;
    }
    pthread_cond_signal(&x->cv);
    pthread_mutex_unlock(&x->mu);
    free(json);
}

static void ss_segment(void *user, int32_t i)
{
    struct sstream *x = user;
    const janas_llm_speech *sp = x->srv->speech;
    double t0, t1;
    char *t = seg_text(sp, i, &t0, &t1);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "type", "transcript.text.delta");
    yyjson_mut_obj_add_strcpy(d, o, "delta", t ? t : "");
    if (x->logprobs)
        add_logprobs(d, yyjson_mut_obj_add_arr(d, o, "logprobs"), sp, i);
    free(t);
    ss_event(x, d);
}

static void *ss_run(void *arg)
{
    struct sstream *x = arg;
    janas_llm_speech *sp = x->srv->speech;
    pthread_mutex_lock(&x->srv->speech_mu);
    int32_t rc = janas_llm_speech_transcribe(sp, x->data, (int64_t)x->n, &x->p);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    if (rc != JANAS_LLM_OK) {
        yyjson_mut_obj_add_str(d, o, "type", "error");
        yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, o, "error");
        yyjson_mut_obj_add_str(d, e, "type",
                               rc == JANAS_LLM_EINVAL ? "invalid_request_error"
                                                      : "server_error");
        yyjson_mut_obj_add_strcpy(d, e, "message", janas_llm_last_error());
    } else {
        if (x->diarize) { /* the turns, once the whole recording is known */
            yyjson_mut_doc *dd = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *arr = yyjson_mut_arr(dd);
            yyjson_mut_doc_set_root(dd, arr);
            add_turns(dd, arr, sp);
            size_t na = yyjson_mut_arr_size(arr);
            for (size_t i = 0; i < na; i++) {
                yyjson_mut_doc *ed = yyjson_mut_doc_new(NULL);
                yyjson_mut_doc_set_root(
                    ed,
                    yyjson_mut_val_mut_copy(ed, yyjson_mut_arr_get(arr, i)));
                ss_event(x, ed);
            }
            yyjson_mut_doc_free(dd);
        }
        char *text = whole_text(sp);
        double dur = janas_llm_speech_duration(sp);
        yyjson_mut_obj_add_str(d, o, "type", "transcript.text.done");
        yyjson_mut_obj_add_strcpy(d, o, "text", text ? text : "");
        free(text);
        if (x->logprobs) {
            yyjson_mut_val *a = yyjson_mut_obj_add_arr(d, o, "logprobs");
            for (int32_t i = 0; i < janas_llm_speech_segments(sp); i++)
                add_logprobs(d, a, sp, i);
        }
        yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
        yyjson_mut_obj_add_str(d, u, "type", "duration");
        yyjson_mut_obj_add_int(d, u, "seconds", (int64_t)ceil(dur));
    }
    pthread_mutex_unlock(&x->srv->speech_mu);
    ss_event(x, d);
    pthread_mutex_lock(&x->mu);
    x->finished = 1;
    pthread_cond_signal(&x->cv);
    pthread_mutex_unlock(&x->mu);
    return NULL;
}

static long ss_next(void *ctx, char *buf, size_t cap)
{
    struct sstream *x = ctx;
    pthread_mutex_lock(&x->mu);
    if (x->off == x->len && !x->finished) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 200000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&x->cv, &x->mu, &ts);
    }
    long n = 0;
    if (x->off < x->len) {
        size_t k = x->len - x->off < cap ? x->len - x->off : cap;
        memcpy(buf, x->buf + x->off, k);
        x->off += k;
        n = (long)k;
    } else if (x->finished) {
        n = -1;
    }
    pthread_mutex_unlock(&x->mu);
    return n;
}

static void ss_done(void *ctx)
{
    struct sstream *x = ctx;
    x->stop = 1; /* a client gone: the transcription stops at its next
                    window */
    if (x->started)
        pthread_join(x->th, NULL);
    pthread_mutex_destroy(&x->mu);
    pthread_cond_destroy(&x->cv);
    free(x->buf);
    free(x->data);
    free(x->prompt);
    for (int i = 0; i < 8; i++) {
        free(x->names[i]);
        free(x->refs[i]);
    }
    free(x);
}

static int speech(struct srv_req *r, int translate)
{
    if (!r->srv->speech)
        return srv_reply_error(r, 400, "invalid_request_error",
                               "model_not_found",
                               "This server has no speech model: start it "
                               "with --speech-model FILE (a Whisper model "
                               "made by hf2jns_whisper).");
    if (!r->conn)
        return bad(r, "not in a batch");
    struct srv_part parts[32];
    char err[200];
    const char *ct =
        MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "Content-Type");
    int n =
        srv_multipart_parse(ct, r->body, r->n_body, parts, 32, err, sizeof err);
    if (n < 0)
        return bad(r, err);
    const struct srv_part *file = srv_multipart_get(parts, n, "file");
    if (!file || !file->n)
        return bad(r, "'file' is required: the recording.");
    char model[256], lang[64] = "", fmt[32] = "json", temp[32] = "";
    char stream[16] = "", *prompt = NULL;
    if (field(parts, n, "model", model, sizeof model) != 1)
        return bad(r, "'model' is required.");
    if ((!translate && field(parts, n, "language", lang, sizeof lang) < 0) ||
        field(parts, n, "response_format", fmt, sizeof fmt) < 0 ||
        field(parts, n, "temperature", temp, sizeof temp) < 0 ||
        field(parts, n, "stream", stream, sizeof stream) < 0)
        return bad(r, "a field too long, or a file where text belongs.");
    int verbose = !strcmp(fmt, "verbose_json");
    int diarized = !strcmp(fmt, "diarized_json");
    if (!verbose && !diarized && strcmp(fmt, "json") && strcmp(fmt, "text") &&
        strcmp(fmt, "srt") && strcmp(fmt, "vtt"))
        return bad(r, "'response_format' is json, text, srt, verbose_json, "
                      "vtt or diarized_json.");
    int streaming = stream[0] && strcmp(stream, "false");
    int logprobs = 0;
    for (int i = 0; i < n; i++)
        if (!strcmp(parts[i].name, "include[]") ||
            !strcmp(parts[i].name, "include")) {
            if (parts[i].n == 8 && !memcmp(parts[i].data, "logprobs", 8))
                logprobs = 1;
            else
                return bad(r, "'include' takes logprobs.");
        }
    if (logprobs && strcmp(fmt, "json") && !verbose)
        return bad(r, "'include[]=logprobs' needs 'response_format' json.");
    if (streaming && strcmp(fmt, "json") && strcmp(fmt, "text") && !diarized)
        return bad(r, "'stream' gives text: 'response_format' json, text or "
                      "diarized_json.");
    int gran = 0;
    for (int i = 0; i < n; i++)
        if (!strcmp(parts[i].name, "timestamp_granularities[]") ||
            !strcmp(parts[i].name, "timestamp_granularities")) {
            if (parts[i].n == 7 && !memcmp(parts[i].data, "segment", 7))
                gran |= 1;
            else if (parts[i].n == 4 && !memcmp(parts[i].data, "word", 4))
                gran |= 2;
            else
                return bad(r, "'timestamp_granularities' takes word and "
                              "segment.");
        }
    /* chunking_strategy: "auto", or {"type": "server_vad", threshold,
       prefix_padding_ms, silence_duration_ms} */
    int chunking = 0, vad_prefix = -1, vad_silence = 0;
    float vad_threshold = 0;
    const struct srv_part *cs =
        srv_multipart_get(parts, n, "chunking_strategy");
    if (cs) {
        if (cs->has_filename)
            return bad(r, "'chunking_strategy' is text.");
        if (cs->n == 4 && !memcmp(cs->data, "auto", 4)) {
            chunking = 1;
        } else {
            yyjson_doc *jd = yyjson_read((const char *)cs->data, cs->n, 0);
            yyjson_val *jo = jd ? yyjson_doc_get_root(jd) : NULL;
            yyjson_val *ty = yyjson_obj_get(jo, "type");
            if (!jo || !yyjson_is_obj(jo) || !yyjson_is_str(ty) ||
                strcmp(yyjson_get_str(ty), "server_vad")) {
                yyjson_doc_free(jd);
                return bad(r, "'chunking_strategy' is \"auto\" or an object "
                              "of type server_vad.");
            }
            chunking = 2;
            yyjson_val *t = yyjson_obj_get(jo, "threshold");
            if (yyjson_is_num(t))
                vad_threshold = (float)yyjson_get_num(t);
            t = yyjson_obj_get(jo, "prefix_padding_ms");
            if (yyjson_is_num(t))
                vad_prefix = (int)yyjson_get_num(t);
            t = yyjson_obj_get(jo, "silence_duration_ms");
            if (yyjson_is_num(t))
                vad_silence = (int)yyjson_get_num(t);
            yyjson_doc_free(jd);
            if (vad_threshold < 0 || vad_threshold > 1 || vad_prefix > 60000 ||
                vad_silence < 0 || vad_silence > 60000)
                return bad(r, "'chunking_strategy': a threshold between 0 and "
                              "1, durations in milliseconds.");
        }
    }
    if (gran && !verbose)
        return bad(r, "'timestamp_granularities' needs 'response_format' "
                      "verbose_json.");
    if (!gran)
        gran = 1;
    float temperature = 0;
    if (temp[0]) {
        char *e;
        temperature = strtof(temp, &e);
        if (*e || !(temperature >= 0 && temperature <= 1))
            return bad(r, "'temperature' is between 0 and 1.");
    }
    const struct srv_part *pp = srv_multipart_get(parts, n, "prompt");
    if (pp) {
        if (pp->has_filename || !(prompt = malloc(pp->n + 1)))
            return bad(r, "'prompt' is text.");
        memcpy(prompt, pp->data, pp->n);
        prompt[pp->n] = 0;
    }
    /* the known speakers, for diarized_json: names and data: URLs */
    char *knames[8] = {0};
    unsigned char *krefs[8] = {0};
    size_t kn[8] = {0};
    int nkn = 0, nkr = 0;
    for (int i = 0; i < n; i++) {
        int is_name = !strcmp(parts[i].name, "known_speaker_names[]") ||
                      !strcmp(parts[i].name, "known_speaker_names");
        int is_ref = !strcmp(parts[i].name, "known_speaker_references[]") ||
                     !strcmp(parts[i].name, "known_speaker_references");
        if (!is_name && !is_ref)
            continue;
        if (!diarized || (is_name ? nkn : nkr) == 8) {
            for (int k = 0; k < 8; k++) {
                free(knames[k]);
                free(krefs[k]);
            }
            free(prompt);
            return bad(r, diarized ? "at most 8 known speakers."
                                   : "known speakers need 'response_format' "
                                     "diarized_json.");
        }
        if (is_name) {
            if ((knames[nkn] = malloc(parts[i].n + 1))) {
                memcpy(knames[nkn], parts[i].data, parts[i].n);
                knames[nkn][parts[i].n] = 0;
            }
            nkn++;
        } else {
            char why[160];
            if (srv_data_url((const char *)parts[i].data, parts[i].n, 64u << 20,
                             &krefs[nkr], &kn[nkr], why, sizeof why) != 0) {
                for (int k = 0; k < 8; k++) {
                    free(knames[k]);
                    free(krefs[k]);
                }
                free(prompt);
                return bad(r, why);
            }
            nkr++;
        }
    }
    if (nkn != nkr) {
        for (int k = 0; k < 8; k++) {
            free(knames[k]);
            free(krefs[k]);
        }
        free(prompt);
        return bad(r, "'known_speaker_names[]' and "
                      "'known_speaker_references[]' come in pairs.");
    }
    struct janas_llm_transcribe_params p;
    janas_llm_transcribe_params_default(&p);
    p.language = lang[0] ? lang : NULL;
    p.translate = translate;
    p.prompt = prompt;
    p.temperature = temperature;
    p.words = (gran & 2) != 0;
    p.chunking = chunking;
    p.vad_threshold = vad_threshold;
    p.vad_prefix_ms = vad_prefix;
    p.vad_silence_ms = vad_silence;
    const char *name_p[8];
    const void *ref_p[8];
    int64_t ref_n[8];
    for (int k = 0; k < nkn; k++) {
        name_p[k] = knames[k] ? knames[k] : "";
        ref_p[k] = krefs[k];
        ref_n[k] = (int64_t)kn[k];
    }
    p.diarize = diarized;
    p.n_speakers = nkn;
    p.speaker_names = name_p;
    p.speaker_refs = ref_p;
    p.speaker_ref_bytes = ref_n;
    if (streaming) {
        struct sstream *x = calloc(1, sizeof *x);
        if (!x || !(x->data = malloc(file->n))) {
            free(x);
            free(prompt);
            return srv_reply_error(r, 500, "server_error", NULL,
                                   "out of memory");
        }
        memcpy(x->data, file->data, file->n);
        x->n = file->n;
        x->srv = r->srv;
        x->prompt = prompt;
        snprintf(x->lang, sizeof x->lang, "%s", lang);
        x->p = p;
        x->p.language = x->lang[0] ? x->lang : NULL;
        x->p.prompt = prompt;
        x->p.on_segment = ss_segment;
        x->p.user = x;
        x->p.stop = &x->stop;
        x->logprobs = logprobs;
        x->diarize = diarized;
        for (int k = 0; k < nkn; k++) { /* the stream's own */
            x->names[k] = knames[k];
            x->refs[k] = krefs[k];
            x->name_p[k] = knames[k] ? knames[k] : "";
            x->ref_p[k] = krefs[k];
            x->ref_n[k] = (int64_t)kn[k];
            knames[k] = NULL;
            krefs[k] = NULL;
        }
        x->p.speaker_names = x->name_p;
        x->p.speaker_refs = x->ref_p;
        x->p.speaker_ref_bytes = x->ref_n;
        pthread_mutex_init(&x->mu, NULL);
        pthread_cond_init(&x->cv, NULL);
        if (pthread_create(&x->th, NULL, ss_run, x) != 0) {
            ss_done(x);
            return srv_reply_error(r, 500, "server_error", NULL,
                                   "cannot start the transcription");
        }
        x->started = 1;
        return srv_reply_stream(r, ss_next, ss_done, x);
    }
    pthread_mutex_lock(&r->srv->speech_mu);
    int32_t rc = janas_llm_speech_transcribe(r->srv->speech, file->data,
                                             (int64_t)file->n, &p);
    int ret;
    if (rc != JANAS_LLM_OK) {
        ret = srv_reply_error(r, rc == JANAS_LLM_EINVAL ? 400 : 500,
                              rc == JANAS_LLM_EINVAL ? "invalid_request_error"
                                                     : "server_error",
                              rc == JANAS_LLM_EINVAL ? "invalid_value" : NULL,
                              "%s", janas_llm_last_error());
    } else if (!strcmp(fmt, "text")) {
        char *t = whole_text(r->srv->speech);
        size_t len = t ? strlen(t) : 0;
        if (t) {
            t[len++] = '\n'; /* over its NUL, as OpenAI ends it */
        }
        ret = srv_reply_bytes(r, 200, t, len, "text/plain; charset=utf-8");
    } else if (!strcmp(fmt, "srt") || !strcmp(fmt, "vtt")) {
        ret = reply_cues(r, r->srv->speech, fmt[0] == 'v');
    } else if (diarized) {
        ret = reply_diarized(r, r->srv->speech);
    } else {
        ret = reply_json(r, r->srv->speech, verbose, translate, gran, logprobs);
    }
    pthread_mutex_unlock(&r->srv->speech_mu);
    free(prompt);
    for (int k = 0; k < 8; k++) {
        free(knames[k]);
        free(krefs[k]);
    }
    return ret;
}

int srv_audio_transcriptions(struct srv_req *r)
{
    return speech(r, 0);
}

int srv_audio_translations(struct srv_req *r)
{
    return speech(r, 1);
}
