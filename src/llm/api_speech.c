/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_speech.c - speech to text in the public API (janas/llm.h): a Whisper
 * model (whisper.h) and the last transcription it made.
 */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "api_speech.h"
#include "chat.h"
#include "common/audio.h"
#include "tokenizer.h"

char *janas_speech_beside(const char *path, const char *file)
{
    const char *slash = strrchr(path, '/');
    size_t dl = slash ? (size_t)(slash - path) + 1 : 0, fl = strlen(file);
    char *p = malloc(dl + fl + 1);
    if (!p)
        return NULL;
    memcpy(p, path, dl);
    memcpy(p + dl, file, fl + 1);
    if (access(p, R_OK) != 0) {
        free(p);
        return NULL;
    }
    return p;
}

void janas_llm_speech_params_default(struct janas_llm_speech_params *p)
{
    if (!p)
        return;
    memset(p, 0, sizeof *p);
    p->size = sizeof *p;
}

int32_t janas_llm_speech_open(const char *path,
                              const struct janas_llm_speech_params *p,
                              janas_llm_speech **out)
{
    if (!path || !out || (p && p->size != sizeof *p))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    *out = NULL;
    janas_llm_speech *s = calloc(1, sizeof *s);
    if (!s)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    char err[512];
    s->w = janas_whisper_open(path, p ? p->threads : 0, err, sizeof err);
    if (!s->w) {
        free(s);
        return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
    }
    /* the voice activity detector: named, or beside the model */
    if (p && p->vad_model) {
        s->vad = janas_vad_open(p->vad_model, err, sizeof err);
        if (!s->vad) {
            janas_llm_speech_close(s);
            return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
        }
    } else {
        char *vp = janas_speech_beside(path, "silero-vad.jns");
        if (vp)
            s->vad = janas_vad_open(vp, err, sizeof err);
        free(vp);
    }
    /* the speaker model for diarization: named, or beside the model */
    if (p && p->speaker_model) {
        s->speaker =
            janas_speaker_open(p->speaker_model, p->threads, err, sizeof err);
        if (!s->speaker) {
            janas_llm_speech_close(s);
            return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
        }
    } else {
        char *sp = janas_speech_beside(path, "wespeaker-resnet34.jns");
        if (sp)
            s->speaker =
                janas_speaker_open(sp, p ? p->threads : 0, err, sizeof err);
        free(sp);
    }
    *out = s;
    return JANAS_LLM_OK;
}

void janas_llm_speech_close(janas_llm_speech *s)
{
    if (!s)
        return;
    janas_whisper_result_free(&s->r);
    janas_speech_turns_free(s);
    janas_whisper_close(s->w);
    janas_vad_close(s->vad);
    janas_speaker_close(s->speaker);
    free(s);
}

void janas_llm_transcribe_params_default(struct janas_llm_transcribe_params *p)
{
    if (!p)
        return;
    memset(p, 0, sizeof *p);
    p->size = sizeof *p;
    p->vad_threshold = 0.5f;
    p->vad_prefix_ms = 300;
    p->vad_silence_ms = 200;
}

struct cb {
    void (*fn)(void *user, int32_t segment);
    void *user;
};

static void on_segment(void *user, uint32_t index)
{
    const struct cb *c = user;
    c->fn(c->user, (int32_t)index);
}

int32_t janas_llm_speech_transcribe(janas_llm_speech *s, const void *data,
                                    int64_t n,
                                    const struct janas_llm_transcribe_params *p)
{
    if (!s || !data || n <= 0 || (p && p->size != sizeof *p) ||
        (p && (p->temperature < 0 || p->temperature > 1)))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_whisper_result_free(&s->r);
    struct janas_audio a;
    char err[512];
    if (janas_audio_decode(data, (size_t)n, &a, err, sizeof err) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "%s", err);
    if (janas_audio_resample(&a, s->w->rate) != 0) {
        janas_audio_free(&a);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    struct janas_whisper_opts o = {0};
    struct cb cb = {0};
    if (p) {
        o.language = p->language;
        o.translate = p->translate != 0;
        o.prompt = p->prompt;
        o.temperature = p->temperature;
        o.seed = p->seed;
        o.words = p->words != 0 || p->diarize != 0;
        if (p->on_segment) {
            o.on_segment = on_segment;
            o.user = &cb;
            cb.fn = p->on_segment;
            cb.user = p->user;
        }
        o.stop = (volatile int *)p->stop;
    }
    if (o.language && o.language[0] &&
        janas_whisper_lang_index(s->w, o.language) < 0 && s->w->multilingual) {
        janas_audio_free(&a);
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "a language this model does not know: %s",
                              o.language);
    }
    /* chunking: the stretches of speech the detector finds */
    size_t *regions = NULL;
    if (p && p->chunking) {
        if (!s->vad) {
            janas_audio_free(&a);
            return janas_api_fail(JANAS_LLM_EINVAL,
                                  "chunking needs a voice activity detector: "
                                  "silero-vad.jns (made by silero2jns) beside "
                                  "the speech model, or vad_model");
        }
        struct janas_vad_params vp = {.threshold = 0.5f,
                                      .min_speech_ms = 250,
                                      .min_silence_ms = 200,
                                      .speech_pad_ms = 300};
        float *src = a.pcm, *norm = NULL;
        if (p->chunking == 2) {
            if (p->vad_threshold > 0)
                vp.threshold = p->vad_threshold;
            if (p->vad_silence_ms > 0)
                vp.min_silence_ms = (uint32_t)p->vad_silence_ms;
            if (p->vad_prefix_ms >= 0)
                vp.speech_pad_ms = (uint32_t)p->vad_prefix_ms;
        } else if ((norm = malloc(sizeof(float) * (a.n ? a.n : 1)))) {
            /* auto: the loudness normalized for the detector, its peak at
               0.9 */
            float peak = 0;
            for (size_t i = 0; i < a.n; i++)
                peak = fabsf(a.pcm[i]) > peak ? fabsf(a.pcm[i]) : peak;
            float g = peak > 0 ? 0.9f / peak : 1;
            for (size_t i = 0; i < a.n; i++)
                norm[i] = a.pcm[i] * g;
            src = norm;
        }
        float *probs = NULL;
        long np = janas_vad_probs(s->vad, src, a.n, &probs);
        long ns =
            np < 0 ? -1
                   : janas_vad_segments(s->vad, probs, np, a.n, &vp, &regions);
        free(probs);
        free(norm);
        if (ns < 0) {
            janas_audio_free(&a);
            return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
        }
        static const size_t NONE[2] = {0, 0};
        o.regions = ns ? regions : NONE;
        o.n_regions = ns ? (size_t)ns : 1;
    }
    janas_speech_turns_free(s);
    if (p && p->diarize && !s->speaker) {
        free(regions);
        janas_audio_free(&a);
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "diarization needs a speaker model: "
                              "wespeaker-resnet34.jns (made by wespeaker2jns) "
                              "beside the speech model, or speaker_model");
    }
    int rc =
        janas_whisper_transcribe(s->w, a.pcm, a.n, &o, &s->r, err, sizeof err);
    if (rc == 0 && p && p->diarize &&
        janas_speech_diarize(s, a.pcm, a.n, p->chunking ? o.regions : NULL,
                             p->chunking ? o.n_regions : 0, p, err,
                             sizeof err) != 0)
        rc = -1;
    free(regions);
    janas_audio_free(&a);
    if (rc != 0)
        return janas_api_fail(strstr(err, "memory")         ? JANAS_LLM_ENOMEM
                              : strstr(err, "English only") ? JANAS_LLM_EINVAL
                                                            : JANAS_LLM_EFAIL,
                              "%s", err);
    return JANAS_LLM_OK;
}

int32_t janas_llm_speech_text(const janas_llm_speech *s, char *buf, int32_t cap,
                              int32_t *len)
{
    if (!s)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    const char *t = s->r.text ? s->r.text : "";
    return janas_api_copy_out(t, strlen(t), buf, cap, len);
}

int32_t janas_llm_speech_language(const janas_llm_speech *s, char *buf,
                                  int32_t cap, int32_t *len)
{
    if (!s)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    return janas_api_copy_out(s->r.language, strlen(s->r.language), buf, cap,
                              len);
}

const char *janas_llm_speech_language_name(const char *code)
{
    return code ? janas_whisper_lang_name(code) : NULL;
}

double janas_llm_speech_duration(const janas_llm_speech *s)
{
    return s ? s->r.duration : 0;
}

int32_t janas_llm_speech_segments(const janas_llm_speech *s)
{
    return s ? (int32_t)s->r.n_segs : 0;
}

int32_t janas_llm_speech_segment(const janas_llm_speech *s, int32_t i,
                                 double *start, double *end, char *buf,
                                 int32_t cap, int32_t *len)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (start)
        *start = g->t0;
    if (end)
        *end = g->t1;
    if (!buf && !len)
        return JANAS_LLM_OK;
    return janas_api_copy_out(g->text, strlen(g->text), buf, cap, len);
}

int32_t janas_llm_speech_segment_info(const janas_llm_speech *s, int32_t i,
                                      float *avg_logprob, float *no_speech_prob,
                                      float *temperature,
                                      float *compression_ratio, int32_t *ids,
                                      int32_t max, int32_t *n)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs || max < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (avg_logprob)
        *avg_logprob = g->avg_logprob;
    if (no_speech_prob)
        *no_speech_prob = g->no_speech_prob;
    if (temperature)
        *temperature = g->temperature;
    if (compression_ratio)
        *compression_ratio = g->compression_ratio;
    if (n)
        *n = (int32_t)g->n_tokens;
    if (ids)
        memcpy(ids, g->tokens,
               sizeof(int32_t) *
                   ((uint32_t)max < g->n_tokens ? (uint32_t)max : g->n_tokens));
    return ids && (uint32_t)max < g->n_tokens
               ? janas_api_fail(JANAS_LLM_ESMALL, "buffer too small")
               : JANAS_LLM_OK;
}

int32_t janas_llm_speech_words(const janas_llm_speech *s)
{
    if (!s)
        return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->r.n_segs; i++)
        n += s->r.segs[i].n_words;
    return (int32_t)n;
}

int32_t janas_llm_speech_word(const janas_llm_speech *s, int32_t i,
                              double *start, double *end, float *probability,
                              char *buf, int32_t cap, int32_t *len)
{
    if (!s || i < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such word");
    uint32_t k = (uint32_t)i;
    for (uint32_t g = 0; g < s->r.n_segs; g++) {
        const struct janas_whisper_seg *sg = &s->r.segs[g];
        if (k >= sg->n_words) {
            k -= sg->n_words;
            continue;
        }
        const struct janas_whisper_word *w = &sg->words[k];
        if (start)
            *start = w->t0;
        if (end)
            *end = w->t1;
        if (probability)
            *probability = w->probability;
        if (!buf && !len)
            return JANAS_LLM_OK;
        return janas_api_copy_out(w->text, strlen(w->text), buf, cap, len);
    }
    return janas_api_fail(JANAS_LLM_EINVAL, "no such word");
}

int32_t janas_llm_speech_segment_logprobs(const janas_llm_speech *s, int32_t i,
                                          float *lp, int32_t max, int32_t *n)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs || max < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (n)
        *n = (int32_t)g->n_tokens;
    if (lp)
        memcpy(lp, g->logprobs,
               sizeof(float) *
                   ((uint32_t)max < g->n_tokens ? (uint32_t)max : g->n_tokens));
    return lp && (uint32_t)max < g->n_tokens
               ? janas_api_fail(JANAS_LLM_ESMALL, "buffer too small")
               : JANAS_LLM_OK;
}

int32_t janas_llm_speech_token(const janas_llm_speech *s, int32_t id, char *buf,
                               int32_t cap, int32_t *len)
{
    if (!s || id < 0 || (uint32_t)id >= s->w->vocab)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such token");
    char tmp[256];
    size_t k = janas_tokenizer_decode(s->w->tk, id, tmp, sizeof tmp);
    if (k > sizeof tmp)
        k = sizeof tmp;
    return janas_api_copy_out(tmp, k, buf, cap, len);
}

int32_t janas_llm_speech_token_is_text(const janas_llm_speech *s, int32_t id)
{
    return s && id >= 0 && id < s->w->eot;
}
