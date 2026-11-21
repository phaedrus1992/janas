/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * web_search.c - the search behind the web_search tool of the Responses
 * API (see web_search.h): Wikipedia's, the open source of the project's
 * services, in one request of its API - the pages that match, best first,
 * with their first sentences and their addresses.
 */
#include "web_search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

#include "services/common/https_get.h"

/* the query as a URL's part */
static void encode(char *out, size_t cap, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    size_t w = 0;
    for (; *s && w + 4 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
            c == '~') {
            out[w++] = (char)c;
        } else {
            out[w++] = '%';
            out[w++] = hex[c >> 4];
            out[w++] = hex[c & 15];
        }
    }
    out[w] = 0;
}

static int by_index(const void *a, const void *b)
{
    const struct srv_web_hit *x = a, *y = b;
    return x->index - y->index;
}

/* A language as a Wikipedia's code: a code (it, en, zh-yue), or the name
   of one of the languages models write most; NULL when neither. */
static const char *code_of(const char *lang)
{
    static const char *const names[][2] = {
        {"english", "en"},    {"italian", "it"},   {"italiano", "it"},
        {"french", "fr"},     {"german", "de"},    {"spanish", "es"},
        {"portuguese", "pt"}, {"dutch", "nl"},     {"russian", "ru"},
        {"chinese", "zh"},    {"japanese", "ja"},  {"korean", "ko"},
        {"arabic", "ar"},     {"polish", "pl"},    {"swedish", "sv"},
        {"turkish", "tr"},    {"ukrainian", "uk"}, {"sardinian", "sc"},
    };
    static _Thread_local char buf[16];
    if (!lang)
        return NULL;
    size_t n = strlen(lang), i;
    if (n >= sizeof buf)
        return NULL;
    for (i = 0; i < n; i++)
        buf[i] =
            (char)(lang[i] >= 'A' && lang[i] <= 'Z' ? lang[i] + 32 : lang[i]);
    buf[n] = 0;
    for (i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        if (!strcmp(buf, names[i][0]))
            return names[i][1];
    /* aa, aaa, aa-bbb: letters, at most one dash */
    size_t head = strspn(buf, "abcdefghijklmnopqrstuvwxyz");
    if ((head == 2 || head == 3) &&
        (buf[head] == 0 ||
         (buf[head] == '-' && strlen(buf + head + 1) >= 2 &&
          strspn(buf + head + 1, "abcdefghijklmnopqrstuvwxyz") ==
              strlen(buf + head + 1))))
        return buf;
    return NULL;
}

static int search_in(const char *query, const char *lg,
                     struct srv_web_hit *hits, int max, char *err,
                     size_t err_len);

int srv_web_search(const char *query, const char *lang,
                   struct srv_web_hit *hits, int max, char *err, size_t err_len)
{
    const char *lg = code_of(lang);
    int n = lg && strcmp(lg, "en") != 0
                ? search_in(query, lg, hits, max, err, err_len)
                : -1;
    if (n <= 0) /* none asked, or nothing there: the English one */
        n = search_in(query, "en", hits, max, err, err_len);
    return n;
}

static int search_in(const char *query, const char *lg,
                     struct srv_web_hit *hits, int max, char *err,
                     size_t err_len)
{
    char q[1024], url[2048];
    encode(q, sizeof q, query);
    snprintf(url, sizeof url,
             "https://%s.wikipedia.org/w/api.php?action=query&format=json&"
             "formatversion=2&generator=search&gsrsearch=%s&gsrlimit=%d&"
             "prop=extracts%%7Cinfo&exintro=1&explaintext=1&exsentences=6&"
             "exlimit=max&inprop=url&redirects=1",
             lg, q, max);
    const char *hdr[] = {"User-Agent: janas-server/" JANAS_VERSION
                         " (https://github.com/prabanta-dev/janas)"};
    struct janas_buf body = {0};
    int st = janas_https_get(url, hdr, 1, &body, err, err_len);
    if (st != 200) {
        if (st > 0)
            snprintf(err, err_len, "Wikipedia answered HTTP %d", st);
        janas_buf_free(&body);
        return -1;
    }
    yyjson_doc *d = yyjson_read(body.p, body.n, 0);
    janas_buf_free(&body);
    yyjson_val *pages = yyjson_obj_get(
        yyjson_obj_get(yyjson_doc_get_root(d), "query"), "pages");
    int n = 0;
    size_t idx, mx;
    yyjson_val *p;
    yyjson_arr_foreach(pages, idx, mx, p)
    {
        if (n == max)
            break;
        const char *title = yyjson_get_str(yyjson_obj_get(p, "title"));
        const char *u = yyjson_get_str(yyjson_obj_get(p, "fullurl"));
        const char *x = yyjson_get_str(yyjson_obj_get(p, "extract"));
        if (!title || !u)
            continue;
        struct srv_web_hit *h = &hits[n++];
        snprintf(h->title, sizeof h->title, "%s", title);
        snprintf(h->url, sizeof h->url, "%s", u);
        snprintf(h->text, sizeof h->text, "%s", x ? x : "");
        yyjson_val *ix = yyjson_obj_get(p, "index");
        h->index = yyjson_is_int(ix) ? (int)yyjson_get_int(ix) : 1000 + n;
    }
    yyjson_doc_free(d);
    if (n)
        qsort(hits, (size_t)n, sizeof(*hits), by_index);
    return n;
}
