/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * web_search.h - the search behind the Responses API's web_search tool
 * (web_search.c): Wikipedia, in the language asked.
 */
#ifndef JANAS_SERVER_WEB_SEARCH_H
#define JANAS_SERVER_WEB_SEARCH_H

#include <stddef.h>

struct srv_web_hit {
    char title[256];
    char url[512];
    char text[2048]; /* the page's first sentences */
    int index;       /* its rank */
};

/* The pages of Wikipedia (lang: "en", "it"...; NULL or not a code: "en")
   that match query, best first, at most max: how many, or -1 with the
   reason in err. */
int srv_web_search(const char *query, const char *lang,
                   struct srv_web_hit *hits, int max, char *err,
                   size_t err_len);

#endif
