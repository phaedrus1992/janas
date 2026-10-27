/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * devices.c - the devices of a kind of pass, as written (devices.h).
 */
#include "devices.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const janas_devices_options[JANAS_DEVICES_OPTIONS] = {
    "--prompt-on", "--drafts-on", "--reply-on"};
const char *const janas_devices_vars[JANAS_DEVICES_OPTIONS] = {
    "JANAS_PROMPT_ON", "JANAS_DRAFTS_ON", "JANAS_REPLY_ON"};

int janas_devices_mask(const char *spec, char *why, size_t len)
{
    if (!spec || !*spec || !strcmp(spec, "auto"))
        return 0;
    int mask = 0;
    const char *p = spec;
    for (;;) {
        /* one alternative: words joined by '+' */
        int combo = 0;
        for (;;) {
            char w[8];
            size_t n = 0;
            while (p[n] && p[n] != '+' && p[n] != ',')
                n++;
            if (n == 0 || n >= sizeof w) {
                snprintf(why, len,
                         "\"%s\": an empty or unknown device (cpu, gpu, npu, "
                         "joined by + or apart by commas, or auto)",
                         spec);
                return -1;
            }
            for (size_t i = 0; i < n; i++)
                w[i] = (char)tolower((unsigned char)p[i]);
            w[n] = 0;
            if (!strcmp(w, "gpu"))
                combo |= JANAS_DEV_GPU;
            else if (!strcmp(w, "npu"))
                combo |= JANAS_DEV_NPU;
            else if (strcmp(w, "cpu")) {
                snprintf(why, len,
                         "\"%s\": \"%s\" is not a device (cpu, gpu, npu)", spec,
                         w);
                return -1;
            }
            p += n;
            if (*p != '+')
                break;
            p++;
        }
        mask |= 1 << combo;
        if (!*p)
            return mask;
        p++; /* the comma: another alternative follows, or the error */
    }
}

void janas_devices_name(int mask, char *buf, size_t len)
{
    static const char *name[JANAS_DEV_COMBOS] = {"cpu", "gpu", "npu",
                                                 "npu+gpu"};
    size_t o = 0;
    buf[0] = 0;
    if (!mask) {
        snprintf(buf, len, "auto");
        return;
    }
    for (int c = 0; c < JANAS_DEV_COMBOS && o < len; c++)
        if (mask >> c & 1)
            o += (size_t)snprintf(buf + o, len - o, "%s%s", o ? "," : "",
                                  name[c]);
}

int janas_devices_option(const char *option, const char *spec, char *why,
                         size_t len)
{
    for (int k = 0; k < JANAS_DEVICES_OPTIONS; k++) {
        if (strcmp(option, janas_devices_options[k]))
            continue;
        char r[200];
        if (!spec) {
            snprintf(why, len, "%s wants a value", option);
            return -1;
        }
        if (janas_devices_mask(spec, r, sizeof r) < 0) {
            snprintf(why, len, "%s: %s", option, r);
            return -1;
        }
        setenv(janas_devices_vars[k], spec, 1);
        return k + 1;
    }
    return 0;
}
