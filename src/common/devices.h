/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * devices.h - the devices a kind of pass may run on, as the user writes
 * them (--prompt-on, --drafts-on, --reply-on; JANAS_PROMPT_ON,
 * JANAS_DRAFTS_ON, JANAS_REPLY_ON): "auto", or alternatives apart by
 * commas, each the devices that work together joined by '+' - "npu",
 * "npu,gpu" (whichever measures faster), "npu+gpu" (both in the same
 * pass), "cpu" (no accelerator). The CPU takes part in every pass.
 */
#ifndef JANAS_COMMON_DEVICES_H
#define JANAS_COMMON_DEVICES_H

#include <stddef.h>

/* a combination of accelerators: their bits; 0 the CPU alone */
enum { JANAS_DEV_GPU = 1, JANAS_DEV_NPU = 2, JANAS_DEV_COMBOS = 4 };

/* The combinations spec allows, bit c for combination c: 0 for "auto" (or
   NULL, empty: any), -1 with the reason in why if it does not read. */
int janas_devices_mask(const char *spec, char *why, size_t len);

/* A mask as the user would write it ("npu,npu+gpu"; "auto" for 0). */
void janas_devices_name(int mask, char *buf, size_t len);

/* The programs' options for them (janas-chat, janas-server, janas-bench),
   in the order of their variables: prompts, drafts, replies. */
#define JANAS_DEVICES_OPTIONS 3
extern const char *const janas_devices_options[JANAS_DEVICES_OPTIONS];
extern const char *const janas_devices_vars[JANAS_DEVICES_OPTIONS];

/* option one of them: spec checked and put in its variable, read when a
   model opens, and its index + 1; another option: 0; a spec that does not
   read: -1, the reason in why. */
int janas_devices_option(const char *option, const char *spec, char *why,
                         size_t len);

#endif
