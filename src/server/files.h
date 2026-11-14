/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * files.h - a file as OpenAI's API writes it (files.c), for the APIs that
 * hand files back: uploads, batches.
 */
#ifndef JANAS_SERVER_FILES_H
#define JANAS_SERVER_FILES_H

#include <yyjson.h>

#include "files_store.h"

/* The File object of x, made in d. */
yyjson_mut_val *srv_file_json(yyjson_mut_doc *d, const struct srv_file *x);

#endif
