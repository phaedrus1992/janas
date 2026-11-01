/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * evals_metrics.h - the text similarity metrics of the evals' graders
 * (evals_metrics.c).
 */
#ifndef JANAS_SERVER_EVALS_METRICS_H
#define JANAS_SERVER_EVALS_METRICS_H

/* The score, 0 to 1, of input against reference by metric: cosine,
   fuzzy_match, bleu, gleu, meteor, rouge_1 to rouge_5, rouge_l. */
double srv_eval_similarity(const char *metric, const char *input,
                           const char *reference);
/* 1 when metric is one of them. */
int srv_eval_metric_ok(const char *metric);

#endif
