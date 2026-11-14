/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jmath.h - the engine's own exp, log, pow, sin, cos and tanh (private).
 *
 * The C library's functions are not the same bits everywhere: glibc 2.41
 * rounds log1pf, tanhf and others correctly where 2.39 did not, and
 * Windows's are other code again. One unit in the last place of a rotary
 * angle or a DeltaNet decay is enough to turn a near tie between two
 * tokens the other way, and the answer at temperature 0 changed with the
 * machine's C library (issue #38). These are computed in double with plain
 * IEEE-754 operations only - add, subtract, multiply, divide, bit moves -
 * in a file of their own, compiled without FMA, so that every x86-64 and
 * arm64 machine and every compiler gives the same bits. The float functions
 * are the double result rounded once: within a unit in the last place,
 * nearly always the correctly rounded value.
 */
#ifndef JANAS_LLM_JMATH_H
#define JANAS_LLM_JMATH_H

/* e^x; 0 below -745.2, infinity above 709.8 */
double janas_exp(double x);

/* natural logarithm; -infinity at 0, NaN below 0 */
double janas_log(double x);

float janas_expf(float x);

/* log(1 + e^x), the softplus */
float janas_softplusf(float x);

/* b^t for b > 0 (the rotary frequencies) */
float janas_powf(float b, float t);

/* sin and cos of x together, for |x| up to about 1.7e12 (a million
   positions make angles up to 1e6) */
void janas_sincosf(float x, float *s, float *c);

float janas_tanhf(float x);

#endif
