/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jmath.c - the engine's own exp, log, pow, sin, cos and tanh (see jmath.h).
 *
 * Built with the project's common flags, which never enable FMA and turn
 * contraction off (-ffp-contract=off), so that no compiler fuses a multiply
 * and an add here; nothing in this file may carry a target attribute. Every
 * polynomial is a Taylor series cut where the next term is below 1e-17 of the
 * result on the reduced range.
 */
#include "jmath.h"

#include <stdint.h>
#include <string.h>

/* ln 2 and pi/2 in parts whose leading ones have their low bits zero, so
   that k times them is exact for the k used here (fdlibm's constants) */
static const double LN2_HI = 6.93147180369123816490e-01; /* 32 bits */
static const double LN2_LO = 1.90821492927058770002e-10;
static const double INV_LN2 = 1.44269504088896338700e+00;
static const double PIO2_1 = 1.57079632673412561417e+00; /* 33 bits */
static const double PIO2_2 = 6.07710050630396597660e-11; /* 33 bits */
static const double PIO2_2T = 2.02226624879595063154e-21;
static const double INV_PIO2 = 6.36619772367581382433e-01;
/* round to the nearest integer, ties to even, for |x| < 2^51 */
static const double ROUNDER = 0x1.8p52;

static double from_bits(uint64_t u)
{
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

static uint64_t to_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

/* 2^k for -1022 <= k <= 1023 */
static double pow2i(int k)
{
    return from_bits((uint64_t)(k + 1023) << 52);
}

double janas_exp(double x)
{
    if (x != x)
        return x;
    if (x > 709.782712893384)
        return from_bits(0x7ff0000000000000ull);
    if (x < -745.1332191019412)
        return 0.0;
    double kd = (x * INV_LN2 + ROUNDER) - ROUNDER;
    int k = (int)kd;
    double r = (x - kd * LN2_HI) - kd * LN2_LO; /* |r| <= 0.347 */
    double p =
        1.0 / 6227020800.0; /* 1/13!, then Horner down to the constant 1 */
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    /* 2^k in two halves, each a normal number: k runs from -1075 to 1024 */
    int k1 = k / 2;
    return p * pow2i(k1) * pow2i(k - k1);
}

double janas_log(double x)
{
    if (x != x || x < 0.0)
        return from_bits(0x7ff8000000000000ull);
    if (x == 0.0)
        return -from_bits(0x7ff0000000000000ull);
    uint64_t u = to_bits(x);
    if (u >= 0x7ff0000000000000ull)
        return x; /* infinity */
    int e = 0;
    if (u < 0x0010000000000000ull) { /* subnormal */
        u = to_bits(x * 0x1p54);
        e = -54;
    }
    e += (int)(u >> 52) - 1023;
    double m = from_bits((u & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
    if (m > 1.4142135623730951) { /* m in [sqrt(1/2), sqrt(2)] */
        m *= 0.5;
        e++;
    }
    /* log m = 2 atanh(s), s = (m - 1)/(m + 1), |s| <= 0.1716 */
    double s = (m - 1.0) / (m + 1.0), s2 = s * s;
    double p = 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    double lm = 2.0 * s + 2.0 * s * s2 * p;
    double ed = (double)e;
    return ed * LN2_HI + (lm + ed * LN2_LO);
}

float janas_expf(float x)
{
    return (float)janas_exp((double)x);
}

float janas_softplusf(float x)
{
    double y = janas_exp((double)x);
    if (y < 0x1p-20) /* log(1 + y) = y - y^2/2 + y^3/3 - ... */
        return (float)(y - y * y * 0.5 + y * y * y * (1.0 / 3.0));
    return (float)janas_log(1.0 + y);
}

float janas_powf(float b, float t)
{
    return (float)janas_exp((double)t * janas_log((double)b));
}

/* sin r and cos r for |r| <= pi/4 */
static void sincos_kernel(double r, double *s, double *c)
{
    double r2 = r * r;
    double ps = -1.0 / 121645100408832000.0; /* -1/19! */
    ps = ps * r2 + 1.0 / 355687428096000.0;
    ps = ps * r2 - 1.0 / 1307674368000.0;
    ps = ps * r2 + 1.0 / 6227020800.0;
    ps = ps * r2 - 1.0 / 39916800.0;
    ps = ps * r2 + 1.0 / 362880.0;
    ps = ps * r2 - 1.0 / 5040.0;
    ps = ps * r2 + 1.0 / 120.0;
    ps = ps * r2 - 1.0 / 6.0;
    *s = r + r * r2 * ps;
    double pc = 1.0 / 2432902008176640000.0; /* 1/20! */
    pc = pc * r2 - 1.0 / 6402373705728000.0;
    pc = pc * r2 + 1.0 / 20922789888000.0;
    pc = pc * r2 - 1.0 / 87178291200.0;
    pc = pc * r2 + 1.0 / 479001600.0;
    pc = pc * r2 - 1.0 / 3628800.0;
    pc = pc * r2 + 1.0 / 40320.0;
    pc = pc * r2 - 1.0 / 720.0;
    pc = pc * r2 + 1.0 / 24.0;
    pc = pc * r2 - 0.5;
    *c = 1.0 + r2 * pc;
}

void janas_sincosf(float xf, float *s, float *c)
{
    double x = (double)xf;
    if (x != x || x - x != 0.0) { /* NaN or infinity */
        *s = *c = (float)(x - x);
        return;
    }
    /* whole turns first, by 2^20 quarter turns (2^18 turns), then quarter
       turns, with the leading part of pi/2 only: x float, |j| < 2^20 and
       |k| < 2^20 make both subtractions exact; the rest of pi/2 times all
       the quarter turns, m, is a small number (m * PIO2_2 under 0.04 for
       |x| up to 1e9) taken off last, in that order */
    double j = (x * (INV_PIO2 / 0x1p20) + ROUNDER) - ROUNDER;
    double y = x - j * (PIO2_1 * 0x1p20);
    double kd = (y * INV_PIO2 + ROUNDER) - ROUNDER;
    double m = j * 0x1p20 + kd;
    double r = ((y - kd * PIO2_1) - m * PIO2_2) - m * PIO2_2T;
    double sr, cr;
    sincos_kernel(r, &sr, &cr);
    switch ((int64_t)kd & 3) {
    case 0:
        *s = (float)sr;
        *c = (float)cr;
        break;
    case 1:
        *s = (float)cr;
        *c = (float)-sr;
        break;
    case 2:
        *s = (float)-sr;
        *c = (float)-cr;
        break;
    default:
        *s = (float)-cr;
        *c = (float)sr;
        break;
    }
}

float janas_tanhf(float xf)
{
    double x = (double)xf, a = x < 0 ? -x : x, t;
    if (x != x)
        return xf;
    if (a > 20.0)
        t = 1.0;
    else if (a < 0.0625) /* x - x^3/3 + 2x^5/15 - 17x^7/315 */
        t = a - a * a * a *
                    (1.0 / 3.0 - a * a * (2.0 / 15.0 - a * a * (17.0 / 315.0)));
    else
        t = 1.0 - 2.0 / (janas_exp(2.0 * a) + 1.0);
    return (float)(x < 0 ? -t : t);
}
