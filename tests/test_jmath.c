/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_jmath.c - the engine's own exp, log, pow, sin, cos and tanh against
 * the C library's double functions, rounded to float: within one unit in
 * the last place over the ranges the engine uses them on (the softmax and
 * sigmoid arguments, the DeltaNet's softplus, the rotary frequencies and
 * angles up to a million positions, the soft cap), and the double exp and
 * log within a few units of the double's last place.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "llm/jmath.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            if (failures < 20) {                                               \
                printf(__VA_ARGS__);                                           \
                printf("\n");                                                  \
            }                                                                  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* distance in units in the last place of two finite floats */
static int64_t ulps(float a, float b)
{
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    if (ia < 0)
        ia = INT32_MIN - ia;
    if (ib < 0)
        ib = INT32_MIN - ib;
    int64_t d = (int64_t)ia - (int64_t)ib;
    return d < 0 ? -d : d;
}

/* one ulp, or an absolute 1e-12 where the result is near zero (a sine
   at a multiple of pi: the reference has no more digits there either) */
static int close_enough(float got, double want)
{
    float w = (float)want;
    return ulps(got, w) <= 1 || fabs((double)got - want) <= 1e-12;
}

int main(void)
{
    alarm(120);
    int64_t worst;

    worst = 0;
    for (int i = -1030000; i <= 885000; i++) {
        float x = (float)i * 1e-4f;
        float got = janas_expf(x);
        CHECK(close_enough(got, exp((double)x)), "expf(%.9g) = %.9g, want %.9g",
              x, got, exp((double)x));
        int64_t u = ulps(got, (float)exp((double)x));
        if (u > worst)
            worst = u;
    }
    CHECK(janas_expf(-200.0f) == 0.0f && isinf(janas_expf(200.0f)),
          "expf out of range");

    for (int i = -100000; i <= 100000; i++) {
        double x = (double)i * 7.1e-3;
        double got = janas_exp(x), want = exp(x);
        /* an infinity must be one, a subnormal (below e^-708, where the
           engine never goes) is rounded twice: a few of its units */
        CHECK(isinf(want) ? got == want
                          : fabs(got - want) <= 4e-16 * want + 1e-321,
              "exp(%.17g) = %.17g, want %.17g", x, got, want);
    }
    for (int i = 1; i <= 200000; i++) {
        double x = (double)i * (double)i * 1.37e-3;
        double got = janas_log(x), want = log(x);
        CHECK(fabs(got - want) <= 4e-16 * fabs(want) + 1e-300,
              "log(%.17g) = %.17g, want %.17g", x, got, want);
    }
    CHECK(fabs(janas_log(1e-310) - log(1e-310)) <= 1e-13, "log of a subnormal");
    CHECK(janas_log(1.0) == 0.0, "log(1) = %.17g", janas_log(1.0));

    for (int i = -1000000; i <= 200000; i++) {
        float x = (float)i * 1e-4f;
        double want = log1p(exp((double)x));
        CHECK(close_enough(janas_softplusf(x), want),
              "softplus(%.9g) = %.9g, want %.9g", x, janas_softplusf(x), want);
    }

    static const float bases[] = {10000.0f, 500000.0f, 1000000.0f, 5000000.0f,
                                  10000000.0f};
    for (int b = 0; b < 5; b++)
        for (int i = 0; i < 512; i++) {
            float t = -2.0f * (float)i / 512.0f;
            double want = pow((double)bases[b], (double)t);
            CHECK(close_enough(janas_powf(bases[b], t), want),
                  "pow(%.9g, %.9g) = %.9g, want %.9g", bases[b], t,
                  janas_powf(bases[b], t), want);
        }

    for (int i = 0; i < 2000000; i++) {
        /* angles as the rotary tables make them, and some far larger */
        float x =
            i < 1000000 ? (float)i * 1.0001f : (float)(i - 1000000) * 997.13f;
        if (i & 1)
            x = -x;
        float s, c;
        janas_sincosf(x, &s, &c);
        /* far angles: the reference may be off itself - Windows's C library
           reduces them with fewer digits of pi, 1.2e-12 off at 9.4e8 where
           ours agrees with the exact value - so a margin of |x| 2^-68 */
        double far = fabs((double)x) * 0x1p-68;
        CHECK(close_enough(s, sin((double)x)) ||
                  fabs((double)s - sin((double)x)) <= far,
              "sin(%.9g) = %.9g, want %.9g", x, s, sin((double)x));
        CHECK(close_enough(c, cos((double)x)) ||
                  fabs((double)c - cos((double)x)) <= far,
              "cos(%.9g) = %.9g, want %.9g", x, c, cos((double)x));
    }
    for (int i = -60000; i <= 60000; i++) {
        float x = (float)i * 5e-4f;
        CHECK(close_enough(janas_tanhf(x), tanh((double)x)),
              "tanh(%.9g) = %.9g, want %.9g", x, janas_tanhf(x),
              tanh((double)x));
    }

    if (failures) {
        printf("test_jmath: %d failures\n", failures);
        return 1;
    }
    printf("test_jmath: ok (expf within %lld ulp)\n", (long long)worst);
    return 0;
}
