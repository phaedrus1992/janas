/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
// fma() rounded once where the driver splits it into a multiply and an add
// (RADV before RDNA2: NIR lowers every 32-bit ffma, NoContraction or not;
// issue #48). build.sh puts this after the #extension lines of every
// shader that calls fma(), as its _xfma variant; gpu_fma.c takes those
// variants where its probe finds fma() split.
//
// a*b is exact in a double; s = a*b + c rounded to a double, and err the
// part of the sum s lost (TwoSum); s made odd when it lost something
// (rounding to odd, 53 bits), then rounded to a float: with 53 >= 24 + 2
// bits that is the sum rounded once (Boldo and Melquiond, 2008).

float janas_xfma(float a, float b, float c)
{
    precise double p = double(a) * double(b);
    precise double dc = double(c);
    precise double s = p + dc;
    precise double bv = s - p;
    precise double err = (p - (s - bv)) + (dc - bv);
    if (err != 0.0) {
        uvec2 u = unpackDouble2x32(s);
        if ((u.x & 1u) == 0u) {
            // one step towards err: away from zero if err has s's sign
            if ((err > 0.0) == (s > 0.0))
                u.x += 1u; // even: never carries
            else {
                if (u.x == 0u)
                    u.y -= 1u;
                u.x -= 1u;
            }
            s = packDouble2x32(u);
        }
    }
    return float(s);
}

#define fma janas_xfma
