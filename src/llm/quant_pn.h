/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_pn.h - the down by neuron in integers (janas_q6kpn_chunk, quant.h),
 * private to quant_q6k.c, which includes it once per instruction set. The
 * includer defines:
 *   PN(name)          the name with the instruction-set suffix
 *   PN_TARGET         the function attribute selecting the instructions
 *   PN_DOT(acc, u, s) acc + the unsigned bytes u times the signed bytes s,
 *                     summed four by four into int32 lanes: maddubs, madd
 *                     and add on AVX2 (the pairs stay under 2 x 127 x 127,
 *                     never saturating), a single vpdpbusd on AVX-VNNI
 * The sums are exact integers and the float operations the reference's,
 * in its order: every variant gives janas_q6kpn_chunk_ref's bits.
 *
 * A half of the block at a time, each neuron's values on the scale of
 * 0..127 whose 64 is zero (q6kp_value2) are unpacked once into a buffer -
 * one branch on the level a neuron, none inside - then the neurons go four
 * at a time, interleaved so that a 32-bit lane holds the four neurons'
 * values of one output, the four neurons' factors of that output's group
 * of 16 beside them. Unpacking within 128-bit lanes leaves outputs 0-3 and
 * 16-19 in the first register, 4-7 and 20-23 in the second, and so on: the
 * low lanes are one group, the high lanes the next, and the sums are put
 * back in order at the end.
 */

/* Neuron i's values in half nh of the block, 128 bytes, at its level. A
   part's two bits sit at bits 2p and 2p + 1 of the planes' bytes: masked
   in place, then shifted to where they weigh in 2v (bits 5, 3, 1 of the
   base, p1, p0), 8v + 3 or 32v + 15, so that no byte carries into the
   next. */
#define PN_L(pl, m, n) _mm256_slli_epi16(_mm256_and_si256(pl, m), n)
#define PN_R(pl, m, n) _mm256_srli_epi16(_mm256_and_si256(pl, m), n)
#define PN_OR3(a, b, c) _mm256_or_si256(_mm256_or_si256(a, b), c)
#define PN_ST(at, v) _mm256_storeu_si256((__m256i *)(void *)(w + (at)), v)

static inline PN_TARGET __attribute__((always_inline)) void
PN(unpack)(const uint8_t *q, const uint8_t *a1, const uint8_t *a0, int lv,
           uint8_t *w)
{
    const __m256i m0 = _mm256_set1_epi8(0x03), m1 = _mm256_set1_epi8(0x0c);
    const __m256i m2 = _mm256_set1_epi8(0x30),
                  m3 = _mm256_set1_epi8((char)0xc0);
    __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)q);
    if (lv >= 3) {
        __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)a1);
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)a0);
        PN_ST(0, PN_OR3(PN_L(b, m0, 5), PN_L(x1, m0, 3), PN_L(x0, m0, 1)));
        PN_ST(32, PN_OR3(PN_L(b, m1, 3), PN_L(x1, m1, 1), PN_R(x0, m1, 1)));
        PN_ST(64, PN_OR3(PN_L(b, m2, 1), PN_R(x1, m2, 1), PN_R(x0, m2, 3)));
        PN_ST(96, PN_OR3(PN_R(b, m3, 1), PN_R(x1, m3, 3), PN_R(x0, m3, 5)));
    } else if (lv == 2) {
        __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)a1);
        const __m256i o = _mm256_set1_epi8(3);
        PN_ST(0, PN_OR3(PN_L(b, m0, 5), PN_L(x1, m0, 3), o));
        PN_ST(32, PN_OR3(PN_L(b, m1, 3), PN_L(x1, m1, 1), o));
        PN_ST(64, PN_OR3(PN_L(b, m2, 1), PN_R(x1, m2, 1), o));
        PN_ST(96, PN_OR3(PN_R(b, m3, 1), PN_R(x1, m3, 3), o));
    } else {
        const __m256i o = _mm256_set1_epi8(15);
        PN_ST(0, _mm256_or_si256(PN_L(b, m0, 5), o));
        PN_ST(32, _mm256_or_si256(PN_L(b, m1, 3), o));
        PN_ST(64, _mm256_or_si256(PN_L(b, m2, 1), o));
        PN_ST(96, _mm256_or_si256(PN_R(b, m3, 1), o));
    }
}
#undef PN_L
#undef PN_R
#undef PN_OR3
#undef PN_ST

static PN_TARGET void PN(q6kpn_chunk)(const struct janas_block_q6kp *base,
                                      const uint8_t *p1, const uint8_t *p0,
                                      size_t rows, const uint32_t *js,
                                      const uint8_t *lv, const float *h,
                                      uint32_t n, size_t b, float *y)
{
    const struct janas_block_q6kp *blk[JANAS_PN_CHUNK];
    const uint8_t *a1[JANAS_PN_CHUNK], *a0[JANAS_PN_CHUNK];
    float u[JANAS_PN_CHUNK], t = 0.0f;
    if (n > JANAS_PN_CHUNK)
        n = JANAS_PN_CHUNK;
    for (uint32_t i = 0; i < n; i++) {
        size_t at = b * rows + js[i];
        blk[i] = base + at;
        a1[i] = lv[i] >= 2 ? p1 + at * JANAS_Q6KP_PLANE : NULL;
        a0[i] = lv[i] >= 3 ? p0 + at * JANAS_Q6KP_PLANE : NULL;
        /* fp16 to float is exact, as the reference's */
        u[i] = h[i] * (_cvtsh_ss(blk[i]->d) * 0.5f);
        if (fabsf(u[i]) > t)
            t = fabsf(u[i]);
    }
    if (!(t > 0.0f))
        return;
    float inv = 127.0f / (t * 128.0f), f = t * 128.0f / 127.0f;

    /* the factors u sc to eight bits - |u inv| <= 127/128 and |sc| <= 128,
       so within -127 .. 127 - by quads: aq[q][g] the four neurons' bytes
       of group g; and each group's sum of them */
    uint32_t nq = (n + 3) / 4;
    uint32_t aq[JANAS_PN_CHUNK / 4][16];
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    for (uint32_t q = 0; q < nq; q++) {
        __m128i r[4];
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t i = 4 * q + k;
            if (i >= n) {
                r[k] = _mm_setzero_si128();
                continue;
            }
            __m256 kk = _mm256_set1_ps(u[i] * inv);
            __m128i sc =
                _mm_loadu_si128((const __m128i *)(const void *)blk[i]->scales);
            __m256i lo = _mm256_cvtps_epi32(_mm256_mul_ps(
                kk, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc))));
            __m256i hi = _mm256_cvtps_epi32(
                _mm256_mul_ps(kk, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_srli_si128(sc, 8)))));
            s0 = _mm256_add_epi32(s0, lo);
            s1 = _mm256_add_epi32(s1, hi);
            __m256i p =
                _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xd8);
            r[k] = _mm_packs_epi16(_mm256_castsi256_si128(p),
                                   _mm256_extracti128_si256(p, 1));
        }
        __m128i t0 = _mm_unpacklo_epi8(r[0], r[1]),
                t1 = _mm_unpackhi_epi8(r[0], r[1]);
        __m128i t2 = _mm_unpacklo_epi8(r[2], r[3]),
                t3 = _mm_unpackhi_epi8(r[2], r[3]);
        _mm_storeu_si128((__m128i *)(void *)&aq[q][0],
                         _mm_unpacklo_epi16(t0, t2));
        _mm_storeu_si128((__m128i *)(void *)&aq[q][4],
                         _mm_unpackhi_epi16(t0, t2));
        _mm_storeu_si128((__m128i *)(void *)&aq[q][8],
                         _mm_unpacklo_epi16(t1, t3));
        _mm_storeu_si128((__m128i *)(void *)&aq[q][12],
                         _mm_unpackhi_epi16(t1, t3));
    }
    int32_t sum[16];
    _mm256_storeu_si256((__m256i *)(void *)sum, s0);
    _mm256_storeu_si256((__m256i *)(void *)(sum + 8), s1);

    /* a half's values, a row of 128 bytes a neuron; the rows past n of the
       last quad are zero (their factors are) */
    uint8_t w[JANAS_PN_CHUNK * 128];
    for (uint32_t i = n; i < 4 * nq; i++)
        memset(w + (size_t)i * 128, 0, 128);
    const __m256 vf = _mm256_set1_ps(f);
    for (int nh = 0; nh < 2; nh++) {
        for (uint32_t i = 0; i < n; i++)
            PN(unpack)(blk[i]->q + 32 * nh, a1[i] ? a1[i] + 32 * nh : NULL,
                       a0[i] ? a0[i] + 32 * nh : NULL, lv[i],
                       w + (size_t)i * 128);
        for (int part = 0; part < 4; part++) {
            int g = 8 * nh + 2 * part, v0 = 128 * nh + 32 * part;
            __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                              _mm256_setzero_si256(), _mm256_setzero_si256()};
            const uint8_t *wp = w + 32 * part;
            for (uint32_t q = 0; q < nq; q++, wp += 4 * 128) {
                __m256i w0 =
                    _mm256_loadu_si256((const __m256i *)(const void *)wp);
                __m256i w1 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(wp + 128));
                __m256i w2 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(wp + 256));
                __m256i w3 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(wp + 384));
                __m256i ab_lo = _mm256_unpacklo_epi8(w0, w1),
                        ab_hi = _mm256_unpackhi_epi8(w0, w1);
                __m256i cd_lo = _mm256_unpacklo_epi8(w2, w3),
                        cd_hi = _mm256_unpackhi_epi8(w2, w3);
                __m256i c = _mm256_set_m128i(_mm_set1_epi32((int)aq[q][g + 1]),
                                             _mm_set1_epi32((int)aq[q][g]));
                acc[0] = PN_DOT(acc[0], _mm256_unpacklo_epi16(ab_lo, cd_lo), c);
                acc[1] = PN_DOT(acc[1], _mm256_unpackhi_epi16(ab_lo, cd_lo), c);
                acc[2] = PN_DOT(acc[2], _mm256_unpacklo_epi16(ab_hi, cd_hi), c);
                acc[3] = PN_DOT(acc[3], _mm256_unpackhi_epi16(ab_hi, cd_hi), c);
            }
            /* back in order, less 64 times the factors' sums: outputs 0-15
               of the 32 are group g, 16-31 group g + 1 */
            __m256i z0 = _mm256_set1_epi32(64 * sum[g]),
                    z1 = _mm256_set1_epi32(64 * sum[g + 1]);
            __m256i r0 = _mm256_permute2x128_si256(acc[0], acc[1], 0x20);
            __m256i r1 = _mm256_permute2x128_si256(acc[2], acc[3], 0x20);
            __m256i r2 = _mm256_permute2x128_si256(acc[0], acc[1], 0x31);
            __m256i r3 = _mm256_permute2x128_si256(acc[2], acc[3], 0x31);
            float *o = y + v0;
            _mm256_storeu_ps(
                o, _mm256_fmadd_ps(vf,
                                   _mm256_cvtepi32_ps(_mm256_sub_epi32(r0, z0)),
                                   _mm256_loadu_ps(o)));
            _mm256_storeu_ps(
                o + 8, _mm256_fmadd_ps(
                           vf, _mm256_cvtepi32_ps(_mm256_sub_epi32(r1, z0)),
                           _mm256_loadu_ps(o + 8)));
            _mm256_storeu_ps(
                o + 16, _mm256_fmadd_ps(
                            vf, _mm256_cvtepi32_ps(_mm256_sub_epi32(r2, z1)),
                            _mm256_loadu_ps(o + 16)));
            _mm256_storeu_ps(
                o + 24, _mm256_fmadd_ps(
                            vf, _mm256_cvtepi32_ps(_mm256_sub_epi32(r3, z1)),
                            _mm256_loadu_ps(o + 24)));
        }
    }
}
