/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deflate.c - the size of the zlib stream zlib's compress() makes at its
 * default level (6: lazy matching, chains of 128, good 8, lazy 16, nice
 * 128, a window of 32 KiB, hash chains of 15 bits, blocks of up to 16383
 * symbols), to the byte: its matches found as zlib's deflate_slow finds
 * them, its Huffman trees built as zlib's trees.c builds them (the heap's
 * order, the depth that breaks ties, the lengths limited to 15 and 7 as
 * zlib limits them), each block stored, with the fixed codes or with its
 * own as zlib chooses. What OpenAI's Whisper measures a text's repetition
 * with (its compression ratio).
 */
#include "deflate.h"

#include <stdlib.h>
#include <string.h>

#define MIN_MATCH 3
#define MAX_MATCH 258
#define MIN_LOOKAHEAD (MAX_MATCH + MIN_MATCH + 1)
#define W_SIZE 32768
#define W_MASK (W_SIZE - 1)
#define MAX_DIST (W_SIZE - MIN_LOOKAHEAD)
#define HASH_BITS 15
#define HASH_SIZE (1 << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define HASH_SHIFT ((HASH_BITS + MIN_MATCH - 1) / MIN_MATCH)
#define TOO_FAR 4096
#define LIT_BUFSIZE 16384
#define GOOD_LENGTH 8
#define MAX_LAZY 16
#define NICE_LENGTH 128
#define MAX_CHAIN 128

#define LITERALS 256
#define END_BLOCK 256
#define L_CODES (LITERALS + 1 + 29)
#define D_CODES 30
#define BL_CODES 19
#define HEAP_SIZE (2 * L_CODES + 1)
#define MAX_BITS 15
#define REP_3_6 16
#define REPZ_3_10 17
#define REPZ_11_138 18

static const int EXTRA_LBITS[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const int EXTRA_DBITS[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                    4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                    9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static const int EXTRA_BLBITS[19] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 2, 3, 7};
static const uint8_t BL_ORDER[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                     11, 4,  12, 3, 13, 2, 14, 1, 15};
static const int BASE_LENGTH[29] = {0,  1,  2,  3,   4,   5,   6,   7,   8,  10,
                                    12, 14, 16, 20,  24,  28,  32,  40,  48, 56,
                                    64, 80, 96, 112, 128, 160, 192, 224, 255};
static const int BASE_DIST[30] = {
    0,    1,    2,    3,    4,    6,    8,    12,    16,    24,
    32,   48,   64,   96,   128,  192,  256,  384,   512,   768,
    1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576};

/* a tree's node: its frequency, and its parent until its length replaces
   it (zlib's union) */
struct node {
    uint32_t freq;
    uint16_t dl; /* dad, then len */
};

struct desc {
    struct node *tree;
    int max_code;
    const struct node *stree; /* the fixed code's lengths, or NULL */
    const int *extra;
    int base, elems, max_length;
};

struct z {
    const uint8_t *win;
    size_t n;
    uint32_t *head, *prev;
    uint32_t ins_h;
    size_t strstart, lookahead, match_start, prev_match;
    unsigned match_length, prev_length;
    int match_available;
    struct node dyn_l[HEAP_SIZE], dyn_d[2 * D_CODES + 1], bl[2 * BL_CODES + 1];
    struct node stat_l[L_CODES + 2], stat_d[D_CODES];
    int heap[2 * L_CODES + 1], heap_len, heap_max;
    uint8_t depth[2 * L_CODES + 1];
    uint16_t bl_count[MAX_BITS + 1];
    uint64_t opt_len, static_len;
    unsigned sym_next;
    size_t block_start;
    uint64_t bits; /* written so far */
};

static int length_code(unsigned len)
{
    for (int c = 28; c >= 0; c--)
        if (len >= (unsigned)BASE_LENGTH[c])
            return c;
    return 0;
}

static int dist_code(unsigned dist)
{
    for (int c = 29; c >= 0; c--)
        if (dist >= (unsigned)BASE_DIST[c])
            return c;
    return 0;
}

static void init_block(struct z *s)
{
    for (int i = 0; i < L_CODES; i++)
        s->dyn_l[i].freq = 0;
    for (int i = 0; i < D_CODES; i++)
        s->dyn_d[i].freq = 0;
    for (int i = 0; i < BL_CODES; i++)
        s->bl[i].freq = 0;
    s->dyn_l[END_BLOCK].freq = 1;
    s->opt_len = s->static_len = 0;
    s->sym_next = 0;
}

#define SMALLER(t, n, m)                                                       \
    ((t)[n].freq < (t)[m].freq ||                                              \
     ((t)[n].freq == (t)[m].freq && s->depth[n] <= s->depth[m]))

static void pqdownheap(struct z *s, struct node *tree, int k)
{
    int v = s->heap[k], j = k << 1;
    while (j <= s->heap_len) {
        if (j < s->heap_len && SMALLER(tree, s->heap[j + 1], s->heap[j]))
            j++;
        if (SMALLER(tree, v, s->heap[j]))
            break;
        s->heap[k] = s->heap[j];
        k = j;
        j <<= 1;
    }
    s->heap[k] = v;
}

static void gen_bitlen(struct z *s, struct desc *d)
{
    struct node *tree = d->tree;
    int max_code = d->max_code, max_length = d->max_length, overflow = 0, h;
    for (int b = 0; b <= MAX_BITS; b++)
        s->bl_count[b] = 0;
    tree[s->heap[s->heap_max]].dl = 0;
    for (h = s->heap_max + 1; h < HEAP_SIZE; h++) {
        int n = s->heap[h];
        int bits = tree[tree[n].dl].dl + 1;
        if (bits > max_length) {
            bits = max_length;
            overflow++;
        }
        tree[n].dl = (uint16_t)bits;
        if (n > max_code)
            continue;
        s->bl_count[bits]++;
        int xbits = n >= d->base ? d->extra[n - d->base] : 0;
        uint64_t f = tree[n].freq;
        s->opt_len += f * (unsigned)(bits + xbits);
        if (d->stree)
            s->static_len += f * (unsigned)(d->stree[n].dl + xbits);
    }
    if (!overflow)
        return;
    do {
        int bits = max_length - 1;
        while (s->bl_count[bits] == 0)
            bits--;
        s->bl_count[bits]--;
        s->bl_count[bits + 1] += 2;
        s->bl_count[max_length]--;
        overflow -= 2;
    } while (overflow > 0);
    for (int bits = max_length; bits != 0; bits--) {
        int n = s->bl_count[bits];
        while (n != 0) {
            int m = s->heap[--h];
            if (m > max_code)
                continue;
            if ((unsigned)tree[m].dl != (unsigned)bits) {
                s->opt_len += ((uint64_t)bits - tree[m].dl) * tree[m].freq;
                tree[m].dl = (uint16_t)bits;
            }
            n--;
        }
    }
}

static void build_tree(struct z *s, struct desc *d)
{
    struct node *tree = d->tree;
    int elems = d->elems, max_code = -1, node;
    s->heap_len = 0;
    s->heap_max = HEAP_SIZE;
    for (int n = 0; n < elems; n++) {
        if (tree[n].freq) {
            s->heap[++s->heap_len] = max_code = n;
            s->depth[n] = 0;
        } else {
            tree[n].dl = 0;
        }
    }
    while (s->heap_len < 2) {
        node = s->heap[++s->heap_len] = max_code < 2 ? ++max_code : 0;
        tree[node].freq = 1;
        s->depth[node] = 0;
        s->opt_len--;
        if (d->stree)
            s->static_len -= d->stree[node].dl;
    }
    d->max_code = max_code;
    for (int n = s->heap_len / 2; n >= 1; n--)
        pqdownheap(s, tree, n);
    node = elems;
    do {
        int n = s->heap[1];
        s->heap[1] = s->heap[s->heap_len--];
        pqdownheap(s, tree, 1);
        int m = s->heap[1];
        s->heap[--s->heap_max] = n;
        s->heap[--s->heap_max] = m;
        tree[node].freq = tree[n].freq + tree[m].freq;
        s->depth[node] =
            (uint8_t)((s->depth[n] >= s->depth[m] ? s->depth[n] : s->depth[m]) +
                      1);
        tree[n].dl = tree[m].dl = (uint16_t)node;
        s->heap[1] = node++;
        pqdownheap(s, tree, 1);
    } while (s->heap_len >= 2);
    s->heap[--s->heap_max] = s->heap[1];
    gen_bitlen(s, d);
}

static void scan_tree(struct z *s, struct node *tree, int max_code)
{
    int prevlen = -1, nextlen = tree[0].dl, count = 0;
    int max_count = 7, min_count = 4;
    if (nextlen == 0)
        max_count = 138, min_count = 3;
    tree[max_code + 1].dl = 0xffff;
    for (int n = 0; n <= max_code; n++) {
        int curlen = nextlen;
        nextlen = tree[n + 1].dl;
        if (++count < max_count && curlen == nextlen)
            continue;
        else if (count < min_count)
            s->bl[curlen].freq += (uint32_t)count;
        else if (curlen != 0) {
            if (curlen != prevlen)
                s->bl[curlen].freq++;
            s->bl[REP_3_6].freq++;
        } else if (count <= 10)
            s->bl[REPZ_3_10].freq++;
        else
            s->bl[REPZ_11_138].freq++;
        count = 0;
        prevlen = curlen;
        if (nextlen == 0)
            max_count = 138, min_count = 3;
        else if (curlen == nextlen)
            max_count = 6, min_count = 3;
        else
            max_count = 7, min_count = 4;
    }
}

/* a block of the symbols tallied, of stored_len bytes from block_start */
static void flush_block(struct z *s, size_t stored_len, int last)
{
    struct desc l = {s->dyn_l,     0,       s->stat_l, EXTRA_LBITS,
                     LITERALS + 1, L_CODES, MAX_BITS};
    struct desc dd = {s->dyn_d, 0,       s->stat_d, EXTRA_DBITS,
                      0,        D_CODES, MAX_BITS};
    struct desc bl = {s->bl, 0, NULL, EXTRA_BLBITS, 0, BL_CODES, 7};
    build_tree(s, &l);
    build_tree(s, &dd);
    scan_tree(s, s->dyn_l, l.max_code);
    scan_tree(s, s->dyn_d, dd.max_code);
    build_tree(s, &bl);
    int max_blindex;
    for (max_blindex = BL_CODES - 1; max_blindex >= 3; max_blindex--)
        if (s->bl[BL_ORDER[max_blindex]].dl != 0)
            break;
    s->opt_len += 3 * ((uint64_t)max_blindex + 1) + 5 + 5 + 4;
    uint64_t opt_lenb = (s->opt_len + 3 + 7) >> 3;
    uint64_t static_lenb = (s->static_len + 3 + 7) >> 3;
    if (static_lenb <= opt_lenb)
        opt_lenb = static_lenb;
    if (stored_len + 4 <= opt_lenb) {
        s->bits += 3;
        s->bits = (s->bits + 7) & ~(uint64_t)7; /* to a byte */
        s->bits += 32 + 8 * (uint64_t)stored_len;
    } else if (static_lenb == opt_lenb) {
        s->bits += 3 + s->static_len;
    } else {
        s->bits += 3 + s->opt_len;
    }
    init_block(s);
    if (last)
        s->bits = (s->bits + 7) & ~(uint64_t)7;
}

static void insert(struct z *s, size_t str, uint32_t *head)
{
    s->ins_h =
        ((s->ins_h << HASH_SHIFT) ^ s->win[str + MIN_MATCH - 1]) & HASH_MASK;
    *head = s->prev[str & W_MASK] = s->head[s->ins_h];
    s->head[s->ins_h] = (uint32_t)str;
}

static unsigned longest_match(struct z *s, uint32_t cur_match)
{
    unsigned chain = MAX_CHAIN;
    const uint8_t *scan = s->win + s->strstart;
    unsigned best_len = s->prev_length;
    unsigned nice = NICE_LENGTH;
    size_t limit = s->strstart > MAX_DIST ? s->strstart - MAX_DIST : 0;
    const uint8_t *strend = s->win + s->strstart + MAX_MATCH;
    if (s->prev_length >= GOOD_LENGTH)
        chain >>= 2;
    if (nice > s->lookahead)
        nice = (unsigned)s->lookahead;
    uint8_t scan_end1 = scan[best_len - 1], scan_end = scan[best_len];
    do {
        const uint8_t *match = s->win + cur_match;
        if (match[best_len] != scan_end || match[best_len - 1] != scan_end1 ||
            match[0] != scan[0] || match[1] != scan[1])
            continue;
        const uint8_t *sc = scan + 2, *mt = match + 2;
        while (sc < strend && *++sc == *++mt)
            ;
        unsigned len = (unsigned)(MAX_MATCH - (strend - sc));
        if (sc > strend)
            len = MAX_MATCH;
        if (len > best_len) {
            s->match_start = cur_match;
            best_len = len;
            if (len >= nice)
                break;
            scan_end1 = scan[best_len - 1];
            scan_end = scan[best_len];
        }
    } while ((cur_match = s->prev[cur_match & W_MASK]) > limit && --chain != 0);
    return best_len <= s->lookahead ? best_len : (unsigned)s->lookahead;
}

static void tally(struct z *s, unsigned dist, unsigned lc, size_t at)
{
    if (!dist) {
        s->dyn_l[lc].freq++;
    } else {
        dist--;
        s->dyn_l[length_code(lc) + LITERALS + 1].freq++;
        s->dyn_d[dist_code(dist)].freq++;
    }
    s->sym_next += 3;
    if (s->sym_next == (LIT_BUFSIZE - 1) * 3) {
        flush_block(s, at - s->block_start, 0);
        s->block_start = at;
    }
}

long janas_deflate_size(const uint8_t *p, size_t n)
{
    /* one window: what this needs, Whisper's texts being short */
    if (n > W_SIZE - MIN_LOOKAHEAD)
        return -1;
    struct z *s = calloc(1, sizeof *s);
    uint8_t *win = calloc(n + MAX_MATCH + MIN_MATCH + 8, 1);
    if (s)
        s->head = calloc(HASH_SIZE, sizeof *s->head);
    if (s)
        s->prev = calloc(W_SIZE, sizeof *s->prev);
    if (!s || !win || !s->head || !s->prev) {
        if (s) {
            free(s->head);
            free(s->prev);
        }
        free(s);
        free(win);
        return -1;
    }
    memcpy(win, p, n);
    s->win = win;
    s->n = n;
    /* the fixed codes' lengths */
    for (int i = 0; i < L_CODES + 2; i++)
        s->stat_l[i].dl = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
    for (int i = 0; i < D_CODES; i++)
        s->stat_d[i].dl = 5;
    init_block(s);
    s->lookahead = n;
    s->match_length = s->prev_length = MIN_MATCH - 1;
    if (n >= MIN_MATCH) {
        s->ins_h = win[0];
        s->ins_h = ((s->ins_h << HASH_SHIFT) ^ win[1]) & HASH_MASK;
    }
    /* deflate_slow */
    while (s->lookahead > 0) {
        uint32_t hash_head = 0;
        if (s->lookahead >= MIN_MATCH)
            insert(s, s->strstart, &hash_head);
        s->prev_length = s->match_length;
        s->prev_match = s->match_start;
        s->match_length = MIN_MATCH - 1;
        if (hash_head != 0 && s->prev_length < MAX_LAZY &&
            s->strstart - hash_head <= MAX_DIST) {
            s->match_length = longest_match(s, hash_head);
            if (s->match_length == MIN_MATCH &&
                s->strstart - s->match_start > TOO_FAR)
                s->match_length = MIN_MATCH - 1;
        }
        if (s->prev_length >= MIN_MATCH && s->match_length <= s->prev_length) {
            size_t max_insert = s->strstart + s->lookahead - MIN_MATCH;
            size_t at = s->strstart - 1;
            unsigned dist = (unsigned)(s->strstart - 1 - s->prev_match);
            unsigned len = s->prev_length - MIN_MATCH;
            s->lookahead -= s->prev_length - 1;
            s->prev_length -= 2;
            do {
                if (++s->strstart <= max_insert)
                    insert(s, s->strstart, &hash_head);
            } while (--s->prev_length != 0);
            s->match_available = 0;
            s->match_length = MIN_MATCH - 1;
            s->strstart++;
            /* the symbols' block ends after the match */
            (void)at;
            tally(s, dist, len, s->strstart);
        } else if (s->match_available) {
            tally(s, 0, win[s->strstart - 1], s->strstart);
            s->strstart++;
            s->lookahead--;
        } else {
            s->match_available = 1;
            s->strstart++;
            s->lookahead--;
        }
    }
    if (s->match_available)
        tally(s, 0, win[s->strstart - 1], s->strstart);
    flush_block(s, s->strstart - s->block_start, 1);
    long size = 2 + (long)(s->bits / 8) + 4;
    free(s->head);
    free(s->prev);
    free(s);
    free(win);
    return size;
}
