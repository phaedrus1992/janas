/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * aligned.h - memory on a boundary of a power of two, and its release.
 *
 * A block from janas_aligned_alloc goes back through janas_aligned_free,
 * never free(), and an ordinary block never through janas_aligned_free.
 * On Linux and macOS they are aligned_alloc and free. Windows's C library
 * has no aligned_alloc, and its own (_aligned_malloc) wants _aligned_free:
 * there a block is malloc's, with a tag before the address returned that
 * says where it began, read only by janas_aligned_free. Until 18 Oct 2026
 * free() itself looked for the tag before any block, and read the page
 * before one that began on a page boundary, which may not be mapped: an
 * access violation, seen on Windows on ARM64 (PR #31).
 *
 * build.sh's asan variant makes Linux's the tagged ones too
 * (JANAS_ALIGNED_CHECK): a tagged block given to free() is not one malloc
 * returned, an ordinary one given to janas_aligned_free has its tag read
 * in the sanitizer's red zone, and either stops the program.
 */
#ifndef JANAS_COMMON_ALIGNED_H
#define JANAS_COMMON_ALIGNED_H

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(JANAS_ALIGNED_CHECK)

/* where malloc's block began, and that address mixed with a constant: a
   block given to the wrong release stops the program rather than freeing
   what the bytes before it say */
#define JANAS_ALIGN_TAG 0x4a414e4153414c47ull
struct janas_align_tag {
    void *base;
    uint64_t check;
};

static inline void *janas_aligned_alloc(size_t align, size_t size)
{
    if (align < sizeof(void *) || (align & (align - 1))) {
        errno = EINVAL;
        return NULL;
    }
    size_t extra = align + sizeof(struct janas_align_tag);
    if (size > SIZE_MAX - extra) {
        errno = ENOMEM;
        return NULL;
    }
    char *base = malloc(size + extra);
    if (!base)
        return NULL;
    uintptr_t a =
        ((uintptr_t)base + sizeof(struct janas_align_tag) + align - 1) &
        ~(uintptr_t)(align - 1);
    struct janas_align_tag *t = (struct janas_align_tag *)a - 1;
    t->base = base;
    t->check = JANAS_ALIGN_TAG ^ (uint64_t)a;
    return (void *)a;
}

static inline void janas_aligned_free(void *p)
{
    if (!p)
        return;
    struct janas_align_tag *t = (struct janas_align_tag *)p - 1;
    if (t->check != (JANAS_ALIGN_TAG ^ (uint64_t)(uintptr_t)p))
        abort(); /* not janas_aligned_alloc's, or freed twice */
    t->check = 0;
    free(t->base);
}

#else

static inline void *janas_aligned_alloc(size_t align, size_t size)
{
    return aligned_alloc(align, size);
}

static inline void janas_aligned_free(void *p)
{
    free(p);
}

#endif
#endif
