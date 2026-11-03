// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// One reply at a time, in the order asked: the model holds one
// conversation. The engine keeps its own order too; the line here makes a
// request's parameters and its reply one step, and says how many wait.

export class Line {
    private tail: Promise<void> = Promise.resolve();
    private count = 0;

    /** requests waiting or running */
    get length(): number {
        return this.count;
    }

    async run<T>(job: () => Promise<T>): Promise<T> {
        this.count++;
        const before = this.tail;
        let done!: () => void;
        this.tail = new Promise<void>((resolve) => (done = resolve));
        try {
            await before;
            return await job();
        } finally {
            this.count--;
            done();
        }
    }
}
