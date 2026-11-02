// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// The reply's events on the wire, as server-sent events: the server writes
// them with encodeEvent, the page reads them back with EventDecoder from
// the chunks fetch() gives it. One file, both sides.

import type { StreamEvent } from "./protocol.js";

export function encodeEvent(e: StreamEvent): string {
    return `data: ${JSON.stringify(e)}\n\n`;
}

export class EventDecoder {
    private buffer = "";

    /** the events completed by this chunk; comments (": ...") are skipped */
    push(chunk: string): StreamEvent[] {
        this.buffer += chunk;
        const events: StreamEvent[] = [];
        let end: number;
        while ((end = this.buffer.indexOf("\n\n")) >= 0) {
            const block = this.buffer.slice(0, end);
            this.buffer = this.buffer.slice(end + 2);
            for (const line of block.split("\n"))
                if (line.startsWith("data: ")) events.push(JSON.parse(line.slice(6)) as StreamEvent);
        }
        return events;
    }
}
