// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// Janas-LLM in this process: janas.node, the module build.sh makes beside
// Janas's programs, with the types of its janas.d.ts. A native module is
// loaded with require, here through createRequire.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import type * as Janas from "../../../../src/node/janas.js";

export type { Chat, Model, Piece, Reply, ReplyStats } from "../../../../src/node/janas.js";

/** the module: the path given, JANAS_NODE, or the build's own */
export function loadJanas(path?: string): typeof Janas {
    const where =
        path ??
        process.env["JANAS_NODE"] ??
        fileURLToPath(new URL("../../../../../bin/x86_64-linux/janas.node", import.meta.url));
    return createRequire(import.meta.url)(where) as typeof Janas;
}
