// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// What the server and the page say to each other: the same types on both
// sides, and the check of a request, which the server runs on what it is
// sent and the page could run on what it sends.

export type Role = "system" | "user" | "assistant";

export interface Message {
    role: Role;
    content: string;
}

export interface ChatOptions {
    /** reasoning before the answer, on the models that do it */
    thinking?: boolean;
    /** 0: always the most likely token */
    temperature?: number;
}

/** POST /api/chat: the whole conversation, the page keeps it */
export interface ChatRequest {
    messages: Message[];
    options?: ChatOptions;
}

/** GET /api/model */
export interface ModelInfo {
    name: string;
    description: string;
    reasons: boolean;
    defaultSystem: string;
    mtp: boolean;
    /** the build of Janas, and of Node, the server runs */
    janas: string;
    node: string;
}

/** what a reply cost, from the engine's counters */
export interface ReplyCost {
    finish: string | null;
    promptTokens: number;
    cachedTokens: number;
    outputTokens: number;
    reasoningTokens: number;
    drafted: number;
    accepted: number;
    firstTokenSeconds: number;
    inputSeconds: number;
    outputSeconds: number;
    totalSeconds: number;
}

/** the reply, as the server streams it (server-sent events) */
export type StreamEvent =
    | { type: "queued"; ahead: number }
    | { type: "preparing"; done: number; total: number; seconds: number }
    | { type: "progress"; read: number; total: number }
    | { type: "text"; text: string; reasoning: boolean }
    | { type: "done"; cost: ReplyCost }
    | { type: "error"; code: string; message: string };

const ROLES: readonly Role[] = ["system", "user", "assistant"];

/** A request as JSON.parse gave it: the request, or what is wrong with it. */
export function parseChatRequest(v: unknown): ChatRequest | string {
    if (typeof v !== "object" || v === null) return "the body must be an object";
    const { messages, options } = v as { messages?: unknown; options?: unknown };
    if (!Array.isArray(messages) || messages.length === 0)
        return "messages must be an array, one message at least";
    const out: Message[] = [];
    for (const [i, m] of messages.entries()) {
        if (typeof m !== "object" || m === null) return `message ${i} is not an object`;
        const { role, content } = m as { role?: unknown; content?: unknown };
        if (typeof role !== "string" || !ROLES.includes(role as Role))
            return `message ${i}: role must be one of ${ROLES.join(", ")}`;
        if (typeof content !== "string") return `message ${i}: content must be a string`;
        if (role === "system" && i > 0) return `message ${i}: only the first may be the system's`;
        out.push({ role: role as Role, content });
    }
    if (out[out.length - 1]?.role !== "user") return "the last message must be the user's";
    const request: ChatRequest = { messages: out };
    if (options !== undefined) {
        if (typeof options !== "object" || options === null) return "options must be an object";
        const { thinking, temperature } = options as { thinking?: unknown; temperature?: unknown };
        if (thinking !== undefined && typeof thinking !== "boolean") return "options.thinking must be a boolean";
        if (temperature !== undefined && (typeof temperature !== "number" || !(temperature >= 0 && temperature <= 2)))
            return "options.temperature must be a number from 0 to 2";
        request.options = {
            ...(thinking !== undefined ? { thinking } : {}),
            ...(temperature !== undefined ? { temperature } : {}),
        };
    }
    return request;
}
