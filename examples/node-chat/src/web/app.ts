// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// The chat's page: the conversation kept here and sent whole with each
// message, the reply read as the server streams it, its reasoning apart, and
// what it cost under it. The types of what goes to the server and back are
// the server's own (../shared/).

import type { ChatRequest, Message, ModelInfo, ReplyCost, StreamEvent } from "../shared/protocol.js";
import { EventDecoder } from "../shared/sse.js";
import { markdown } from "./markdown.js";

function $<T extends HTMLElement>(id: string): T {
    const node = document.getElementById(id);
    if (!node) throw new Error(`the page has no #${id}`);
    return node as T;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    node.className = className;
    return node;
}

const log = $<HTMLElement>("log");
const input = $<HTMLTextAreaElement>("input");
const sendButton = $<HTMLButtonElement>("send");
const stopButton = $<HTMLButtonElement>("stop");
const newButton = $<HTMLButtonElement>("new");
const system = $<HTMLTextAreaElement>("system");
const thinking = $<HTMLInputElement>("thinking");
const thinkingRow = $<HTMLElement>("thinking-row");
const temperature = $<HTMLInputElement>("temperature");
const temperatureValue = $<HTMLOutputElement>("temperature-value");

const conversation: Message[] = [];
let running: AbortController | null = null;

/** a reply on the page, as it comes */
class ReplyView {
    readonly box = element("div", "message assistant");
    private readonly reasoningBox = element("details", "reasoning");
    private readonly reasoningText = element("div", "reasoning-text");
    private readonly answerBox = element("div", "answer");
    private readonly meta = element("div", "meta");
    reasoning = "";
    answer = "";

    constructor() {
        const summary = element("summary", "");
        summary.textContent = "reasoning";
        this.reasoningBox.append(summary, this.reasoningText);
        this.reasoningBox.hidden = true;
        this.meta.textContent = "…";
        this.box.append(this.reasoningBox, this.answerBox, this.meta);
    }

    add(text: string, isReasoning: boolean): void {
        if (isReasoning) {
            this.reasoning += text;
            this.reasoningBox.hidden = false;
            this.reasoningBox.open = this.answer === "";
            this.reasoningText.textContent = this.reasoning;
        } else {
            if (this.answer === "") this.reasoningBox.open = false;
            this.answer += text;
            this.answerBox.innerHTML = markdown(this.answer);
        }
        this.meta.textContent = "writing…";
    }

    status(text: string): void {
        this.meta.textContent = text;
    }

    done(cost: ReplyCost): void {
        this.meta.textContent = describe(cost);
    }

    fail(text: string): void {
        this.meta.textContent = text;
        this.meta.classList.add("error");
    }
}

function describe(c: ReplyCost): string {
    const rate = c.outputSeconds > 0 ? c.outputTokens / c.outputSeconds : 0;
    const parts = [
        `${c.outputTokens} tokens at ${rate.toFixed(1)} tok/s`,
        `the first after ${c.firstTokenSeconds.toFixed(2)} s`,
        `prompt ${c.promptTokens} tokens, ${c.cachedTokens} of them already read`,
    ];
    if (c.reasoningTokens > 0) parts.push(`${c.reasoningTokens} of reasoning`);
    if (c.drafted > 0) parts.push(`drafts kept ${c.accepted} of ${c.drafted}`);
    if (c.finish === "stopped") parts.push("stopped");
    else if (c.finish === "length") parts.push("cut at the limit");
    return parts.join(" · ");
}

function scroll(): void {
    log.scrollTop = log.scrollHeight;
}

function busy(on: boolean): void {
    sendButton.disabled = on;
    newButton.disabled = on;
    stopButton.hidden = !on;
}

function show(view: ReplyView, e: StreamEvent): void {
    switch (e.type) {
        case "queued":
            view.status(`waiting: ${e.ahead} before this one`);
            break;
        case "preparing":
            view.status(`preparing the NPU ${e.done}/${e.total}, ${Math.round(e.seconds)} s (once a session)`);
            break;
        case "progress":
            view.status(`reading the prompt: ${e.read} of ${e.total} tokens`);
            break;
        case "text":
            view.add(e.text, e.reasoning);
            scroll();
            break;
        case "done":
            view.done(e.cost);
            break;
        case "error":
            view.fail(`${e.code}: ${e.message}`);
            break;
        default: {
            const unknown: never = e;
            return unknown;
        }
    }
}

async function send(): Promise<void> {
    const text = input.value.trim();
    if (!text || running) return;
    input.value = "";
    conversation.push({ role: "user", content: text });
    const mine = element("div", "message user");
    mine.textContent = text;
    const view = new ReplyView();
    log.append(mine, view.box);
    scroll();

    const head: Message[] = system.value.trim() ? [{ role: "system", content: system.value.trim() }] : [];
    const request: ChatRequest = {
        messages: [...head, ...conversation],
        options: { thinking: thinking.checked, temperature: Number(temperature.value) },
    };
    const ask = new AbortController();
    running = ask;
    busy(true);
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request),
            signal: ask.signal,
        });
        if (!res.ok || !res.body) {
            const e = (await res.json().catch(() => ({}))) as { error?: string };
            throw new Error(e.error ?? `HTTP ${res.status}`);
        }
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        const decoder = new EventDecoder();
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            for (const e of decoder.push(value)) show(view, e);
        }
    } catch (e) {
        if (ask.signal.aborted) view.status(view.answer ? "stopped" : "stopped before the answer");
        else view.fail(`no reply: ${(e as Error).message}`);
    } finally {
        if (view.answer) conversation.push({ role: "assistant", content: view.answer });
        else {
            conversation.pop(); // unanswered: it leaves the conversation
            mine.classList.add("unanswered");
        }
        running = null;
        busy(false);
        input.focus();
    }
}

sendButton.addEventListener("click", () => void send());
stopButton.addEventListener("click", () => running?.abort());
newButton.addEventListener("click", () => {
    conversation.length = 0;
    log.replaceChildren();
    input.focus();
});
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        void send();
    }
});
temperature.addEventListener("input", () => {
    temperatureValue.value = temperature.value;
});

async function start(): Promise<void> {
    const info = (await (await fetch("/api/model")).json()) as ModelInfo;
    $("model-name").textContent = info.name || "a model";
    $("model-desc").textContent = info.description;
    $("versions").textContent = `Janas ${info.janas} · Node ${info.node}${info.mtp ? " · MTP" : ""}`;
    system.value = info.defaultSystem;
    thinkingRow.hidden = !info.reasons;
    document.title = `${info.name || "Janas"} · Janas-LLM in Node.js`;
    temperatureValue.value = temperature.value;
    input.focus();
}

start().catch((e: unknown) => {
    $("model-desc").textContent = `the server did not answer: ${(e as Error).message}`;
});
