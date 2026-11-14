// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// The few marks a reply uses most, as HTML: fenced code, inline code, bold,
// paragraphs and line breaks. Everything else stays text: the model's
// words are escaped first, so nothing it writes becomes markup. A fence not
// closed yet (a reply still being written) is code to the end.

const ESCAPES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

function escape(s: string): string {
    return s.replace(/[&<>"']/g, (c) => ESCAPES[c] ?? c);
}

function prose(text: string): string {
    return text
        .split(/\n{2,}/)
        .filter((p) => p.trim() !== "")
        .map((p) => {
            const html = escape(p.trim())
                .replace(/`([^`\n]+)`/g, "<code>$1</code>")
                .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
                .replace(/\n/g, "<br>");
            return `<p>${html}</p>`;
        })
        .join("");
}

export function markdown(text: string): string {
    return text
        .split("```")
        .map((part, i) => {
            if (i % 2 === 0) return prose(part);
            const nl = part.indexOf("\n");
            const code = nl >= 0 ? part.slice(nl + 1) : part;
            return `<pre><code>${escape(code.replace(/\n$/, ""))}</code></pre>`;
        })
        .join("");
}
