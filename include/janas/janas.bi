'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' janas.bi - Janas for FreeBASIC, and so for BASIC MODERN, the dialect
'' Prabanta implements: the two libraries in one include.
''
'' llm.bi declares Janas-LLM (a model and its conversation), mcp.bi Janas-MCP
'' (the tools of MCP servers), procedure by procedure as their C headers do.
'' janas_oop.bi builds classes on them, for FreeBASIC.
''
'' Build:   fbc yours.bas -i <janas>/include -p <janas>/bin/x86_64-linux
'' Run:     the libraries must be found (LD_LIBRARY_PATH, or install them)

#pragma once
#include once "janas/llm.bi"
#include once "janas/mcp.bi"
