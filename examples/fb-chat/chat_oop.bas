'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' chat_oop.bas - the chat of chat.bas, with the classes of janas_oop.bi: a
'' Janas.Model, its Janas.Chat, and a handler that prints the reply as it is
'' written and its counters at the end. Nothing to close by hand: the objects
'' close what they hold when the program ends.
''
''   fbc examples/fb-chat/chat_oop.bas -i include -p bin/x86_64-linux -x janas-chat-oop
''   LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-oop model.jns
''   echo "Ciao" | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-oop model.jns
''
#include once "janas/janas_oop.bi"

'' FreeBASIC reads the console, not the standard input: its LINE INPUT waits
'' for a terminal and never sees a pipe. So the terminal keeps LINE INPUT,
'' with the line editing it brings, and a pipe is read through the CONS
'' device (as in chat.bas).
declare function isatty cdecl alias "isatty" (byval fd as long) as long

type Console
    declare constructor()
    declare destructor()
    '' one line into msg; False at the end of the input
    declare function Ask(byref msg as string) as boolean
private:
    tty as boolean
    file as long
end type

constructor Console()
    tty = isatty(0) <> 0
    if tty then exit constructor
    file = freefile
    if open cons(for input as #file) <> 0 then file = 0
end constructor

destructor Console()
    if file then close #file
end destructor

function Console.Ask(byref msg as string) as boolean
    if tty then
        line input "> ", msg
        return true
    end if
    print "> ";
    if file = 0 orelse eof(file) then
        print
        return false
    end if
    line input #file, msg
    print msg '' a pipe has no echo of its own
    return true
end function

'' the reply as it is written, then what it cost
type Terminal extends Janas.Printer
    declare sub OnEnd(byref st as const Janas.ReplyStats) override
end type

sub Terminal.OnEnd(byref st as const Janas.ReplyStats)
    print
    print using "[#### token in ###.## s, ###.# token/s, context #####/#####]"; _
        st.output_tokens; st.output_seconds; st.TokensPerSecond(); _
        st.context_used; st.context_size
end sub

dim as string path = command(1)
if path = "" then
    print "usage: chat_oop <model.jns>"
    end 2
end if
if Janas.Model.AbiVersion() <> JANAS_LLM_ABI then
    print "the library speaks another ABI"
    end 1
end if

dim as Janas.Model model = Janas.Model(path)
if model.IsOpen = false then
    print model.ErrorText
    end 1
end if
print model.Description

dim as Janas.ChatOptions options
options.max_reply = 400 '' a model left to itself can fill the whole context
dim as Janas.Chat chat = Janas.Chat(model, options)
if chat.IsOpen = false then
    print chat.ErrorText
    end 1
end if

dim as Console cons_
dim as Terminal term
do
    dim as string msg
    if cons_.Ask(msg) = false then exit do
    if msg = "" orelse msg = "/quit" then exit do
    if chat.Send(msg) = false orelse chat.ReadReply(term) = false then
        print
        print chat.ErrorText
    end if
loop
