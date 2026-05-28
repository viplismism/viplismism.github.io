---
title: "coding agents"
source: obsidian
---



a coding agents comprises of 

1. live repo context
2. prompt cache
3. tools
4. context management , aka minimizing the context bloat
5. session memory 
6. subagents

all of them are implemented from [scratch](https://github.com/rasbt/mini-coding-agent)



memory perspective:

To summarise, a coding agent separates state into (at least) two layers:

- working memory: the small, distilled state the agent keeps explicitly
- a full transcript: this covers all the user requests, tool outputs, and LLM responses

the compact transcript and working memory have slightly different jobs. The compact transcript is for prompt reconstruction. Its job is to give the model a compressed view of recent history so it can continue the conversation without seeing the full transcript every turn. The working memory is more meant for task continuity. Its job is to keep a small, explicitly maintained summary of what matters across turns, things like the current task, important files, and recent notes.


A subagent is only useful if it inherits enough context to do real work. But if we don’t restrict it, we now have multiple agents duplicating work, touching the same files, or spawning more subagents, and so on.
