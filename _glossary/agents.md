---
title: "agents"
source: obsidian
---

An agent is an LLM-centered software system that can execute a task through multiple steps instead of producing one isolated completion. The model is usually responsible for interpreting the task, deciding the next action, and synthesizing outputs. The surrounding system handles state, tools, memory, retries, permissions, and evaluation.

An agent is not a separate kind of model. It is a control loop around a model.

A minimal agent loop looks like this:

```text
observe state
decide next action
call tool or produce message
read result
update state
continue or stop
```

The action can be a tool call, file edit, browser step, API request, database query, shell command, or final answer. The important part is that the model is not only generating text; it is selecting actions inside a larger system.

Agent quality depends on more than model quality. Tool schemas, context selection, observation formatting, memory design, stop conditions, error recovery, and evaluation all affect behavior.

Many useful tasks are process tasks, not one-shot language tasks. Debugging code, searching documentation, booking workflows, analyzing logs, or fixing CI failures require inspection, action, feedback, and revision.

This is why agents connect directly to inference, context engineering, tool use, and reinforcement learning. The model has to decide what to do next, but the system has to make those decisions safe and observable.

Related: [mcp](/glossary/mcp), [context engineering](/glossary/context-engineering), [reinforcement learning](/glossary/reinforcement-learning), [reasoning models](/glossary/reasoning-models).
