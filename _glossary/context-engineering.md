---
title: "context engineering"
source: obsidian
---

Context engineering is the design of what information gets placed around the model at inference time. It is broader than prompt engineering.

Prompt engineering asks how to word the instruction. Context engineering asks what the model should see, in what order, in what format, from which systems, under what token and latency budget.

Context can include:

```text
system instructions
user request
retrieved documents
conversation history
tool outputs
schemas
examples
memory
current state
constraints
```

Good context engineering controls relevance, compression, ordering, freshness, and trust boundaries. Bad context engineering causes context rot: the model technically has many tokens, but the useful signal is buried inside noise.

Model quality is often limited by context quality. A strong model with bad context can fail. A smaller model with clean context can perform surprisingly well.

For agents, context engineering becomes a systems problem. The agent has to decide what to retrieve, what to keep, what to discard, what to summarize, and what tool outputs should be shown back to the model.

Related: [rag](/glossary/rag), [agents](/glossary/agents), [context window](/glossary/context-window), [mcp](/glossary/mcp).
