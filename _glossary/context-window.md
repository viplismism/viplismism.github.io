---
title: "context window"
source: obsidian
---

The context window is the maximum active sequence length a model can handle for one request. It is the model's working view during inference.

The budget usually includes both prompt tokens and generated tokens. A `32k` context window does not mean `32k` prompt tokens plus unlimited output. It means the active sequence has to fit inside that shared budget.

Context length affects both capability and cost. Long context can help with large documents, long chats, codebase chunks, RAG pipelines, and multi-step task state. But it also increases prefill work, KV-cache memory, latency, and serving cost.

The two important questions are:

```text
can the model ingest this many tokens?
can the model use those tokens effectively for the task?
```

Those are not the same thing. A model may support a large context length but still degrade when the task requires dense reasoning across the entire context.

There are two different limits:

```text
training/adaptation context: what sequence lengths the model was trained to handle
serving context: what sequence length the deployment exposes in practice
```

A model might technically support a large context window, but a deployment may choose a smaller limit because the KV cache is too large, latency is too high, or concurrency falls apart.

So supported context and practical context are related, but not identical.

During autoregressive inference, the model stores key/value states for past tokens in the [kv cache](/glossary/kv-cache). Longer active sequences usually mean larger cache state.

This is why a model can fit in GPU memory at short context but fail under long context. The weights may fit, but the dynamic cache grows with active tokens, number of layers, number of KV heads, and hidden dimension.

Context window is therefore an architecture and deployment concept, not only a model-card number.

Longer context does not remove the need for context engineering. Bad context still hurts. Common waste includes repeated instructions, irrelevant retrieved chunks, huge tool outputs, stale chat history, too many examples, and low-quality summaries.

The useful question is not:

```text
how many tokens can I stuff in?
```

The useful question is:

```text
how much relevant context can I include before quality, latency, or cost gets worse?
```

For RAG and agents, context management is a core system design problem. The system has to decide what to retrieve, what to compress, what to discard, and what to keep exact.

Common mistakes:

```text
confusing bigger context with better reasoning
forgetting generated tokens count too
ignoring KV-cache memory
assuming all positions are used equally well
stuffing prompts with irrelevant data
exposing the maximum context even when serving cannot afford it
```

Long context is useful, but it is not free memory. It is an expensive working set.

Related: [kv cache](/glossary/kv-cache), [inference](/glossary/inference), [context engineering](/glossary/context-engineering), [rag](/glossary/rag), [mla](/glossary/mla).
