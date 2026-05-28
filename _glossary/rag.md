---
title: "rag"
source: obsidian
---

RAG, or retrieval augmented generation, is a pattern where a model retrieves external information at inference time and uses it as context for generation.

The model does not need to memorize everything in its weights. It can consult a document store, codebase, database, or search system when answering.

A basic RAG pipeline looks like this:

```text
user query
→ retrieve relevant chunks
→ place chunks into context
→ generate grounded answer
```

The retrieval step may use keyword search, embeddings, hybrid search, reranking, metadata filters, or tool calls. The generation step depends heavily on how the retrieved context is formatted and ordered.

RAG is useful when the required information is private, recent, large, or frequently changing. It is often better to retrieve the relevant data than to fine-tune the model every time the data changes.

The main failure mode is bad retrieval. If the wrong chunks are retrieved, the model can answer confidently from irrelevant context. This is why RAG is really a context-engineering problem, not just a vector database problem.

Related: [vector databases](/glossary/vector-databases), [embedding models](/glossary/embedding-models), [context engineering](/glossary/context-engineering), [agents](/glossary/agents).
