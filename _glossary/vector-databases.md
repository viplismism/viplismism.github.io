---
title: "vector databases"
source: obsidian
---

A vector database stores embeddings and supports similarity search over those embeddings.

It is commonly used in RAG systems to retrieve chunks of text, code, images, or other data that are semantically close to a query.

The basic flow is:

```text
document chunks → embedding model → vectors → vector index
query → embedding model → query vector → nearest-neighbor search
```

The database returns items whose vectors are close to the query vector under a metric such as cosine similarity, dot product, or Euclidean distance.

Many vector databases also support metadata filters, hybrid keyword/vector search, namespaces, and reranking integrations.

Vector databases are useful when exact keyword matching is not enough. They can retrieve semantically related content even when wording differs.

The limitation is that semantic similarity is not always task relevance. For code, legal, finance, and technical docs, exact identifiers, metadata, and keyword search may be just as important as embeddings.

Related: [embedding models](/glossary/embedding-models), [vectors](/glossary/vectors), [rag](/glossary/rag).
