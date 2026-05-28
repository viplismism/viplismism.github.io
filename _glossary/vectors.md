---
title: "vectors"
source: obsidian
---

A vector is an ordered list of numbers. In machine learning, vectors are used to represent tokens, documents, images, audio segments, model states, and many other objects.

For language models, vectors are the internal numerical form of meaning and context.

Example:

```text
"cat" → [0.12, -0.08, 0.44, ...]
```

The individual dimensions are usually not directly human-interpretable. What matters is the geometry: distances, directions, clusters, and transformations.

Models operate on vectors using linear algebra: matrix multiplication, dot products, normalization, projections, attention, and nonlinear activations.

Vectors are the bridge between discrete symbols and continuous computation. Tokens start as IDs, become embeddings, move through transformer layers as vectors, and eventually produce logits over the next token.

Understanding vectors makes attention, embeddings, retrieval, and similarity search much easier to reason about.

Related: [embedding models](/glossary/embedding-models), [attention](/glossary/attention), [vector databases](/glossary/vector-databases).
