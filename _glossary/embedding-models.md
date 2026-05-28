---
title: "embedding models"
source: obsidian
---

An embedding model converts text, code, images, or other inputs into vectors that capture semantic information.

In LLM systems, embeddings are commonly used for retrieval, clustering, deduplication, similarity search, classification, and recommendation.

An embedding model maps an input to a fixed-size vector:

```text
"reset password email not sent" → [0.12, -0.44, ...]
```

Inputs with similar meaning should produce vectors that are close under a similarity metric such as cosine similarity or dot product.

Embedding models are not the same as generative LLMs. A generative LLM predicts tokens. An embedding model produces vector representations that are useful for search and comparison.

Inside an LLM, each token is represented by a learned embedding vector. A token ID by itself is just a discrete label. If `"once"` is token `123` and `"more"` is token `124`, the number `123.5` does not mean anything useful. Neural networks work with continuous vectors, so the token ID has to be mapped into a vector space first.

The embedding table is just a matrix with one row per token:

```text
vocab_size x embedding_dim
```

If token `1001` appears in the input, the model looks up row `1001` in that matrix. That row is the embedding for the token.

This is closely related to one-hot encoding. A one-hot vector for token `1001` would be a very long vector full of zeros, with a single `1` at position `1001`. If you multiply that one-hot vector by the embedding matrix, all the zero positions contribute nothing and the `1` selects row `1001`.

So these two operations are equivalent:

```text
one_hot_token_vector @ embedding_matrix
```

```text
lookup row token_id in embedding_matrix
```

In real code, the model does not usually build the huge one-hot vector. `nn.Embedding` directly fetches the right row. The one-hot explanation is useful because it shows that an embedding layer is basically a trainable input layer: one learned vector per token, updated during training like the rest of the model's weights.

Embeddings are the foundation of vector search and many RAG systems. Instead of matching exact words, the system can retrieve semantically related chunks.

The failure mode is that embedding similarity is not the same as task relevance. A chunk can be semantically similar but useless for the exact question. This is why reranking, metadata filters, keyword search, and context engineering often matter.

Related: [vectors](/glossary/vectors), [vector databases](/glossary/vector-databases), [rag](/glossary/rag).
