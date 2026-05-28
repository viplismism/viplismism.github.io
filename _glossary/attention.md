---
title: "attention"
source: obsidian
---

Attention is the mechanism that lets each token compute which other tokens are relevant for building its next representation. It is the core operation behind transformer models.

Instead of compressing the whole sequence into one hidden state, attention lets tokens directly compare against other tokens and mix information from the useful ones.

Each token is projected into three vectors:

```text
query = what this token is looking for
key   = what this token offers as a match
value = the information this token contributes
```

The query of one token is compared with the keys of other tokens. Those scores are scaled, passed through softmax, and used as weights over the value vectors.

In simplified form:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
$$

Read it as: compare queries against keys, turn the comparisons into weights, then use those weights to mix values.

Attention creates contextual representations. The word `bank` can represent a river bank or a financial bank depending on surrounding tokens. The token embedding alone is not enough; attention lets the model reshape meaning based on context.

Attention also explains why inference has a KV-cache problem. During decoding, previous keys and values are reused for every new token, so the serving system has to store and manage them efficiently.

Related: [vectors](/glossary/vectors), [embedding models](/glossary/embedding-models), [kv cache](/glossary/kv-cache), [mla](/glossary/mla), Transformers, yet another explanation.
