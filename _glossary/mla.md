---
title: "mla"
source: obsidian
---

MLA, or Multi-head Latent Attention, is an attention design that compresses the key/value state used during inference.

It matters because KV-cache memory is one of the main bottlenecks in long-context and high-concurrency LLM serving.

Standard multi-head attention stores keys and values for previous tokens across layers and heads. During decoding, those cached tensors are read repeatedly.

MLA changes the representation so the model stores a smaller latent form and reconstructs or projects the needed attention components from it.

The high-level idea is:

```text
standard attention:
store full K/V cache

MLA-style attention:
store compressed latent attention state
recover useful K/V-like information when needed
```

The exact implementation depends on the model architecture.

Long context is often limited by KV-cache memory, not only by model weights. Reducing cache size can improve:

```text
maximum practical context
batch size
concurrency
serving cost
decode memory pressure
```

This is why MLA is a deployment-relevant architecture change, not just a modeling detail.

MHA gives each attention head its own keys and values. MQA shares key/value heads more aggressively. GQA groups query heads so multiple query heads share fewer KV heads.

MLA is another step in the same broad direction: reduce the memory burden of attention state while preserving model quality.

Compression is not free. MLA changes the attention projections and may require specific kernel/runtime support to get the full serving benefit. It also affects model architecture compatibility, training behavior, and implementation complexity.

The practical question is:

```text
does the lower KV-cache cost outweigh the added architecture/runtime complexity?
```

Related: [attention](/glossary/attention), [kv cache](/glossary/kv-cache), [context window](/glossary/context-window), [inference](/glossary/inference), [deployment](/glossary/deployment).
