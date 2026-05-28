---
title: "kv cache"
source: obsidian
---

The KV cache stores the key and value tensors from previous tokens during autoregressive generation.

It exists so the model does not recompute attention state for the entire prefix every time it generates a new token.

In self-attention, each token produces queries, keys, and values. During decoding, the new token needs to attend to all previous tokens. The previous keys and values do not change, so they can be cached.

Without a KV cache, generation would repeatedly recompute the prefix:

```text
step 1: process prompt
step 2: process prompt + token_1
step 3: process prompt + token_1 + token_2
...
```

With a KV cache:

```text
process prompt once
store K/V tensors
for each new token:
    compute Q/K/V only for new token
    append new K/V to cache
    attend new Q over cached K/V
```

This is one of the main reasons modern LLM inference is practical.

KV-cache size depends on:

```text
batch size
sequence length
number of layers
number of KV heads
head dimension
precision
```

A rough shape is:

```text
[layers, batch, kv_heads, sequence_length, head_dim]
```

There are two tensors: keys and values. So memory grows with active tokens and active requests.

This is why long context and high concurrency are expensive even when model weights fit in memory.

During prefill, the model processes the prompt and fills the initial KV cache. During decode, the model appends one token's K/V tensors at each step.

Prefill can be compute-heavy for long prompts. Decode is often memory-bandwidth sensitive because every new token reads from cached K/V states.

KV cache is dynamic state. Weights are fixed, but active requests constantly allocate and extend cache memory.

This creates serving problems:

```text
memory fragmentation
variable request lengths
different generation lengths
cache eviction
prefix reuse
concurrency limits
```

Serving engines such as [vllm](/glossary/vllm) use paged/block-based KV-cache management to reduce fragmentation and support continuous batching.

Attention variants change the KV-cache story. MQA and GQA reduce the number of KV heads. [mla](/glossary/mla) compresses attention state differently. KV-cache quantization stores cache tensors in lower precision to reduce memory.

So KV cache is not just an implementation detail. It is one of the main reasons architecture and inference design are linked.

Related: [attention](/glossary/attention), [inference](/glossary/inference), [context window](/glossary/context-window), [vllm](/glossary/vllm), [mla](/glossary/mla).
