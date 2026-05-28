---
title: "inference"
source: obsidian
---

Inference is using a trained model after training is complete. The weights are frozen, gradients are not computed, and the system is generating outputs from inputs.

This sounds simple, but inference is where a model becomes a systems problem. The question changes from “how do we make the model learn?” to “how do we make the model usable?”

Autoregressive LLM inference generates one token at a time:

```python
tokens = tokenizer.encode(prompt)

while True:
    logits = model(tokens)
    next_id = sample(logits[-1])
    tokens.append(next_id)
    if next_id == tokenizer.eos_id:
        break
```

Real engines do not recompute the full prefix every step. They use [kv cache](/glossary/kv-cache) so previous attention keys and values can be reused during decoding.

Inference has two main phases:

```text
prefill: process the prompt and build the KV cache
decode: generate new tokens one at a time using the cache
```

Prefill affects time to first token. Decode affects steady-state tokens per second and total generation time.

One “speed” number is not enough. Important metrics include:

```text
TTFT: time to first token
TPOT: time per output token
throughput: total tokens/sec across requests
latency: user-facing delay
memory usage: weights + KV cache + runtime overhead
concurrency: number of active requests
cache hit rate: prefix reuse effectiveness
```

A setup can have good throughput but poor first-token latency. Another can be fine for one user but fail under concurrency. Long-context workloads can be bottlenecked by prefill and cache memory even if short prompts look fast.

The model produces logits over the vocabulary. Decoding turns those logits into selected tokens.

Common strategies include:

```text
greedy decoding
temperature sampling
top-k sampling
top-p sampling
beam search
speculative decoding
structured decoding
```

Sampling affects output diversity, determinism, latency, and correctness. For structured systems, decoding may be constrained to valid JSON, tool calls, or schemas.

Inference at scale requires request scheduling, batching, memory management, quantization, monitoring, and failure handling.

Serving engines such as [vllm](/glossary/vllm) exist because real traffic is irregular. Requests have different prompt lengths, generation lengths, arrival times, and cache sizes. Efficient serving depends on continuous batching, KV-cache management, prefix caching, and good scheduling.

Inference is where model quality meets cost. A model that is strong but too slow, too memory-heavy, or too expensive may be a poor product choice.

For SLMs, inference is central to the value proposition: lower latency, lower cost, private deployment, and high-volume narrow workflows.

Related: [kv cache](/glossary/kv-cache), [vllm](/glossary/vllm), [deployment](/glossary/deployment), [quantization](/glossary/quantization), [context window](/glossary/context-window).
