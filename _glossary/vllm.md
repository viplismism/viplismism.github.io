---
title: "vllm"
source: obsidian
---

vLLM is an LLM inference and serving engine focused on high-throughput generation.

It is not a model. It is the runtime layer that loads model weights, schedules requests, manages KV cache, batches work, and exposes a serving interface.

vLLM matters because real LLM serving is irregular. Requests arrive at different times, have different prompt lengths, generate different output lengths, and hold different amounts of KV-cache state.

The core serving problems are:

```text
request scheduling
continuous batching
KV-cache allocation
memory fragmentation
prefix reuse
throughput under concurrency
latency control
```

vLLM is best known for PagedAttention, which manages KV cache in blocks/pages rather than requiring each sequence to occupy one contiguous memory allocation.

Naive batching forms a batch and waits until it finishes. That is inefficient for LLM serving because requests have different generation lengths.

Continuous batching lets the engine admit new requests while existing requests are decoding. This keeps the GPU more utilized under live traffic.

The mental model is:

```text
training batch: fixed group of examples
serving batch: dynamic group of active requests
```

PagedAttention applies a virtual-memory-like idea to KV cache. Instead of allocating one large contiguous cache per request, the cache is split into blocks. A sequence can map to multiple blocks.

This helps with:

```text
fragmentation
variable sequence lengths
concurrency
long-context serving
prefix sharing
```

The key idea is that smarter cache management improves throughput because KV cache is one of the main dynamic memory costs in inference.

vLLM is useful when serving many requests, long contexts, or high-throughput workloads. A notebook-style generation loop may work for one prompt, but it wastes hardware under real traffic.

For SLM projects such as PatchGuard, vLLM may become useful when many PRs or code changes need test generation concurrently. The model may be small, but the system still needs structured serving, batching, and observability.

vLLM is usually a GPU-serving choice. For small local CPU-first usage, GGUF/llama.cpp-style stacks may be simpler. For NVIDIA peak performance, TensorRT-LLM may be better. For structured agent workflows, SGLang may be attractive.

The correct runtime depends on workload, hardware, latency target, and deployment environment.

Related: [inference](/glossary/inference), [deployment](/glossary/deployment), [kv cache](/glossary/kv-cache), [context window](/glossary/context-window), [quantization](/glossary/quantization).
