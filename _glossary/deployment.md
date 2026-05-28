---
title: "deployment"
source: obsidian
---

Deployment is the full system problem of running a model reliably for real users or real workflows.

It is not just starting a server. Deployment includes hardware, runtime, artifact format, quantization, serving limits, batching, scheduling, observability, scaling, security, and failure handling.

Deployment answers questions like:

```text
does the model fit?
is latency acceptable?
is throughput enough?
what is the cost per request?
how much context can we serve?
how many concurrent users can we handle?
what happens when requests fail?
how do we monitor quality and system health?
```

This is the difference between “the model is good” and “the system is usable.”

Local or edge deployment usually emphasizes:

```text
smaller models
quantization
GGUF artifacts
CPU or consumer GPU execution
low memory footprint
privacy and offline use
```

Server or API deployment usually emphasizes:

```text
GPU utilization
batching
concurrency
serving engines such as vLLM or SGLang
observability
autoscaling
request scheduling
```

The same model may require very different choices depending on the target environment.

Inference and deployment can be separated conceptually, but in practice they are welded together.

If KV cache grows too large, that is both an inference and deployment problem. If batching improves throughput but hurts per-user latency, that is a deployment tradeoff. If quantization makes a model fit but reduces task quality, that is a product decision.

Deployment balances:

```text
quality vs latency
long context vs memory
throughput vs responsiveness
cost vs user experience
simple stack vs hardware utilization
```

Context length is a deployment decision. A model may advertise a huge context window, but exposing that full limit can reduce concurrency and increase latency because prefill and KV-cache memory grow with sequence length.

Concurrency changes everything. One request may fit easily. Many mixed-length requests can create memory pressure, scheduling complexity, and unstable latency.

This is why serving engines matter. They manage batching, request admission, KV-cache allocation, prefix reuse, and runtime scheduling.

Architecture choices show up during deployment. [moe](/glossary/moe) changes routing and expert serving. [mla](/glossary/mla) changes KV-cache economics. [quantization](/glossary/quantization) changes memory and hardware requirements. Long-context models change prefill and cache pressure.

Deployment is where architecture stops being a paper detail and becomes an operational constraint.

Related: [inference](/glossary/inference), [vllm](/glossary/vllm), [gguf](/glossary/gguf), [ggml](/glossary/ggml), [quantization](/glossary/quantization), [context window](/glossary/context-window).
