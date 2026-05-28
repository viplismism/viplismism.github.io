---
title: "ggml"
source: obsidian
---

GGML refers to the local inference runtime lineage associated with efficient model execution on CPUs and consumer hardware.

It is often mentioned alongside llama.cpp and GGUF, but the terms are not identical.

The useful distinction is:

```text
GGML: runtime / tensor library / local inference ecosystem lineage
GGUF: model file format used by that ecosystem
```

GGML-style local inference focuses on running models with limited hardware: CPUs, Apple Silicon, consumer GPUs, and mixed acceleration paths.

It became important because it made local LLM usage practical through quantized weights, efficient tensor operations, and simple deployment workflows.

Not every model deployment is a datacenter GPU deployment. Many workflows need local or private inference:

```text
personal machines
edge devices
offline tools
private enterprise environments
developer laptops
small automation systems
```

GGML/llama.cpp-style stacks matter because they make those deployments possible without requiring a full GPU serving cluster.

GGUF is the artifact format. It stores model tensors and metadata in a way local runtimes can load.

GGML is better thought of as the execution ecosystem and runtime lineage. GGUF is the file you often download and run inside that ecosystem.

Local inference is convenient and private, but it has limits. CPU or consumer-device inference may have lower throughput than GPU serving engines. Long context can still be expensive because KV-cache memory grows during generation. Quantization reduces weight memory but does not remove all runtime costs.

So local inference is not fake deployment. It is a different deployment target with different constraints.

Related: [gguf](/glossary/gguf), [deployment](/glossary/deployment), [quantization](/glossary/quantization), [inference](/glossary/inference).
