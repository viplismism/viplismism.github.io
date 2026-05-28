---
title: "quantization"
source: obsidian
---

Quantization stores or computes model values in lower precision than the original training format.

The goal is to reduce memory, bandwidth, and sometimes latency. The tradeoff is possible quality loss, especially when precision is reduced too aggressively or applied to sensitive parts of the model.

Model weights are usually trained in formats such as FP32, BF16, or FP16. Quantization represents those values using fewer bits:

```text
FP16/BF16: common training and inference precision
FP8: lower precision used on modern accelerators
INT8: common inference quantization
INT4: aggressive weight compression
NF4: 4-bit format often used in QLoRA-style training
FP4: newer very-low precision direction
```

Quantization usually maps real-valued tensors into a smaller discrete range using scales and sometimes zero points.

A simplified affine quantization looks like:

$$
q
=
\text{round}
\left(
\frac{x}{s}
\right)
$$

Read it as: divide the real value by a scale, round it into a lower-precision value, and store that cheaper representation.

Weight quantization reduces the memory footprint of model parameters. This helps the model fit on smaller hardware and reduces memory bandwidth pressure.

KV-cache quantization reduces the memory footprint of dynamic attention state during inference. This matters for long context and high concurrency because the KV cache grows with active tokens and active requests.

These are different problems:

```text
weight quantization: makes the model artifact smaller
KV-cache quantization: makes active serving state smaller
```

A model can have quantized weights and still run out of memory from KV cache under long-context traffic.

Post-training quantization applies quantization after the model is trained. It is easier to use but may lose quality if the calibration or format is poor.

Quantization-aware training trains or adapts the model while simulating low precision. It can preserve quality better, but it is more expensive.

LoRA/QLoRA-style training keeps most model weights frozen and trains small adapters while loading the base model in quantized form. This makes fine-tuning much cheaper.

Quantization is one of the main reasons local LLMs and cheaper SLM deployment are practical. It can turn an impossible memory requirement into a workable one.

It affects:

```text
VRAM usage
CPU/GPU memory bandwidth
model loading time
batch size
context length
concurrency
latency
quality
```

The right quantization depends on workload. A chat model, code model, embedding model, long-context model, and CI test-generation model may tolerate different precision levels.

Quantization can damage rare-token behavior, math, code accuracy, multilingual performance, long-context stability, and tool-call formatting. Small models may be more sensitive because they have less spare capacity.

The correct question is not “what is the smallest quantization?” The correct question is:

```text
what is the lowest precision that still preserves task quality?
```

Related: [inference](/glossary/inference), [deployment](/glossary/deployment), [gguf](/glossary/gguf), [small language models](/glossary/small-language-models), [kv cache](/glossary/kv-cache).
