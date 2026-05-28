---
title: "gguf"
source: obsidian
---

GGUF is a model file format commonly used by local inference runtimes such as llama.cpp.

It packages model tensors and metadata so a runtime can load and execute the model.

A GGUF file can contain:

```text
model weights
tensor metadata
tokenizer metadata
architecture information
quantization information
runtime-relevant config
```

The file format matters because local inference needs a portable artifact that contains enough information for the runtime to load the model correctly.

GGUF is not the same as GGML.

The practical distinction is:

```text
GGUF = file format / model artifact
GGML = runtime ecosystem / tensor execution lineage
```

People often mix the names because they are part of the same local inference world.

GGUF is important for local model deployment. If you download a quantized local model, it is often distributed as a GGUF file.

This is useful for:

```text
running models on laptops
CPU-first inference
Apple Silicon inference
private/offline deployment
quick experimentation with quantized models
```

Many GGUF files are quantized. The quantization level affects memory use, speed, and quality. Smaller quantizations make models easier to run but can reduce output quality, especially for reasoning, code, math, or long-context tasks.

The correct choice depends on hardware and task quality requirements.

Related: [ggml](/glossary/ggml), [quantization](/glossary/quantization), [deployment](/glossary/deployment), [inference](/glossary/inference).
