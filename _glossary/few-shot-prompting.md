---
title: "few shot prompting"
source: obsidian
---

Few-shot prompting is a way to steer a model by including a small number of examples inside the prompt.

The model weights are not changed. The behavior is shaped at inference time by showing the model the pattern to follow.

A few-shot prompt usually contains:

```text
instruction
example input 1
example output 1
example input 2
example output 2
new input
```

The model infers the task format from the examples and applies it to the new input. This is useful for formatting, classification, extraction, style transfer, and structured outputs.

Few-shot prompting is often the fastest way to test whether a model can do a task before fine-tuning. If a task works with a few clean examples, fine-tuning may not be necessary.

The tradeoff is context cost. Examples consume tokens, and behavior may be unstable if examples are poorly chosen or conflict with the instruction.

Related: [context engineering](/glossary/context-engineering), [fine tuning](/glossary/fine-tuning), [inference](/glossary/inference).
