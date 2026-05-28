---
title: "small language models"
source: obsidian
---

Small language models, or SLMs, are language models small enough to fine-tune, serve, or deploy more cheaply than frontier-scale LLMs.

There is no strict parameter cutoff, but models around `0.5B`, `1B`, `1.5B`, `3B`, and sometimes `7B` parameters are often treated as small depending on context.

An SLM is usually weaker than a frontier model in broad general capability, but it can be useful when specialized.

SLMs are attractive for:

```text
low latency
lower serving cost
private deployment
edge/local inference
domain-specific fine-tuning
high-volume narrow workflows
```

The key is specialization. A small model should not be judged only as a weaker general assistant. It should be evaluated on the narrow workflow it is trained for.

For enterprise systems, SLMs can be more practical than large models when the task is constrained, repeated, and sensitive to cost or privacy.

PatchGuard-SLM is an example: the goal is not to beat a frontier model at all coding, but to train a small model to generate regression tests reliably inside a specific workflow.

Related: [fine tuning](/glossary/fine-tuning), [distillation](/glossary/distillation), [deployment](/glossary/deployment), [inference](/glossary/inference).
