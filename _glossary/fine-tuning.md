---
title: "fine tuning"
source: obsidian
---

Fine-tuning is continued training of a pretrained model on a narrower dataset or behavior target.

Pretraining gives the model broad language ability. Fine-tuning adapts that ability toward a task, domain, style, format, or workflow.

Common fine-tuning types include:

```text
SFT: supervised fine-tuning on prompt-response examples
LoRA/QLoRA: parameter-efficient fine-tuning through adapter weights
preference tuning: training from comparisons or rankings
RL post-training: optimizing outputs against rewards
```

The key difference from prompting is that fine-tuning changes model weights. The behavior becomes more internal to the model instead of being carried in the prompt every time.

Fine-tuning is useful when a behavior must be reliable, domain-specific, cheap at inference time, or repeated many times. It is especially relevant for SLMs because small models need task specialization to compete with larger general models.

Fine-tuning can also damage a model if the dataset is low quality, too narrow, or badly formatted. Evaluation before and after fine-tuning is not optional.

Related: [small language models](/glossary/small-language-models), [distillation](/glossary/distillation), [reinforcement learning](/glossary/reinforcement-learning).
