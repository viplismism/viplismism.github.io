---
title: "distillation"
source: obsidian
---

Distillation is a training method where a smaller student model learns from a larger or stronger teacher model.

The goal is to transfer useful behavior into a model that is cheaper to run, easier to deploy, or easier to fine-tune.

The teacher can provide:

```text
final answers
step-by-step traces
preference labels
logits or probability distributions
tool-use trajectories
synthetic datasets
```

The student is trained to imitate some part of that behavior. This can be simple SFT on teacher-generated responses, or more technical logit distillation where the student learns from the teacher's probability distribution over tokens.

Distillation is central to small language model work. A small model usually cannot learn every capability from raw data alone at reasonable cost. A teacher model can generate high-quality examples, reasoning traces, corrections, and domain-specific demonstrations.

The limitation is capacity. The student does not magically become the teacher. It can inherit patterns that fit inside its architecture and training data, but it may fail on tasks that require more capacity or broader knowledge.

Related: [small language models](/glossary/small-language-models), [fine tuning](/glossary/fine-tuning), [reasoning models](/glossary/reasoning-models).
