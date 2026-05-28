---
title: "self supervised learning"
source: obsidian
---

Self-supervised learning is training where the data itself provides the target label.

For language models, the most common objective is next-token prediction: given previous tokens, predict the next token.

The model sees text like:

```text
the capital of France is Paris
```

and training examples are created automatically:

```text
input:  the capital of France is
target: Paris
```

No human has to label every example. The sequence already contains the answer because the next token is known.

Self-supervised learning is what makes foundation-scale pretraining possible. It turns massive raw corpora into training data.

The limitation is that next-token prediction teaches language patterns, not necessarily helpfulness, safety, correctness, or task-following. That is why pretraining is usually followed by SFT, preference tuning, or RL-style post-training.

Related: [llm](/glossary/llm), [tokenization](/glossary/tokenization), [fine tuning](/glossary/fine-tuning).
