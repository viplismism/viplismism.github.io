---
title: "reasoning models"
source: obsidian
---

Reasoning models are language models optimized to spend more computation on tasks that require multi-step reasoning, planning, verification, or search.

They are not simply “regular models but smarter.” The important change is that the training or inference process encourages more deliberate intermediate computation.

Reasoning behavior can come from several mechanisms:

```text
chain-of-thought style training
process supervision
verifiers
reinforcement learning
search over candidate solutions
test-time compute scaling
tool-use loops
```

The output may or may not expose the reasoning trace. Some systems show intermediate reasoning; others keep it internal and only return the final answer.

Reasoning models are useful for math, code, planning, scientific tasks, multi-hop QA, and agent workflows. These tasks often fail when the model tries to answer in one shallow pass.

The tradeoff is cost and latency. More reasoning usually means more tokens, more tool calls, more sampling, or more verifier work.

Related: [chain of thoughts](/glossary/chain-of-thoughts), [reinforcement learning](/glossary/reinforcement-learning), [agents](/glossary/agents).
