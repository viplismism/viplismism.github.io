---
title: "chain of thoughts"
source: obsidian
---

Chain of thought is the idea that a model can solve a problem through intermediate reasoning steps instead of jumping directly from prompt to answer.

The core point is not that the reasoning text must always be visible. The important idea is that some tasks benefit from intermediate computation.

For a simple factual question, direct answering may be enough. For math, planning, code debugging, or multi-hop reasoning, the model may need to decompose the task:

```text
understand the question
identify known facts
derive intermediate steps
check consistency
produce final answer
```

There are several variants: visible chain-of-thought prompting, hidden reasoning, scratchpads, process supervision, verifier-guided reasoning, and distilled reasoning traces.

Chain-of-thought-style behavior is one form of inference-time compute. The model spends more tokens or internal computation to improve the answer. This connects directly to reasoning models, test-time scaling, verifiers, and agent workflows.

The risk is that visible reasoning can be verbose, wrong, or misleading. A model can produce convincing reasoning text while still making an error. So reasoning traces should be evaluated by outcomes, not just by whether they look thoughtful.

Related: [reasoning models](/glossary/reasoning-models), [reinforcement learning](/glossary/reinforcement-learning), [agents](/glossary/agents).
