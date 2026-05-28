---
title: "moe"
source: obsidian
---

MoE, or Mixture of Experts, is an architecture where each token is routed through a subset of expert networks instead of using the same full feed-forward path for every token.

The goal is to increase total model capacity while keeping active compute per token lower than a dense model of the same total parameter count.

A typical MoE layer has:

```text
router: decides which experts a token should use
experts: separate feed-forward networks
top-k selection: choose one or a few experts per token
combine step: merge expert outputs back into the token representation
```

The model may have many total parameters, but only a subset is active for each token.

Example:

```text
total parameters: 1T
active parameters per token: 30B
```

This is why MoE models can look huge on paper while having lower active compute than a dense model with the same total size.

The router is central. It scores experts for each token and sends the token to the top experts.

Routing creates training and serving challenges:

```text
load balancing
expert collapse
capacity limits
token dropping
communication between devices
grouped GEMM
expert parallelism
```

If routing is imbalanced, some experts get overloaded while others are underused.

MoE is important because it changes the scaling equation. Dense models activate all parameters for every token. MoE models can add capacity without increasing active compute proportionally.

This can improve quality for a given compute budget, but serving becomes more complex. The runtime has to dispatch tokens to experts, batch expert computation efficiently, and sometimes communicate across GPUs.

MoE serving is harder than dense serving. It introduces routing overhead, irregular token-to-expert assignment, grouped GEMM, all-to-all communication, and expert-parallel scheduling.

A model can be architecturally efficient on paper but difficult to serve well if the runtime does not handle expert dispatch efficiently.

MoE gives more capacity and specialization potential, but it adds complexity:

```text
better parameter efficiency
more routing complexity
harder training stability
more complex serving
possible load imbalance
hardware communication overhead
```

The practical question is not only whether MoE improves benchmarks. The question is whether it can be trained and served efficiently for the target workload.

Related: [llm](/glossary/llm), [inference](/glossary/inference), [deployment](/glossary/deployment), [vllm](/glossary/vllm).
