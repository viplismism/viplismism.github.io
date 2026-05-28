---
title: "reinforcement learning"
source: obsidian
---

Reinforcement learning is a training setup where an agent learns from rewards produced by actions in an environment.

Instead of only imitating examples, the system gets feedback about whether its behavior led to a good or bad outcome.

The basic objects are:

```text
state: what the agent observes
action: what the agent does
reward: feedback from the environment
policy: the rule/model that chooses actions
return: accumulated future reward
```

In LLMs, the environment may be a human preference model, test runner, verifier, game, tool system, or execution sandbox. The action may be a token, a full response, a tool call, or a sequence of steps.

RL matters when the desired behavior cannot be captured well by static supervised examples. It is central to agents, RLHF, RLOO, GRPO, reasoning models, tool use, and execution-based training.

The hard parts are credit assignment, reward design, variance, policy collapse, bad value estimates, exploration, and evaluation.

Related: [agents](/glossary/agents), [reasoning models](/glossary/reasoning-models), [small language models](/glossary/small-language-models), [fine tuning](/glossary/fine-tuning).
