---
title: "llm"
source: obsidian
---

A large language model is a neural network trained to predict the next token from previous tokens. That definition sounds small, but the training distribution is enormous: text, code, math, markup, conversations, documentation, tables, instructions, and many other sequence patterns.

The objective is simple, but the compression problem is rich. To predict tokens well, the model learns syntax, style, facts, code structure, task formats, conversational patterns, and many statistical regularities in the training data.

An LLM should be understood at three layers:

```text
mathematical layer: probability distribution over next tokens
model layer: tokenizer + transformer architecture + learned weights
system layer: inference runtime + KV cache + batching + deployment interface
```

Confusion happens when these layers get mixed. A model can have strong weights but a weak runtime. A model can have good benchmark scores but be too slow or expensive for a product. A model can be strong as a base model but bad as a chat assistant if it was not instruction tuned.

During generation, the loop is:

```python
tokens = tokenize(prompt)

while not stop_condition(tokens):
    logits = model(tokens)
    next_token = sample(logits[-1])
    tokens.append(next_token)

return detokenize(tokens)
```

Real inference engines optimize this loop with [kv cache](/glossary/kv-cache), batching, scheduling, quantization, and specialized kernels. But conceptually, autoregressive generation is still token-by-token prediction.

A base model is trained mainly to continue text. It may complete documents, code, or dialogue, but it is not necessarily optimized to follow explicit user instructions.

An instruct or chat model is further adapted through supervised fine-tuning, preference tuning, RLHF, or other post-training methods. It learns response format, conversational behavior, refusal behavior, tool schemas, and instruction-following patterns.

This distinction matters in evaluation. A base model may look broken if tested like a chatbot. An instruct model may be better for assistants but less pure as a continuation model.

In practice, an LLM is not only weights. A deployable model includes:

```text
tokenizer
architecture config
learned parameters
context limit
precision format
inference runtime
sampling strategy
deployment target
```

The tokenizer affects token count, context usage, KV-cache growth, and latency. The architecture affects memory and compute. The runtime affects batching and throughput. Quantization affects memory and sometimes quality.

So “which model should I use?” is usually not the first question. Better questions are:

```text
what task is this for?
what quality is required?
what latency is acceptable?
how much memory is available?
how long is the context?
how many concurrent users exist?
can the model be quantized?
which runtime will serve it?
```

LLMs are general sequence models, not only chatbots. Chat is one interface on top of the model. The same model class can be used for code completion, summarization, classification, extraction, translation, retrieval-augmented generation, planning, and agents.

The most useful mental model is:

```text
LLM = learned sequence model
chatbot = product interface
agent = control loop around a model
deployment = system that makes the model usable
```

Keeping those layers separate prevents shallow thinking.

Related: [tokenization](/glossary/tokenization), [attention](/glossary/attention), [self supervised learning](/glossary/self-supervised-learning), [fine tuning](/glossary/fine-tuning), [inference](/glossary/inference), [deployment](/glossary/deployment).
