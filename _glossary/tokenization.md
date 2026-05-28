---
title: "tokenization"
source: obsidian
---

Tokenization is the process of converting raw text into tokens that a model can process.

Models do not directly operate on words or characters. They operate on token IDs, which are mapped to vectors by the embedding layer.

A sentence like:

```text
the model is fast
```

may become token IDs like:

```text
[1820, 1646, 374, 5043]
```

The exact IDs and token boundaries depend on the tokenizer. Some words become one token. Some become multiple subword tokens. Spaces, punctuation, code symbols, and non-English text can change token counts significantly.

Tokenization affects cost, context usage, latency, and sometimes model quality. A 10k-character document may produce very different token counts across tokenizers.

For code models, tokenization also affects how identifiers, indentation, operators, and file paths are represented. Bad tokenization can make certain languages or formats more expensive for the model.

Related: [llm](/glossary/llm), [self supervised learning](/glossary/self-supervised-learning), [context window](/glossary/context-window).
