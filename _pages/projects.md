---
layout: page
title: Projects
---

### [rlm-cli](https://github.com/viplismism/rlm-cli)

a CLI implementation of Recursive Language Models ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). instead of stuffing your entire context into one LLM call, rlm writes code to actually process the data - slicing, chunking, running sub-queries on pieces and looping until it gets the answer. works with claude, gpt, gemini, auto-loads your project file tree as context. `npm i -g rlm-cli` and you're running in 30 seconds. 190+ stars on GitHub.

### [fim-coder-model](https://github.com/viplismism/fim-coder-model)

a FIM (fill-in-the-middle) training framework with AST-aware extraction for code completion. extracts semantic boundaries from Rust source files, generates targeted training samples, and fine-tunes using QLoRA. the idea is that AST-aware extraction gives you better training samples than just splitting by lines.

