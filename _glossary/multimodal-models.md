---
title: "multimodal models"
source: obsidian
---

Multimodal models can process more than one modality, such as text, images, audio, video, or actions.

The important shift is that language becomes one interface among several, not the only input/output channel.

A multimodal system usually needs encoders or adapters that convert non-text inputs into representations the language model can use.

Examples:

```text
image → vision encoder → tokens/features → language model
audio → audio encoder → tokens/features → language model
video → frame/audio features → language model
```

Some models generate across modalities too, such as text-to-image, speech output, or action plans.

Many real tasks are not text-only. CAD, robotics, medical imaging, voice agents, document understanding, UI agents, and scientific workflows often require perception plus language.

The hard part is grounding. The model must not only talk about images/audio/actions; it must connect representations to actual visual, acoustic, spatial, or physical structure.

Related: [llm](/glossary/llm), [agents](/glossary/agents), [embedding models](/glossary/embedding-models).
