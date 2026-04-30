---
layout: page
title: Projects
---

<div class="projects-showcase">
  <p class="projects-note">
    A selected list of things I have built across agents, retrieval systems, multimodal AI, and applied deep learning. More writing and experiments are on the <a href="/">blog</a>.
  </p>

  <hr>

  <h2>Selected Projects</h2>

  <article class="project-entry">
    <a class="project-entry__media" href="https://github.com/rawwerks/rlm-cli">
      <img src="/images/swe-evals-work-tweet/retrieval-benchmark-cover.png" alt="RLM CLI retrieval benchmark visual">
    </a>
    <div class="project-entry__body">
      <h3><a href="https://github.com/rawwerks/rlm-cli">RLM CLI</a></h3>
      <p class="project-entry__links">
        <a href="https://github.com/rawwerks/rlm-cli">Code</a>
        <span>/</span>
        <a href="https://x.com/viplismism/status/2032103820969607500?s=20">Launch post</a>
      </p>
      <p>
        Recursive Language Model command-line tooling for querying codebases, reviewing diffs, analyzing files and URLs, and returning structured JSON-first results. Built around directory-as-context workflows, recursive decomposition, search, and support for multiple model backends.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2026/04/27/why-file-search-is-secretly-a-big-deal-for-coding-agents">
      <img src="/images/swe-evals-work-tweet/coding-agent-retrieval-results.png" alt="Coding agent retrieval evaluation results">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2026/04/27/why-file-search-is-secretly-a-big-deal-for-coding-agents">Coding Agent Retrieval Evals</a></h3>
      <p class="project-entry__links">
        <a href="/2026/04/27/why-file-search-is-secretly-a-big-deal-for-coding-agents">Writeup</a>
      </p>
      <p>
        Evaluation work around file search for coding agents, looking at how retrieval choices affect codebase understanding and downstream task performance. The project focuses on practical agent workflows rather than abstract benchmark scores alone.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="https://www.lancedb.com/blog/python-package-to-convert-image-datasets-to-lance-type">
      <img src="/images/cli-for-lance-converter/lancify-title-image.png" alt="Lancify package cover image">
    </a>
    <div class="project-entry__body">
      <h3><a href="https://www.lancedb.com/blog/python-package-to-convert-image-datasets-to-lance-type">Lancify: Image Dataset to Lance Converter</a></h3>
      <p class="project-entry__links">
        <a href="https://www.lancedb.com/blog/python-package-to-convert-image-datasets-to-lance-type">Writeup</a>
        <span>/</span>
        <a href="/2024/12/04/lance-converter-package">Local note</a>
      </p>
      <p>
        A package and CLI workflow for converting image datasets into Lance format so vision training pipelines can use faster loading, cleaner metadata handling, and LanceDataset-based training experiments.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2025/06/07/realesate-nlq-agent">
      <img src="/images/real-estate-nlq-agent/title_image.png" alt="Real estate natural language query agent architecture">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2025/06/07/realesate-nlq-agent">Real Estate NLQ Agent</a></h3>
      <p class="project-entry__links">
        <a href="/2025/06/07/realesate-nlq-agent">Writeup</a>
      </p>
      <p>
        A natural-language query agent for real estate search, combining structured filters with weighted vector search so users can ask fuzzy questions about budget, location, features, and intent.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2025/03/10/creating-a-superlinked-research-agent">
      <img src="/images/superlinked-research-agent/agent-meme.png" alt="Superlinked research agent project visual">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2025/03/10/creating-a-superlinked-research-agent">Superlinked Research Agent</a></h3>
      <p class="project-entry__links">
        <a href="/2025/03/10/creating-a-superlinked-research-agent">Writeup</a>
      </p>
      <p>
        A research paper agent using semantic similarity and recency signals to retrieve useful papers, balance freshness against relevance, and turn retrieval into a more controllable research workflow.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2025/02/23/create-a-fintech-agent">
      <img src="/images/fintech-ai-agent/fintech-ai-agent.excalidraw.png" alt="Fintech AI agent architecture">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2025/02/23/create-a-fintech-agent">Fintech AI Agent</a></h3>
      <p class="project-entry__links">
        <a href="/2025/02/23/create-a-fintech-agent">Writeup</a>
      </p>
      <p>
        An agentic fintech workflow combining loan prediction, semantic claim validation, and retrieval-backed decision support with LanceDB and classical machine-learning components.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2024/03/03/multimodal-rag-application">
      <img src="/images/multimodal_rag/multimodalrag.png" alt="Multimodal RAG architecture">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2024/03/03/multimodal-rag-application">Multimodal GTA V RAG</a></h3>
      <p class="project-entry__links">
        <a href="/2024/03/03/multimodal-rag-application">Writeup</a>
      </p>
      <p>
        A multimodal retrieval system over text and images using CLIP-style embeddings, LanceDB, and RAG patterns to query visual game data with natural-language prompts.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="/2024/05/31/movie-recommendation-system-with-rag-and-genre-classification">
      <img src="/images/movie-recommendation-using-rag/architecture_recommendation.png" alt="Movie recommendation system architecture">
    </a>
    <div class="project-entry__body">
      <h3><a href="/2024/05/31/movie-recommendation-system-with-rag-and-genre-classification">Movie Recommendation with RAG</a></h3>
      <p class="project-entry__links">
        <a href="/2024/05/31/movie-recommendation-system-with-rag-and-genre-classification">Writeup</a>
      </p>
      <p>
        A recommendation system that combines genre classification, retrieval, and LLM-powered explanations to turn movie search into a conversational recommendation workflow.
      </p>
    </div>
  </article>

  <article class="project-entry">
    <a class="project-entry__media" href="https://sketchgpt.art">
      <img src="/images/make_your_application_with_rag/cat.png" alt="SketchGPT placeholder sketch">
    </a>
    <div class="project-entry__body">
      <h3><a href="https://sketchgpt.art">SketchGPT</a></h3>
      <p class="project-entry__links">
        <a href="https://sketchgpt.art">Website</a>
      </p>
      <p>
        A small creative web app that turns prompts into simple black-and-white sketch-style images, built as an experiment in playful image generation interfaces.
      </p>
    </div>
  </article>
</div>
