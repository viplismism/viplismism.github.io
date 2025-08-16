---
layout: page
title: About
---

I’m Vipul Maheshwari from Bharat, and I did my B.Tech in Computer Science.

My journey into the world of Artificial Intelligence and Deep Learning began in my second year when a senior student of mine was working on a classified project for the Indian Army. I volunteered in that project and I was tasked with object localization and creating a manual test set using bounding boxes. The project was a huge success, and it immediately grabbed my attention. I was amazed at how a blend of linear algebra, computational resources, and a dash of Python could work wonders.

Following that project, I embarked on a research apprenticeship with one of my professors. Our research focused on creating unified model predictors for neurological diseases. We delved into neuroimaging data related to Impulse Control Disorder (ICD) and developed an LSTM synthesizer capable of predicting the deterioration levels of patients. This research grabbed the attention of reviewers at ICICV, and we bagged the Best Paper Award.

Moving on, I had the opportunity to intern with [aiensured](https://www.aiensured.com/) as a computer vision intern. During this internship, I utilized YOLO, image slicing, and segmentation algorithms to build impressive machine learning datasets and pipelines for a wide range of tasks.

Soon after this internship, I joined [Saarthi.ai](https://www.saarthi.ai/) specializing in the creation of multilingual AI bots using deep learning algorithms as an NLP Deep Learning Intern. My role involved resolving numerous multilingual Natural Language Understanding (NLU) and Named Entity Recognition (NER) challenges for over 15 clients. I implemented NER entity extraction using rule-based regex and pattern dictionaries to enhance intent coverage. Additionally, I improved and refactored the Python scripts for NER data creation and the multilingual NLU code base for production-level NLP applications. I also conducted extensive tests on knowledge distillation, where I demonstrated the use of transfer learning to decrease the latency of production models and the usage of Large Language Models (LLMs) in large-scale conversational AI.

Now comes the time when I started to move to some big leagues.

My time with [Endeavor Labs](https://www.endeavorlabs.co/), a New York-based research and consultancy outfit, kicked off in an unexpected way. I posted a casual intro about my goals and what I was hoping to dive into on the MongoDB #general channel, and somehow that caught Nathan’s eye—though I can’t quite recall the exact message, but that lead to me working here.

At Endeavor, I jumped into a supervised machine learning project, tackling multiclass classification to boost operational efficiency. I leaned on to  [CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification) for build solid models, tweaking them with gradient optimization and some clever feature tweaks to keep overfitting aside. It was a hands-on effort that really sharpened the predictive edge and helped clients make smarter, data-backed calls. I mean the feature engineering was the big part here. The data was messy as **fcuk** but that project was a huge success. 

Along with this project, I also designed and implemented an OCR and NLP-based invoice processing system, leveraging Tesseract for optical character recognition and transformer-based NLP models, such as BERT, for entity extraction and semantic parsing. This system automated the extraction of structured data from unstructured invoices, utilizing custom tokenization pipelines and contextual embeddings to improve processing efficiency and accuracy.

moving ahead, I took up the role of Founding Engineer at [Bolna](https://www.bolna.ai/).

There, I worked on building the [knowledge base feature](https://www.linkedin.com/posts/vipulmaheshwarii_excited-to-share-what-ive-been-working-activity-7241083342593384448-rd4Q?utm_source=share&utm_medium=member_desktop&rcm=ACoAADWFgLQBXZygDPUaqFOS9b7G3wtkLnIxpIs) from the ground up—a RAG integration that plugged directly into our Voice AI orchestration engine. The whole thing came together by wiring up LanceDB as the vector store, using a proxy engine to manage and store knowledge base files, and setting up a queue-based flow so we could stream tokens smoothly from STT → LLM → TTS. That queue part was especially important to make sure the knowledge base responses worked in real-time during calls.

End result: you could basically connect your own knowledge base and have an AI agent handle live inbound and outbound calls with context—pretty close to a Jarvis-like experience. It was a short haul for me at Bolna, but one filled with some really exciting technical challenges.

Now comes [LanceDB](https://lancedb.com/).

I worked with the team as a Machine Learning Consultant, focusing on bridging LanceDB with deep learning and multimodal applications. A large part of this was around the deep learning recipes and integrations. Specifically, I worked on comparing PyTorch dataloaders vs Lance dataloaders, showing how Lance improves I/O throughput, reduces memory bottlenecks, and scales efficiently for large datasets. These experiments provided concrete benchmarks and reproducible training workflows for vision tasks.

On top of that, I made a [Lancify package](https://www.lancedb.com/blog/python-package-to-convert-image-datasets-to-lance-type), which converts raw image datasets into Lance format. This enabled cleaner, faster ingestion of large-scale vision datasets and plugged directly into the Lance dataloaders for downstream training.

To demonstrate real-world use cases, I built and documented several examples:

- [Multimodal GTA V RAG](https://vipul-maheshwari.github.io/2024/03/03/multimodal-rag-application)— combining text and images for retrieval and grounding with LanceDB.
- [Training CNNs with LanceDataset](https://vipul-maheshwari.github.io/2024/06/26/train-a-cnn-with-lancedataset) — showcasing end-to-end model training pipelines powered by Lance dataloaders.
- [Zero-shot image classification with CLIP + LanceDB](https://vipul-maheshwari.github.io/2024/07/14/zero-shot-image-classification) — leveraging Lance for scalable multimodal search and classification tasks.

And a few more along the same lines, always with the goal of showing how LanceDB can fit directly into deep learning workflows and not just as a standalone vector database. Check more of these [here](https://vipul-maheshwari.github.io/)

Additionally, here and there, I collaborated with consulting companies such as [Figr](https://figr.design/) or [Superlinked](https://superlinked.com/). At Figr, I worked on zero-shot and few-shot learning approaches to generate Figma components for text-to-Figma code, contributing to a custom library for Figr.

Currently, I am working with [Juspay](https://juspay.io/in) on the [Xyne team](https://xynehq.com/) where I am developing advanced AI systems focused on semantic search, information retrieval, and agent-based architectures. My work involves designing and implementing end-to-end Retrieval-Augmented Generation (RAG) pipelines, optimizing vector search for low-latency query processing, and building intelligent agentic frameworks. These efforts leverage state-of-the-art transformer models and knowledge graphs to enhance contextual understanding and scalability in production-grade AI applications.

Well that's it.. I am still not sure what I like most. So if you’re looking to collaborate on a project related to AI or ML, feel free to reach out—I’m just a message away!
