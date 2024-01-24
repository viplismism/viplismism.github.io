---
layout: post
title: Create LLM apps using RAG
description: RAG and Langcahin for creating the personalized bots
summary: This blog post shows how you can create RAG applications with Langchain 
tags: [LLM,  Deep Learning]
---

*If you've ever thought about creating a custom bot for your documents or website that interacts based on specific data, you're in the right place. I'm here to assist you in developing a bot that leverages Langchain and RAG strategies for this purpose.*

### Limitation of ChatGPT and LLMs
ChatGPTs and other Large Language Models (LLMs) undergo extensive training on corpora to understand language semantics and coherence. Despite their remarkable utility, these models have limitations that need consideration for specific use cases. One notable challenge is the potential for hallucinations, where the model may generate inaccurate or contextually irrelevant information. Imagine asking the model to improve your company policies; in such cases, ChatGPTs and other Large Language Models might struggle to provide a factual response because they aren't trained on your company's specific policies. Instead, they may produce nonsensical or irrelevant responses, which may not be helpful.

So, how can we make an LLM understand our specific data and generate responses accordingly? This is where techniques like Retrieval Augmentation Generation (RAG) come to the rescue.

### What's RAG really?
In a nutshell, RAG combines two key functionalities: retrieving information from a vast database and then generating responses using natural language processing. The process is akin to this: first, the relevant context is retrieved from a vector database and next, the retrieved information is used to create a specific prompt that is fed to the LLM (large language model) which then generates a response.

Let's take an example of company policies again. Suppose you have an HR bot that handles queries related to your Company policies. Now if someones asks anything specific to the policies, The bot can pull the most recent policy documents and summarize the relevant points in a conversational manner. This not only improves the quality of responses but also ensures that the information provided is up-to-date and in line with current company policies.

### To break it down, here are the steps to create any RAG application:
1. Extract the relevant information from your data sources. 
2. Break the information into the small chunks
3. Store the chunks as their embedddings into a vector database
4. Create a prompt template which will be fed to the LLM
5. Convert the query to it's relevant embedding using same embedding model.
6. Fetch k number of relevant documents related to the query from the vector database
7. Pass the relevant documents to the LLM with the specific prompt
8. Get the response.

We will do all this with LangChain, a framework for interfacing with LLMs to create chains of operations and autonomous agents.