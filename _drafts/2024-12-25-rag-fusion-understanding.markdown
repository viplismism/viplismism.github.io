---
layout: post
title: Understanding RAG fusion
description: This post gives an adequate understanding of how the RAG fusion works.
tags: [RAG, LanceDB, RAG Fusion, Improvement in RAG]
version: Released
release: 12-25-2024
---

If you don't have any clue on what RAG is, please go through this [one](https://viplismism.github.io/2024/02/14/rag-application-with-langchain) to get a brief on what this is all about. 

So when the RAG model ends, RAG Fusion picks up by adding more layers that improve the RAG retrieval phase, particularly by adding more sophisticated mechanisms for interpretation and integration of the retrieval output. RAG Fusion tries to combat some of the weaknesses inherent also to RAG, including better response to ambiguous queries and returning more relevant, accurate information by improving the retrieval-to-generation loop.

## What was missed in RAG?

1. Constraints with Current Search Technologies: RAG is limited by the same things limiting our retrieval-based lexical and vector search technologies.
2. Human Search Inefficiencies: Humans are not great at writing what they want into search systems, such as typos, vague queries, or limited vocabulary, which often lead to missing the vast reservoir of information that lies beyond the obvious top search results. While RAG assists, it hasn’t entirely solved this problem.
3. Over-Simplification of Search: Our prevalent search paradigm linearly maps queries to answers, lacking the depth to understand the multi-dimensional nature of human queries. This linear model often fails to capture the nuances and contexts of more complex user inquiries, resulting in less relevant results.

## How the improvement really happens?

So basically when we talk about the traditional RAG, it works by ranking documents in the order of relevance to the query based on vector similarity distances, usually using cosine similarity.

RAG Fusion on the other hands addresses the challenges of document retrieval using

1. Query Transformation: Generates multiple new queries from different angels based on the original query and
2. Reciprocal Rank Fusion(RRF): Reranking the document relevance based on Reciprocal Rank Fusion(RRF)

That being said, when RAG Fusion receives the original query, it sends the original query to the large language model(LLM) to generate a number of new search queries based on the original query from different perspectives.

So what really happens is 
1. Query Duplication with a Twist: Translate a user’s query into similar, yet distinct queries via an LLM.
2. Vector Search Unleashed: Perform vector searches for the original and its newly generated query siblings.
3. Intelligent Reranking: Aggregate and refine all the results using reciprocal rank fusion.
4. Final Step:  Pair the cherry-picked results with the new queries, guiding the large language model to a crafted output that considers all the queries and the reranked list of results.

Now For all the documents retrieved from the vector database for each query, like a list of lists.

- Determine the rank of each document within its respective ranked list.
- For each document, compute the reciprocal of its rank (e.g., rank 1 → 1/1 = 1; rank 3 → 1/3).
- Sum the reciprocal ranks of each retrieved document across all generated queries.
- Order the documents based on their total aggregated scores to determine their final ranking.

And then now the top-ranked retrieved documents will be then sent to the LLM along with all the queries to generate a response.
