---
layout: post
title: Create LLM apps using RAG
description: RAG and Langcahin for creating the personalized bots
summary: This blog post shows how you can create RAG applications with Langchain 
tags: [LLM,  Deep Learning]
---

Last Updated: 30/01/2024
<br>
Status: Working in progress

*If you've ever thought about creating a custom bot for your documents or website that interacts based on specific data, you're in the right place. I'm here to assist you in developing a bot that leverages Langchain and RAG strategies for this purpose.*

### Understanding the Limitations of ChatGPT and LLMs

ChatGPTs and other Large Language Models (LLMs) are extensively trained on text corpora to comprehend language semantics and coherence. Despite their impressive capabilities, these models have limitations that require careful consideration for particular use cases. One significant challenge is the potential for hallucinations, where the model might generate inaccurate or contextually irrelevant information.

Imagine requesting the model to enhance your company policies; in such scenarios, ChatGPTs and other Large Language Models might struggle to provide factual responses because they lack training on your company's specific policies. Instead, they may generate nonsensical or irrelevant responses, which can be unhelpful. So, how can we ensure that an LLM comprehends our specific data and generates responses accordingly? This is where techniques like Retrieval Augmentation Generation (RAG) come to the rescue.

### What is RAG?
RAG or Retrieval Augmented Generation uses three main workflows to generate and give the better response

- Information Retrieval: When a user asks a question, the AI system retrieves the relevant data from a well-maintained knowledge library or external sources like databases, articles, APIs, or document repositories. This is achieved by converting the query into a numerical format or vector that can be understood by machines.

- LLM: The retrieved data is then presented to the LLM or Large Language Model, along with the user's query. The LLM uses this new knowledge and its training data to generate the response.

- Response: Finally, the LLM generates a response that is more accurate and relevant since it has been augmented with the retrieved information. I mean we gave LLM some additional information from our Knowledge library which allows LLMs to provide more contextually relevant and factual responses, solving the problem of models when they are just hallucinating or providing irrelevant answer.

Let's take an example of company policies again. Suppose you have an HR bot that handles queries related to your Company policies. Now if someones asks anything specific to the policies, The bot can pull the most recent policy documents from the knowledge libray, pass the relevant context to a well crafted prompt which is then passed further to the LLM for generating the response. 

To make it more easy, Imagine a LLM as your knowledgeable friend who seems to know everything, from Geography to Computer Science from Politics to Philosophy. Now, picture yourself asking this friend a few questions:

- "Who handles my laundry on weekends?"
- "Who lives next door to me?"
- "What brand of peanut butter do I prefer?"

Chances are, your friend wouldn't be able to answer these questions, right? Most of the time, no.
But let's say this distant friend becomes closer to you over time, he comes at your place regularly, know your parents very well, you both hangout pretty often, you go on outings, blah blah blah.. You got the point right? I mean he is gaining access to personal and insider information about you.  Now, when you pose the same questions, they can somehow answer those question with more relevance now because because he is better suited with your personal insights.

Similarly, an LLM, when provided with additional information or access to your use case data, won't guess or hallucinate. Instead, it will leverage that data to provide relevant and accurate answers. 


### To break it down, here are the exact steps to create any RAG application...
1. Extract the relevant information from your data sources. 
2. Break the information into the small chunks.
3. Store the chunks as their embedddings into a vector database.
4. Create a prompt template which will be fed to the LLM with the query and the context.
5. Convert the query to it's relevant embedding using same embedding model.
6. Fetch k number of relevant documents related to the query from the vector database.
7. Pass the relevant documents to the LLM and get the response.

### FAQs

1. We will be using  <u><a href="https://python.langchain.com/docs/get_started/introduction">Langchain</a></u> for this task, Basically it's like a wrapper which lets you talk and manage to your LLM operations better. 

2. Along with it we will be using  <u><a href="https://huggingface.co/">Hugging Face</a></u>, it is like an open-source library for building, training, and deploying state-of-the-art machine learning models, especially about NLP. To use the HuggingFace we need the access token, Get your access token <u><a href="https://huggingface.co/docs/hub/security-tokens">here</a></u>

3. For our models, we'll need two key components: a LLM (Large Language Model) and an embedding model. While paid sources like OpenAI offer these, we'll be utilizing open-source models to ensure accessibility for everyone.

4. To keep things simple, our data ingestion process will involve using a URL and some PDFs. While you can incorporate additional data sources if needed, we'll concentrate solely on these two for now.

With Langchain for the interface, Hugging Face for fetching the models, along with open-source components, we're all set to go! This way, we save some bucks while still having everything we need. Let's move to the next steps

### Environment Setup

Open your favorite editor, create a python environment and install the relevant dependencies

```python
python3 -m venv env
source env/bin/activate
pip3 install dotenv langchain langchain_community 
```

Now create a .env file in the same directory to place your Hugging face api credentials like this

```python
HUGGINGFACEHUB_API_TOKEN = hf_KKNWfBqgwCUOHdHFrBwQ.....
```

Ensure the name "HUGGINGFACEHUB_API_TOKEN" remains unchanged, as it is crucial for authentication purposes.

Finally create a data folder in the project's root directory, designated as the central repository for storing PDF documents. You can add some sample PDFs for testing purposes; for instance, I am using the  <u><a href="https://arxiv.org/pdf/2207.02696.pdf">Yolo V7</a></u> and <u><a href="https://arxiv.org/abs/1706.03762">Transformers</a></u> paper for demonstration. It's important to note that this designated folder will function as our primary source for data ingestion. 

It seems like everything is in order, and we're all set!

### Step 1 : Extracting the relevant information

To get your RAG application running, the first thing we need to do is to extract the relevant information from the various data sources. It can be a website page, a PDF file, a notion link, a google doc whatever it is, it needs to be extracted from it's original source first. 

```python
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

# Loading the environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Loading the web url and data 
url_loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
documents_loader = DirectoryLoader('../data/', glob="./*.pdf", loader_cls=PyPDFLoader)

# Creating the instances
url_docs = url_loader.load()
data_docs = documents_loader.load()

# Combining all the data that we ingested
docs = url_docs + data_docs
```

This will ingest all the data from the URL link and the PDFs.

### Step 2 : Breaking the information into smaller chunks

We've all the necessary data for developing our RAG application. Now, it's time to break down this information into smaller chunks. Later, we'll utilize an embedding model to convert these chunks into their respective embeddings. But why it's important?

Think of it like this: If you're tasked with digesting a 100-page book all at once and then asked a specific question about it, it would be challenging to retrieve the necessary information from the entire book to provide an answer. However, if you're permitted to break the book into smaller, manageable chunks—let's say 10 pages each—and each chunk is labeled with an index or numbering from 0 to 9, the process becomes much simpler. When the same question is posed after this breakdown, you can easily locate the relevant chunk based on its index and then extract the information needed to answer the question accurately.

Picture the book as your extracted information, with each 10-page segment representing a small chunk of data, and the index as the embedding. Essentially, we'll apply an embedding model to these chunks to transform the information into their respective embeddings. While as humans, we may not directly comprehend or relate to these embeddings, they serve as numeric representations of the chunks to our application. 

### Step 3 : Creating the embeddings and store them into a vectordatabase

Now we are just going to pass those chunks to our embedding model and the embedding model will convert the chunks into their corresponding embeddings. There are two options for doing this, either you can download an embedding model in your local, manage the preprocessing and perform compuations locally and use it for the embedding creation,  or you can use Huggingface hub which hosts a vast collection of pre-trained models for various natural language processing (NLP) tasks, including embeddings. 

So what we are going to do is we will create an instance of an embedding model which is provided by Hugging Face, then we will pass the chunks to the embedding model and all the heavy stuff like handling preprocessing, offloading comupatation will be done on the Huggingface servers.

There are so many options for choosing a right embedding model, you can look at the ![https://huggingface.co/spaces/mteb/leaderboard]leaderboard here and choose the right embedding model according to your needs. For now, we will just use the "bge-base-en-v1.5" which works amazinglly well for creating the embeddings for english texts and it's compartively small so it's fast enough to load and do it's tasks. 



Ok so the embeddings are created. You could see how many number of embeddings are there for each chunk like this:

```python

```

Now comes the vectordatabase. There are ton of different vector databases available in the market, some of them are paid like Pinecone which works fast and have additional features over the ones which are open source like FAISS, Chroma. But to be honest, if you are not looking for anything like scaling your database to few hundred users who are fetching and storing the data from your database every minute, you are good to go with opensource ones. They works amazingingly well. So what we are going to do is we will create an instance of Chroma vector database, and will store our embeddins in it. That's it. No rocekt science here.


### Step 4 : Create a prompt template which will be fed to the LLM

Ok now comes the prompt template. So when you write a question to the ChatGPT and it answers that question, you are basically providing a prompt to the model so that it can understand what's the question is. When companies train the models, they decide what kind of prompt they are going to use for invoking the model and ask the question. So for example,if you are working with Mistral 7B instruct and you want the optimal results it's recommended to use the following chat template:

```python
<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
```

Note that <s> and </s> are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings. It's just that the Mistral 7B instruct is made in such a way that the model look for those special tokens to understand the question better. Different types of LLMs have different kinds of instructed prompts. For our case we will be working with zephyer and it's prompt is like as this:

```python

```

### Step 5 : Convert the query to it's relevant embedding using same embedding model.

Ok so far we have converted our textual information into chunks, then we embedded those chunks into there specific embeddings and further more we have stored our embedded chunks into the vector database. Now comes the query or the question that we want to ask to our RAG application. We just can't pass the query to our model and ask for the information, instead the query will be passed to the same embedding model that is being used for chunks earlier and then only we can do the next set of tasks. Why it's important? Wel by embedding queries, we enable models to efficiently compare them with previously processed chunks of text, as then only we can do tasks like finding similar documents or generating relevant responses. It's like translating language into a language computers understand. Imagine you are English Speaker and your friend is a Hindi Speaker, no one understands each other language. Now you give a 1 page document written in Hindi to your English Speaker friend, now your friend will convert that document into the Hindi one first and then will understand the information.

Now you asks a question to your english speaking friend in Hindi which is related to that documet. Now what would happen is, your friend will convert that question into the english first and then only will be able to understand the question and find relevant response from that information which you have shared earlier. 

Ok so your friend is an embedding model which converted your previous texts into the embeddings. Now when you asks a query or a question, it will be converted into the relevant embeddings from the same embedding model that is being used ealier for chunks and then some search operation will be performed to find the relevant response to your query. 

I hope you get it why we did convert our query into the embeddings first.

### Step 6 : Fetch K number of documents.

Now comes the retriver. What it does is, it goes into the vector database, performs *somekind of search* to find the relevant documents (chunks) and return k number of relevant documents which are ranked accordingly based on which one is more contextually related to the query or the question that you asked. We can set k as a parameter which means if you want 2 relevant documents or 5 or 10. Genearlly, if you have less amount of data, it's advisable to keep k to 2 and for longer documents, it's recommended to keep k in bewteen 10 - 20


### Step 7 : Pass the relevant documents to the LLM and get the response.

Now comes the last step. So to make things more robust, we will ask our retriever to fetch the k number of relevant documents from the database. We will then pass those k number of documents to our LLM as the context and will ask the LLM to generate a relevant response from that context. That's it. 


### What's next?
So there are so many things which can be tweaked here. We can change our embedding model for something more better and better in terms of indexing, searching technique for retriever can be changed too, as well as a better LLM to generate the relevant response. 