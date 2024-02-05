---
layout: post
title: Create LLM apps using RAG
description: RAG and Langcahin for creating the personalized bots
summary: This blog post shows how you can create RAG applications with Langchain 
tags: [LLM,  Deep Learning]
---


![cat]()

*If you've ever thought about creating a custom bot for your documents or website that interacts based on specific data, you're in the right place. I'm here to assist you in developing a bot that leverages Langchain and RAG strategies for this purpose.*

### Understanding the Limitations of ChatGPT and LLMs

ChatGPTs and other Large Language Models (LLMs) are extensively trained on text corpora to comprehend language semantics and coherence. Despite their impressive capabilities, these models have limitations that require careful consideration for particular use cases. One significant challenge is the potential for hallucinations, where the model might generate inaccurate or contextually irrelevant information.

Imagine requesting the model to enhance your company policies; in such scenarios, ChatGPTs and other Large Language Models might struggle to provide factual responses because they lack training on your company's specific policies. Instead, they may generate nonsensical or irrelevant responses, which can be unhelpful. So, how can we ensure that an LLM comprehends our specific data and generates responses accordingly? This is where techniques like Retrieval Augmentation Generation (RAG) come to the rescue.

![RAG](https://github.com/vipul-maheshwari/vipul-maheshwari.github.io/blob/main/images/make_your_application_with_rag/LLM_without_RAG.png?raw=true)

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

1. We will be using [Langchain](https://python.langchain.com/docs/get_started/introduction) for this task, Basically it's like a wrapper which lets you talk and manage to your LLM operations better. 

2. Along with it we will be using [Hugging Face](https://huggingface.co/),  it is like an open-source library for building, training, and deploying state-of-the-art machine learning models, especially about NLP. To use the HuggingFace we need the access token, Get your access token [here](https://huggingface.co/docs/hub/security-tokens)

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

Finally create a data folder in the project's root directory, designated as the central repository for storing PDF documents. You can add some sample PDFs for testing purposes; for instance, I am using the [Yolo V7](https://arxiv.org/pdf/2207.02696.pdf) and [Transformers](https://arxiv.org/abs/1706.03762) paper for demonstration. It's important to note that this designated folder will function as our primary source for data ingestion. 

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

Picture the book as your extracted information, with each 10-page segment representing a small chunk of data, and the index as the embedding. Essentially, we'll apply an embedding model to these chunks to transform the information into their respective embeddings. While as humans, we may not directly comprehend or relate to these embeddings, they serve as numeric representations of the chunks to our application.  This is how you can do this in Python

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
documents = text_splitter.split_documents(docs)
```

Now the chunk_size parameter specifies the maximum number of characters that a chunk can contain, while the chunk_overlap parameter specifies the number of characters that should overlap between two adjacent chunks. With the chunk_overlap set to 50, the last 50 characters of the adjacent chunks will be shared between each other. This approach helps to prevent important information from being split across two chunks, ensuring that each chunk contains sufficient contextual information for the subsequent processing or analysis. As the shared information at the boundary of neighboring chunks enables a more seamless transition and understanding of the text's content. The best strategy for choosing the chunk_size and chunk_overlap parameters largely depends on the nature of the documents and the purpose of the application.

### Step 3 : Creating the embeddings and store them into a vectordatabase

We have two ways to get embeddings from these chunks. First, we can download a model, handle preprocessing, and do computations on our own. Or, we can use Hugging Face's model hub. They've got lots of pre-trained models for different NLP tasks, including embeddings. With this approach, we'll use one of their embedding models. We'll just give our chunks to this model, and Hugging Face's servers will do the hard work like preprocessing and computing. This saves us from doing it all on our own machines.

We have a bunch of options for embedding models, and you can check out the leaderboard [here](https://huggingface.co/spaces/mteb/leaderboard). But for now, we'll go with "bge-base-en-v1.5". It's great for making embeddings for English text, plus it's smaller, so it loads quickly and gets the job done fast.

```python
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

# Creating the embeddings object
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)
```

Here's a way to see the number of embeddings for each chunk

```python
query = "Hello I want to see the length of the embeddings for this document."
embeddings.embed_documents([query])[0]

# 768
```

When it comes to vector databases, there are plenty of options out there. Some, like Pinecone, are paid but offer fast performance and extra features compared to open-source alternatives like FAISS or Chroma. However, if you don't need to scale your database to handle hundreds of users fetching and storing data every minute, open-source options are more than sufficient. They work really well. So, what we'll do is create an instance of FAISS vector database and store our embeddings in it. It's straightforward and doesn't require any rocket science.

```python
from langchain_community.vectorstores import FAISS

# Creating a vectorstore object
vectorstore = FAISS.from_documents(documents, embeddings)
```

### Step 4 : Create a prompt template which will be fed to the LLM

Ok now comes the prompt template. So when you write a question to the ChatGPT and it answers that question, you are basically providing a prompt to the model so that it can understand what's the question is. When companies train the models, they decide what kind of prompt they are going to use for invoking the model and ask the question. So for example,if you are working with "Mistral 7B instruct" and you want the optimal results it's recommended to use the following chat template:

```python
<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
```

Note that <s> and </s> are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings. It's just that the Mistral 7B instruct is made in such a way that the model look for those special tokens to understand the question better. Different types of LLMs have different kinds of instructed prompts. 

Now for our case we are going to use "huggingfaceh4/zephyr-7b-alpha" which is a text generation model. Just to make it clear, Zephyr-7B-α has not been aligned or formated to human preferences with techniques like RLHF (Reinforcement Learning with Human Feedback) or deployed with in-the-loop filtering of responses like ChatGPT, so the model can produce problematic outputs (especially when prompted to do so). Instead of writing a Prompt on our own, I will use ChatPromptTemplate class which creates a prompt template for the chat models. Basically, instead of writing a specified prompt I am letting ChatPromptTemplate to do it for me. Here is an example prompt template that is being generated from the manual messsages.

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

messages = template.format_messages(
    name="Bob",
    user_input="What is your name?"
)
```

If you don't want to write the manual instructions, you can just use the *from_template* function to generate a more generic prompt template which I used for this project. Here it is..

```python
from langchain_core.prompts import ChatPromptTemplate

template = """
{query}
"""

prompt = ChatPromptTemplate.from_template(template)
```

Our prompt is set! We've crafted a single message, assuming it's from a human or you xD . If you're not using the from_messages function, the ChatPromptTemplate will ensure your prompt works seamlessly with the language model by reserving some additional system messages. While there's always room for improvement with more generic prompts to achieve better results, this setup should work for now!

### Step 5 : Convert the query to it's relevant embedding using same embedding model.

Now, let's talk about the query or question we want to ask our RAG application. We can't just pass the query to our model and expect information in return. Instead, we need to pass the query through the same embedding model used for the chunks earlier. Why is this important? Well, by embedding queries, we allow models to compare them efficiently with previously processed chunks of text. This enables tasks like finding similar documents or generating relevant responses. It's like translating language into a language computers understand.

It's like translating language into a language computers understand. Imagine you're an English speaker and your friend speaks Hindi. Neither of you understands each other's language. You hand your English-speaking friend a one-page document written in Hindi. Your friend has to translate that document into English before they can understand its content. Now, if you ask a question in Hindi related to that document, your friend has to translate the question into English first to understand it and find a relevant response from the information you shared earlier. In this scenario, your friend acts like an embedding model. They converted your previous texts into embeddings. Now, when you ask a query or question, it will be translated into the relevant embeddings using the same embedding model used for the chunks. Then, a search operation will be performed to find the relevant response to your query. I hope this clarifies why we converted our query into embeddings first.

That being said, any query or a question that you want to ask will first be collectilevy used for creating a generic prompt then the whole piece of text will be embedded using the same embedding model that you used earlier for the chunks. As the embedding is done, we can process further 

### Step 6 : Fetch K number of documents.

Now, let's talk about the retriever. Its job is to dive into the vector database and perform a search to find relevant documents. It returns a set number, let's call it "k", of these documents, which are ranked based on their contextual relevance to the query or question you asked. You can set "k" as a parameter, indicating how many relevant documents you want - whether it's 2, 5, or 10. Generally, if you have a smaller amount of data, it's best to stick with a lower "k", around 2. For longer documents or larger datasets, a "k" between 10 and 20 is recommended.

Different [search techniques](https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore) can be employed to fetch relevant documents more effectively and quickly from a vector database. The choice depends on various factors such as your specific use case, the amount of data you have, what kind of vector database you are using and the context of your problem.

```python
retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("what did you know about Yolo V7?")
print(docs)
```

When you run this code, the retriever will fetch 3 most most relevant documents from the vector database based on the mmr search criteria. All these documents will be the contexts for our LLM model to generate the response for our query.


### Step 7 : Pass the relevant documents to the LLM and get the response.

So far, we've asked our retriever to fetch a set number of relevant documents from the database. Now, we need a language model (LLM) to generate a relevant response based on that context. To ensure robustness, let's remember that at the beginning of this blog, I mentioned that LLMs like ChatGPT can sometimes generate irrelevant responses, especially when asked about specific use cases or contexts. However, this time, we're providing the context from our own data to the LLM as a reference. So, it will consider this reference along with its general capabilities to answer the question. That's the whole idea behind using RAG!

Now, let's dive into implementing the language model (LLM) aspect of our RAG setup. We'll be using a powerful model architecture from the Hugging Face Hub. Here's how we do it in Python:

```python
from langchain.llms import HuggingFaceHub

# Model architecture
model =  HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 4096 ,"max_new_tokens": 2048 }
)
```

In this code snippet, we're instantiating our language model using the Hugging Face Hub. Specifically, we're selecting the Zephyr 7 billion model which is placed in this repository ID "huggingfaceh4/zephyr-7b-alpha". Choice of choosing this model isn't arbitrary; it's based on the model's suitability for our specific task and requirements. As we are already implementing only Open Source components, Zephyr 7 billion works good enough to generate the useful response with minimal overhead and low latency.

This model comes with some additional parameters to fine-tune its behavior. We've set the temperature to 0.5, which controls the randomness of the generated text. As a lower temperature tends to result in more conservative and predictable outputs and    when the temperature is set to max which is 1, the model tries to be as much creative as it could, so based on what type of output you want for your use case, you can tweak this parameter. For the sake of the simplicity and demonstration purposes, I set it to 0.5 to make sure we get decent results. Next is max_length parameter, it defines the maximum length of the generated text which includes the size of your prompt as well as the response, and max_new_tokens which sets the maximum number of new tokens that can be generated. As a general rule of thumb, the max_new_tokens should always be less than or equal to the max_length parameter. Why? Think about it..

### Step 8 : Create a chain for invoking the LLM.

We have everything we want for our RAG application, last thing we need to do is to create a chain for invoking the LLM on our query to generate the response. There are different types of chains for the different types of use cases, if you like your LLM to remember the context of the chat over the time like the ChatGPT , you would need a memory instance which can be shared among multiple conversation pieces, for such cases, there are conversational chains available. 

For now we just need a chain which can combine our retrieved contexts and pass it with the query to the LLM to generate the response. 

```python
rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

We have our Prompt, model, context and the query! All of them are combined into a single chain. It's pretty much what all the chains does! Now before running the final code, I want to give a quick check on these two helper functions: 
`RunnablePassthrough()` and `StrOutputParser()`.  

The `RunnablePassthrough` class in `LangChain` serves to pass inputs unchanged or with additional keys. In our chain, a prompt expects input in the form of a map with keys "context" and "question." However, user input only includes the "question." or the "query".  Here, `RunnablePassthrough` is utilized to pass the user's question under the "question" key while retrieving the context using a retriever. It just ensures that the input to the prompt conforms to the expected format. 

Secondally, `StrOutputParser` is typically employed in RAG chains to parse the output of the model into a human-readable string. In the layman terms, It is responsible for transforming the model's output into a more coherent and grammatically correct sentence, which is generally better readable by Humans! That's it! 

### D-Day

![With_RAG](https://github.com/vipul-maheshwari/vipul-maheshwari.github.io/blob/main/images/make_your_application_with_rag/LLM_with_RAG.png?raw=true)

To make sure we get the entire idea even if the response gets cut off, I've implemented a function called `get_complete_sentence()`. Basically this function helps extract the last complete sentence from the text. So, even if the response hits the maximum token limit that we set upon and it gets truncated midway, we will still get a coherent understanding of the message.

For practical testing, I suggest storing some low sized PDFs in the data folder of your project. You can choose PDFs related to various topics or domains that you want the chatbot to interact with. Additionally, providing a URL as a reference for the chatbot can be helpful for testing. For example, you could use a Wikipedia page, a research paper, or any other online document relevant to your testing goals. During my testing, I used a URL containing information about Jon Snow from Game of Thrones,  and PDFs of Transformers paper, and the YOLO V7 paper to evaluate the bot's performance. Let's see how our bot performs in varied content.



```python
import os
from dotenv import load_dotenv

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# Loading the environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Loading the web url and breaking down the information into chunks
loader = WebBaseLoader("https://gameofthrones.fandom.com/wiki/Jon_Snow")
documents_loader = DirectoryLoader('../data/', glob="./*.pdf", loader_cls=PyPDFLoader)

# URL loader
url_docs = loader.load()

# Document loader
data_docs = documents_loader.load()

# Combining all the information into single variable
docs = url_docs + data_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
documents = text_splitter.split_documents(docs)

# Creating the embeddings object
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

# Creating a vectorstore object
vectorstore = FAISS.from_documents(documents, embeddings)

# Defining a retriever 
retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k": 3})

# Model architecture
model =  HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 4096 ,"max_new_tokens": 2048 }
)

template = """
Question : {query}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def get_complete_sentence(response):
    """
    Extracts the last complete sentence from a given text.
    
    Args: response (str): The input text from which the last complete 
    sentence will be extracted.
    
    Returns: str: The last complete sentence found in the input text. 
    If no complete sentence is found, returns the entire input text.
    """

    # Find the last occurrence of a period (.)
    last_period_index = response.rfind('.') 

    # If a period is found, return the text up to that point
    if last_period_index != -1: return response[:last_period_index + 1]  
    
    # If no period is found, return the entire response
    else: return response  

        
# Invoke the RAG chain and retrieve the response
response = rag_chain.invoke("Who killed Jon Snow?")

# Get the complete sentence
complete_sentence = get_complete_sentence(response)

print(complete_sentence)
```

This is the response I received after 10 seconds, which is quite good. The time it takes can vary depending on your system's configuration, but I believe you'll get decent results in just a few minutes. So, please be patient if it's taking a bit longer.

```python
Human: 
Question : Who killed Jon Snow?

Answer: 
In the TV series Game of Thrones, Jon Snow was stabbed by his 
fellow Night's Watch members in season 5, episode 9, 
"The Dance of Dragons." However, he was later resurrected by Melisandre 
in season 6, episode 3, "Oathbreaker." So, technically, 
no one killed Jon Snow in the show.
```

Have fun experimenting with various data sources! You can try changing the website addresses, adding new PDF files or maybe change the template a bit. LLMs are fun, you never know what you get! 

### What's next?
There are plenty of things we can adjust here. We could switch to a more effective embedding model for better indexing, try different searching techniques for the retriever, add a reranker to improve the ranking of documents, or use a more advanced LLM with a larger context window and faster response times. Essentially, every RAG application is just an enhanced version based on these factors. However, the fundamental concept of how RAG applications function always stays the same.
