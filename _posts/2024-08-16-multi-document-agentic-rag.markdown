---
layout: post
title: Multi Document Agentic RAG
description: This post shows how you can create a Multi Document Agentic RAG using LanceDB.
tags: [RAG, AI Agents, LanceDB]
version: Released
release: 16-08-2024
---

Agentic RAG (Retrieval-Augmented Generation) represents a significant leap in how we handle information. Traditional [RAG](https://viplismism.github.io/2024/02/14/rag-application-with-langchain) systems are built to retrieve information and relevant context which are then passively send that to a language model (LLM) to generate responses. However, Agentic RAG takes this further by adding more independence. Now, the system can not only gather data but also make decisions and take actions on its own.

Think of it as a shift from simple tools to smarter, more capable systems. Agentic RAG transforms what was once a passive process into an active one, where AI can work towards specific goals without needing constant guidance.

### How Does an Agentic RAG Work?

To understand Agentic RAG, let's first break down what an "agent" is. Simply put, an agent is a smart system capable of making decisions on its own. When given a question or task, it figures out the best way to handle it by breaking the task into smaller steps and using the right tools to complete it.

![Multi Document Agentic RAG](/images/multi-document-agentic-rag/whole-process.png)

Now, when we talk about Agentic RAG, we’re taking this concept further. Instead of just retrieving information like regular RAG, Agentic RAG uses intelligent strategies to ensure the system provides the best possible response. It doesn't stop at giving a basic, context-aware reply. Instead, these agents think through the query, select the right approach, and deliver a more thoughtful and refined answer. It's like having a team of smart agents working together to solve a problem, one step at a time.

Here’s how the process works:

1. **Understanding the Query**: The agent starts by analyzing the question to grasp its specifics. It looks at the context, purpose, and intent to determine what information is needed.
2. **Using Memory**: The agent checks its memory for relevant information from past tasks that might assist with the current query.
3. **Choosing Tools**: Agentic RAG agents are designed to be intuitive. After analyzing the query, they evaluate available tools and resources to select the best method for retrieving the precise information. It’s like having a savvy assistant who knows exactly where to look, even for the most challenging questions.

### How to Use it?

Let’s dive into developing an automotive-themed RAG agent based on our current understanding.

Consider a scenario where you own a vehicle and require assistance with tasks ranging from diagnosing issues to planning routine maintenance. Now imagine a specialized agent designed specifically for this purpose. This agent should be capable of interpreting your car’s symptoms, analyzing the issue, and delivering a detailed diagnosis, including potential causes.

Furthermore, the bot should assist with identifying specific parts, estimating repair costs, or creating a personalized maintenance schedule based on your vehicle’s mileage and model. This agent must effectively manage a range of tasks by utilizing relevant context from various datasources, employing different tools, and reasoning with the available information to provide accurate and meaningful responses. Given the complexity of these tasks, which involves multiple retrieval and reasoning steps, I am going to use [LanceDB](https://viplismism.github.io/2024/03/15/embedded-databases) to ensure fast retrieval by storing relevant embedded data chunks into it.

To meet our data needs, I will use six JSON files, each containing specific types of information for querying. You can get these JSON files here : [Data](https://github.com/lancedb/vectordb-recipes/tree/main/examples/multi-document-agentic-rag/json_files?ref=blog.lancedb.com)

Here’s a brief overview of each file:

1. **car_maintenance.json**: Contains details about the car's maintenance schedule, including relevant tasks based on mileage and estimated time for completion.
2. **car_problems.json**: Provides information on various car problems, including a brief description of each issue, required parts for resolution, estimated repair time, and other relevant metadata.
3. **car_parts.json**: Lists car parts used for maintenance and diagnosis, detailing their brands, categories, whether they are solid or liquid, and other relevant attributes.
4. **car_diagnosis.json**: Outlines the diagnosis process, potential causes based on symptoms, recommended actions, and related problems. The issues and parts mentioned should align with those in **car_problems.json** and **car_parts.json** to ensure the agent has relevant context for problem-solving.
5. **car_cost_estimates.json**: Provides cost estimates for addressing car problems, based on the issues listed in **car_problems.json**.
6. **car_models.json**: Contains information on common problems associated with specific car models, such as a 2017 Honda Accord with 190,000 kilometers, detailing typical issues that users might encounter in this range.

Please review the JSON files to see their structure and feel free to make any changes.

### How to: Tech Stack

To build our ReAct-like agent, we’ll be using a few key tools to make things run smoothly:

1. **LlamaIndex**: Think of [LlamaIndex](https://www.llamaindex.ai/) as the backbone of our agent. This framework will be central to our implementation. It facilitates the abstraction of agentic logic. If the specifics seem unclear now, they will become more evident as we proceed with the implementation.

2. **Memory Management**: When we query the agent, it handles each question on its own without remembering past interactions or in isolation without maintaining state. To address this, we will use memory to retain conversational history. The agent stores chat history in a conversational memory buffer, which by default is a flat list managed by LlamaIndex. This ensures that the agent can refer to past as well as current conversation when deciding on the next set of actions.

3. **Vector Databases**: For the retrieval process, we will use VectorDBs. Queries will be embedded and matched semantically against the relevant VectorDB through our retrievers. We will employ [LanceDB](https://github.com/lancedb/lancedb) due to its exceptional retrieval speed and on-disk storage capabilities, which allows for local management of our database. Additionally, being open-source and free, it fits within our budget constraints.

4. **LLM Integration**: On the language model side, we’ll go with OpenAI’s GPT-4 for generating responses. For embeddings, we’re using Hugging face which provides seamless integration of local embedding models.

Ok I think this is enough, let's dive in for our code part..

### Environment Setup

```python
import os
import tqdm
import json
import time
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import logging
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext,
    Settings,
    Document,
)
from datetime import datetime, timedelta
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool, ToolOutput
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# LLM setup
llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# Embedding model setup
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update the Settings with the new embedding model
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

Make sure your `.env` file includes the `OPENAI_API_KEY`. You can adjust and use different LLMs and embedding models as needed. The key is to have an LLM for reasoning and an embedding model to handle data embedding. Feel free to experiment with various models to find the best fit for your needs.

### Step 1 : Creating our DBs

Let's setup our database which will be used to store our data.

```python
# Vector store setup
problems_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='problems_table',
    mode="overwrite",
)

parts_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='parts_table',
    mode="overwrite",
)

diagnostics_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='diagnostics_table',
    mode="overwrite",
)

cost_estimates_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='cost_estimates_table',
    mode="overwrite",
)

maintenance_schedules_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='maintenance_schedules_table',
    mode="overwrite",
)

cars_vector_store = LanceDBVectorStore(
    uri='./lancedb',
    table_name='car_maintenance_table',
    mode="overwrite",
)
```

Since we’re dealing with various types of information, we’ll need multiple tables to organize and retrieve our data effectively. The `uri` specifies that our data is stored in a database called `lancedb`, which contains different tables for each type of data. Let’s go ahead and load the data into the appropriate tables.

```python
def load_and_index_documents(directory: str, vector_store: LanceDBVectorStore) -> VectorStoreIndex:
    """Load documents from a directory and index them."""
    documents = SimpleDirectoryReader(input_dir=directory).load_data()
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context)

def create_retriever(index: VectorStoreIndex) -> VectorIndexRetriever:
    """Create a retriever from the index."""
    return index.as_retriever(similarity_top_k=5)

# Load and index documents
problems_index = load_and_index_documents("../rag-agentic-system/problems", problems_vector_store)
parts_index = load_and_index_documents("../rag-agentic-system/parts", parts_vector_store)
cars_index = load_and_index_documents("../rag-agentic-system/cars_models", cars_vector_store)
diagnostics_index = load_and_index_documents("../rag-agentic-system/diagnostics", diagnostics_vector_store)
cost_estimates_index = load_and_index_documents("../rag-agentic-system/cost_estimates", cost_estimates_vector_store)
maintenance_schedules_index = load_and_index_documents("../rag-agentic-system/maintenance_schedules", maintenance_schedules_vector_store)


problems_retriever = create_retriever(problems_index)
parts_retriever = create_retriever(parts_index)
cars_retriever = create_retriever(cars_index)
diagnostics_retriever = create_retriever(diagnostics_index)
cost_estimates_retriever = create_retriever(cost_estimates_index)
maintenance_retriever = create_retriever(maintenance_schedules_index)
```

Each Vector DB will provide a retriever instance, a Python object that returns a list of documents matching a given query. For example, our problems_retriever will fetch documents related to car problems based on the query, while cars_retriever will help identify common issues faced by customers with their vehicles.

Keep in mind that if the bot misses some information or seems to hallucinate, it might be due to missing data in our JSON files. If you spot any inaccuracies or gaps, add the relevant data to the JSON files and re-index them to ensure everything stays up to date.

Now, let’s test our retrievers to ensure they’re working correctly.

### Step 2 : Testing our retrievers

Let's test our `cost_estimates_retriever` to ensure it's working properly. We’ll use a query related to a brake problem and check if the retriever returns the correct documents about cost estimates. Additionally, I'll verify if our query engine is accurately interpreting the query and if, by providing the relevant documents, we are receiving the correct response.

```python
query = "My brake pad isn't working or I don't know, but the brakes are poor, and by the way, what's the cost for the solution?"
query_engine = cost_estimates_index.as_query_engine()
response = query_engine.query(query)
results = cost_estimates_retriever.retrieve(query)

print(f"Response: {response}")

for result in results:
    print(f"Result - Node ID: {result.node_id}")
    print(f"Relevant Text: {result.text[:100]}...")  
    print(f"Score: {result.score:.3f}")
```

This is the output I'm receiving in the cell

```text
Response: The cost for the solution to address the issue with your brake pad would be around $100 to $300 for brake pad replacement.
Result - Node ID: 51b91e6e-1243-405b-a742-89e09d78616f
Relevant Text: [
    {
        "repair": "Brake pad replacement",
        "average_cost": 150,
        "cost_range": {
            "min": 100,
            "max": 300...
Score: 0.584
```

Everything looks good—our retriever is working as expected. We’re ready to move on to the next step.

### Step 3 : Creating our agentic tools

LlamaIndex Agents are designed to process natural language input to execute actions rather than simply generating responses. The effectiveness of these agents relies on how well we abstract and utilize tools. So, what exactly does "tool" mean in this context? To clarify, imagine tools as weapons given to a warrior in battle. Just as a warrior might choose different weapons based on the opponent's tactics, tools for our agent are like specialized API interfaces that help the agent interact with data sources or reason through queries to deliver the best possible responses.

In LlamaIndex, there are various types of tools. One important type is the `FunctionTool`, which transforms any user-defined function into a tool, capable of inferring the function’s schema and usage. These tools are essential for our agents, allowing them to reason about queries and perform actions effectively.

For each tool, it’s essential to provide a clear description of its purpose and functionality, as this helps the agent use the tool effectively. To start, we will create tools to leverage the retriever objects defined earlier.

```python

max_context_information = 200

def retrieve_problems(query: str) -> str:
    """Searches the problem catalog to find relevant automotive problems for the query."""
    docs = problems_retriever.retrieve(query)
    information = str([doc.text[:max_context_information]for doc in docs])
    return information
    
def retrieve_parts(query: str) -> str:
    """Searches the parts catalog to find relevant parts for the query."""
    docs = parts_retriever.retrieve(query)
    information = str([doc.text[:max_context_information]for doc in docs])
    return information

def retrieve_car_details(make: str, model: str, year: int) -> str:
    """Retrieves the make, model, and year of the car."""
    docs = car_details_retriever.retrieve(make, model, year)
    information = str([doc.text[:max_context_information]for doc in docs])

def diagnose_car_problem(symptoms: str) -> str:
    """Uses the diagnostics database to find potential causes for given symptoms."""
    docs = diagnostics_retriever.retrieve(symptoms)
    information = str([doc.text[:max_context_information]for doc in docs])
    return information

def estimate_repair_cost(problem: str) -> str:
    """Provides a cost estimate for a given car problem or repair."""
    docs = cost_estimates_retriever.retrieve(problem)
    information = str([doc.text[:max_context_information]for doc in docs])
    return information

def get_maintenance_schedule(mileage: int) -> str:
    """Retrieves the recommended maintenance schedule based on mileage."""
    docs = maintenance_retriever.retrieve(str(mileage))
    information = str([doc.text[:max_context_information]for doc in docs])
    return information

retrieve_problems_tool = FunctionTool.from_defaults(fn=retrieve_problems)
retrieve_parts_tool = FunctionTool.from_defaults(fn=retrieve_parts)
diagnostic_tool = FunctionTool.from_defaults(fn=diagnose_car_problem)
cost_estimator_tool = FunctionTool.from_defaults(fn=estimate_repair_cost)
maintenance_retriever_tool = FunctionTool.from_defaults(fn=get_maintenance_schedule)
```

With the retriever tools now set up, our agent can effectively select the appropriate tool based on the query and fetch the relevant contexts. Next, we'll create additional helper tools that will complement the existing ones, providing the agent with more context and enhancing its reasoning capabilities.

```python
def comprehensive_diagnosis(symptoms: str) -> str:
    """
    Provides a comprehensive diagnosis including possible causes, estimated costs, and required parts.
    
    Args:
        symptoms: A string describing the car's symptoms.
    
    Returns:
        A string with a comprehensive diagnosis report.
    """
    # Use existing tools
    possible_causes = diagnose_car_problem(symptoms)
    
    # Extract the most likely cause (this is a simplification)
    likely_cause = possible_causes[0] if possible_causes else "Unknown issue"
    
    estimated_cost = estimate_repair_cost(likely_cause)
    required_parts = retrieve_parts(likely_cause)
    
    report = f"Comprehensive Diagnosis Report:\n\n"
    report += f"Symptoms: {symptoms}\n\n"
    report += f"Possible Causes:\n{possible_causes}\n\n"
    report += f"Most Likely Cause: {likely_cause}\n\n"
    report += f"Estimated Cost:\n{estimated_cost}\n\n"
    report += f"Required Parts:\n{required_parts}\n\n"
    report += "Please note that this is an initial diagnosis. For accurate results, please consult with our professional mechanic."
    
    return report

def plan_maintenance(mileage: int, car_make: str, car_model: str, car_year: int) -> str:
    """
    Creates a comprehensive maintenance plan based on the car's mileage and details.
    
    Args:
        mileage: The current mileage of the car.
        car_make: The make of the car.
        car_model: The model of the car.
        car_year: The year the car was manufactured.
    
    Returns:
        A string with a comprehensive maintenance plan.
    """
    car_details = retrieve_car_details(car_make, car_model, car_year)
    car_model_info = get_car_model_info(mileage, car_make, car_model, car_year)
    
    plan = f"Maintenance Plan for {car_year} {car_make} {car_model} at {mileage} miles:\n\n"
    plan += f"Car Details: {car_details}\n\n"
    
    if car_model_info:
        plan += f"Common Issues:\n"
        for issue in car_model_info['common_issues']:
            plan += f"- {issue}\n"
        
        plan += f"\nEstimated Time: {car_model_info['estimated_time']}\n\n"
    else:
        plan += "No specific maintenance tasks found for this car model and mileage.\n\n"
    
    plan += "Please consult with our certified mechanic for a more personalized maintenance plan."
    
    return plan

def create_calendar_invite(event_type: str, car_details: str, duration: int = 60) -> str:
    """
    Simulates creating a calendar invite for a car maintenance or repair event.
    
    Args:
        event_type: The type of event (e.g., "Oil Change", "Brake Inspection").
        car_details: Details of the car (make, model, year).
        duration: Duration of the event in minutes (default is 60).
    
    Returns:
        A string describing the calendar invite.
    """
    # Simulate scheduling the event for next week
    event_date = datetime.now() + timedelta(days=7)
    event_time = event_date.replace(hour=10, minute=0, second=0, microsecond=0)
    
    invite = f"Calendar Invite Created:\n\n"
    invite += f"Event: {event_type} for {car_details}\n"
    invite += f"Date: {event_time.strftime('%Y-%m-%d')}\n"
    invite += f"Time: {event_time.strftime('%I:%M %p')}\n"
    invite += f"Duration: {duration} minutes\n"
    invite += f"Location: Your Trusted Auto Shop, 123 Main St, Bengaluru, India\n\n"
    
    return invite

def coordinate_car_care(query: str, car_make: str, car_model: str, car_year: int, mileage: int) -> str:
    """
    Coordinates overall car care by integrating diagnosis, maintenance planning, and scheduling.
    
    Args:
        query: The user's query or description of the issue.
        car_make: The make of the car.
        car_model: The model of the car.
        car_year: The year the car was manufactured.
        mileage: The current mileage of the car.
    
    Returns:
        A string with a comprehensive car care plan.
    """
    car_details = retrieve_car_details(car_make, car_model, car_year)
    
    # Check if it's a problem or routine maintenance
    if "problem" in query.lower() or "issue" in query.lower():
        diagnosis = comprehensive_diagnosis(query)
        plan = f"Based on your query, here's a diagnosis:\n\n{diagnosis}\n\n"
        
        # Extract the most likely cause (this is a simplification)
        likely_cause = diagnosis.split("Most Likely Cause:")[1].split("\n")[0].strip()
        
        # Create a calendar invite for repair
        invite = create_calendar_invite(f"Repair: {likely_cause}", car_details)
        plan += f"I've prepared a calendar invite for the repair:\n\n{invite}\n\n"
    else:
        maintenance_plan = plan_maintenance(mileage, car_make, car_model, car_year)
        plan = f"Here's your maintenance plan:\n\n{maintenance_plan}\n\n"
        
        # Create a calendar invite for the next maintenance task
        next_task = maintenance_plan.split("Task:")[1].split("\n")[0].strip()
        invite = create_calendar_invite(f"Maintenance: {next_task}", car_details)
        plan += f"I've prepared a calendar invite for your next maintenance task:\n\n{invite}\n\n"
    
    plan += "Remember to consult with a professional mechanic for personalized advice and service."
    
    return plan
```

Additionally, we’ll implement some helper functions that, while not tools themselves, will be used internally within the tools to support the logic and enhance their functionality.

```python
def get_car_model_info(mileage: int, car_make: str, car_model: str, car_year: int) -> dict:
    """Retrieve car model information from cars_models.json."""
    with open('cars_models/cars_models.json', 'r') as file:
        car_models = json.load(file)

    for car in car_models:        
        if (car['car_make'].lower() == car_make.lower() and car['car_model'].lower() == car_model.lower() and car['car_year'] == car_year):
            return car
    return {}

def retrieve_car_details(make: str, model: str, year: int) -> str:
    """Retrieves the make, model, and year of the car and return the common issues if any."""
    car_details = get_car_model_info(0, make, model, year)  # Using 0 for mileage to get general details
    if car_details:
        return f"{year} {make} {model} - Common Issues: {', '.join(car_details['common_issues'])}"
    return f"{year} {make} {model} - No common issues found."
```

Here are the additional tools in their complete form

```python
comprehensive_diagnostic_tool = FunctionTool.from_defaults(fn=comprehensive_diagnosis)
maintenance_planner_tool = FunctionTool.from_defaults(fn=plan_maintenance)
calendar_invite_tool = FunctionTool.from_defaults(fn=create_calendar_invite)
car_care_coordinator_tool = FunctionTool.from_defaults(fn=coordinate_car_care)
retrieve_car_details_tool = FunctionTool.from_defaults(fn=retrieve_car_details)
```

Now, let's combine all these tools into a comprehensive tools list, which we will pass to our agent to utilize.

```python
tools = [
    retrieve_problems_tool,
    retrieve_parts_tool,
    diagnostic_tool,
    cost_estimator_tool,
    maintenance_schedule_tool,
    comprehensive_diagnostic_tool,
    maintenance_planner_tool,
    calendar_invite_tool,
    car_care_coordinator_tool,
    retrieve_car_details_tool
]
```

### Step 4 : Creating the Agent

Now that we’ve defined the tools, we’re ready to create the agent. With LlamaIndex, this involves setting up an Agent reasoning loop. Basically, this loop allows our agent to handle complex questions that might require multiple steps or clarifications. Essentially, our agent can reason through tools and complete tasks across several stages.

LlamaIndex provides two main components for creating an agent: `AgentRunner` and `AgentWorkers`.

The `AgentRunner` acts as the orchestrator, like in a symphony, managing the overall process. It handles the current state, conversational memory, and tasks, and it runs steps for each task while providing a high-level user interface on what's going on. On the other hand, `AgentWorkers` are responsible for the operational side. They select and use the tools and choose the LLM to interact with these tools effectively.

Now, let's set up both the AgentRunner and AgentWorker to bring our agent to life.

```python
# Function to reset the agent's memory
def reset_agent_memory():
    global agent_worker, agent
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools, 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

# Initialize the agent
reset_agent_memory()
```

Every time you call `reset_agent_memory()`, a new, fresh agent is created, ready to reason through and act on the user's query.

With everything now set up—our tools, an agent for reasoning, and databases for retrieving relevant context—let’s test to see if our agent can handle simple questions effectively.

### Step 5 : D-Day

Let's ask the agent a straightforward question related to car maintenance based on the mileage count and see how well it handles it.

```python
response = agent.query(
    "My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?"
)
```

and the response I got is

```python
Added user message to memory: My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?
=== Calling Function ===
Calling function: get_maintenance_schedule with args: {"mileage": 60000}

=== Calling Function ===
Calling function: estimate_repair_cost with args: {"problem": "Oil and filter change"}

=== Calling Function ===
Calling function: estimate_repair_cost with args: {"problem": "Tire rotation"}

=== Calling Function ===
Calling function: estimate_repair_cost with args: {"problem": "Air filter replacement"}

=== Calling Function ===
Calling function: estimate_repair_cost with args: {"problem": "Brake inspection"}

=== LLM Response ===
At 60,000 miles, the recommended maintenance tasks for your car are:

1. Oil and filter change: This typically costs around $250.
2. Tire rotation: The average cost for this service is around $50.
3. Air filter replacement: This usually costs about $70.
4. Brake inspection: The cost for this can vary, but it's typically around $100.

Please note that these are average costs and can vary based on your location and the specific make and model of your car. It's always a good idea to get a few quotes from different service providers.
```

Well, this is amazing! The agent effectively understood the query and provided an excellent response. Notice how it first called the `maintenance_schedule_tool`, which utilized the `get_maintenance_schedule` retriever object to gather context on the relevant maintenance schedule, including different tasks based on the car's mileage. This context was then used by the `cost_estimator_tool`.

The best part is that it passed the relevant parameters—problems extracted from the `maintenance_schedule_tool`—to the cost estimator tool, deciding on its own based on the user query. Finally, with all the gathered context, it produced a comprehensive response that perfectly addresses the user's needs.

Btw, If you want the agent to retain the context of previous conversations, replace `.query` with `.chat` to ensure context is preserved. Keep in mind that the context size is limited by the information you provide when calling the retrievers. Watch out for the `max_context_information` parameter in the retrievers to avoid exceeding the token limits for the LLMs.

And that's it! You've successfully created an agentic RAG that not only understands the user's query but also delivers a well-reasoned and contextually accurate answer. Here is the [colab](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/examples/multi-document-agentic-rag/main.ipynb) for this example.
