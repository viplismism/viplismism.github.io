---
layout: post
title: Create a Movie Recommendation System using RAG
description: This post gives a detailed view on how you can use the embeddings to create a Movie Recommendation sys
tags: [LLM, RAG]
version: Released
release: 17-05-2024
---

### Movie recommendation system using VectorDB and Genre spectrum embeddings

This article provides a comprehensive guide on creating a scalable and production-grade movie recommendation system by leveraging the power of Genre Spectrum Embeddings and VectorDB. Inspired by the implementation outlined in this paper, We'll explore how combining these two techniques can significantly enhance the recommendation experience, addressing key challenges faced by traditional systems.

Here's what we cover below:

1. Introduction to Genre Spectrum Embeddings
2. Data Ingestion and preprocessing techniques for movie metadata
3. Generating embeddings using Doc2Vec 
4. Extracting the unique genre labels
5. Training and Testing a Neural Network for genre_classification task
6. A movie recommendation system
7. Using Genre Spectrum Embeddings to get the relevant recommendations
8. Get the relevant recommendation for a given Movie

Let's get started!

## Introduction to Genre Spectrum Embeddings

Scrolling through streaming platforms can be frustrating when the movie suggestions don't match our interests. Building recommendation systems is a complex task as there isn't one metric that can measure the quality of recommendations. To improve this, we propose combining Genre Spectrum Embeddings and VectorDB for better recommendations.

The Genre Spectrum approach involves combining the various movie genres or characteristics of a movie to form Initial embeddings, which offer a comprehensive portrayal of the movie content. Then these embeddings are used as a input to train a Deep Learning model producing Genre Spectrum embeddings at the penultimate layer. 

These embeddings serve dual purposes: they can either be directly inputted into a classification model for genre classification or stored in a VectorDB. By storing embeddings in a VectorDB, efficient retrieval and query search for recommendations become possible at a later stage. This architecture offers a holistic understanding of the underlying processes involved.

![image]()

## Data Ingestion and preprocessing techniques for movie metadata

Our initial task involves gathering and organizing information about movies. This includes gathering extensive details such as the movie's type, plot summary, genres, audience ratings, and more.

Thankfully, we have access to a robust dataset on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) containing information from various sources for approximately 45,000 movies. If you require additional data, you can supplement the dataset by extracting information from platforms like Rotten Tomatoes, IMDb, or even box-office records. 

Our next step is to extract the core details from this dataset and generate a universal summary for each movie. Initially, I'll combine the movie's title, genre, and overview into a single textual string. Then, this text will be tagged to create TaggedDocument instances, which will be utilized to train the Doc2Vec model later on.

Before moving forward, let's install the relevant libraries to make our life easier..

```python
pip install torch scikit-learn pylance nltk gensim lancedb scipy==1.12
```

Next, we'll proceed with the ingestion and preprocessing of the data. To simplify the process, we'll work with chunks of 1000 movies at a time. For clarity, we'll only include movie indices with non-null values for genres, accurate titles, and complete overviews. This approach ensures that we're working with high-quality, relevant data for our analysis.

```python
import tqdm
import pandas as pd

from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

movie_data = pd.read_csv('movies_metadata.csv', low_memory=False)
def preprocess_data(movie_data_chunk):
    tagged_docs = []
    valid_indices = []

    # Wrap your loop with tqdm
    for i, row in tqdm(movie_data_chunk.iterrows(), total=len(movie_data_chunk)):
        try:
            # Constructing movie text
            movies_text = ''
            movies_text += "Overview: " + row['overview'] + '\n'
            genres = ', '.join([genre['name'] for genre in eval(row['genres'])])
            movies_text += "Genres: " + genres + '\n'
            movies_text += "Title: " + row['title'] + '\n'
            tagged_docs.append(TaggedDocument(words=word_tokenize(movies_text.lower()), tags=[str(i)]))
            valid_indices.append(i)
        except Exception as e:
            continue

    return tagged_docs, valid_indices

chunk_size = 1000
tagged_data = []
valid_indices = []
for chunk_start in range(0, len(movie_data), chunk_size):
    movie_data_chunk = movie_data.iloc[chunk_start:chunk_start+chunk_size]
    chunk_tagged_data, chunk_valid_indices = preprocess_data(movie_data_chunk)
    tagged_data.extend(chunk_tagged_data)
    valid_indices.extend(chunk_valid_indices)
```

## Generating embeddings using Doc2Vec

Next, we'll utilize the Doc2Vec model to generate embeddings for each movie based on the preprocessed text. We'll allow the Doc2Vec model to train for several epochs to capture the essence of the various movies and their metadata in the multidimensional latent space. This process will help us represent each movie in a way that captures its unique characteristics and context.

```python
def train_doc2vec_model(tagged_data, num_epochs=10):
    model = Doc2Vec(vector_size=100, min_count=2, epochs=num_epochs)
    model.build_vocab(tqdm(tagged_data, desc="Building Vocabulary"))

    for epoch in range(num_epochs):
        model.train(tqdm(tagged_data, desc=f"Epoch {epoch+1}"), total_examples=model.corpus_count, epochs = model.epochs)

doc2vec_model = train_doc2vec_model(tagged_data)
doc2vec_model.save("doc2vec_model")
```

The `train_doc2vec_model` function trains a Doc2Vec model on the tagged movie data, producing 100-dimensional embeddings for each movie. These embeddings act as input features for the subsequent neural network, which we'll employ to generate the `genre_embeddings`.

Up to this point, we've gathered sufficient metadata for movies and consolidated genres, overviews, and titles. Then, leveraging the Doc2Vec model, we've generated meaningful representations of the movies based on their relevance.

Essentially, our Doc2Vec model has produced vectors and positioned them in a multidimensional space so that similar movies are closer to each other in the latent space. With our current training setup we are sure that the movies with identical genres and similar overviews will be positioned closer to each other in this space, reflecting their thematic and content similarities.

## Extracting the unique genre labels

Next, our focus shifts to compiling the names of relevant movies along with their genres. Subsequently, we'll employ a tool called `MultiLabelBinarizer` to transform these genres into a more comprehensible format.

To illustrate this, let's consider a movie with three genres: 'drama', 'comedy', and 'horror'. Using the `MultiLabelBinarizer`, we'll represent these genres with lists of 0s and 1s. If a movie belongs to a particular genre, it will be assigned a 1, and if it doesn't, it will receive a 0. Consequently, each row in our dataset will indicate which genres are associated with a specific movie. This approach simplifies the genre representation for easier analysis.

Let's take the movie "Top Gun Maverick" as a reference. We'll associate its genres using binary encoding. Suppose this movie is categorized only under 'drama', not 'comedy' or 'horror'. When we apply the `MultiLabelBinarizer` the representation would be: Drama: 1, Comedy: 0, Horror: 0. This signifies that "Top Gun Maverick" is classified as a drama but not as a comedy or horror.

We'll replicate this process for all the movies in our dataset to identify the unique genre labels present in our data.

## Training a NN for genre_classification task

We'll define a class called GenreClassifier, which encapsulates a neural network consisting of four linear layers with ReLU activations. The final layer utilizes softmax activation to generate probability scores for various genres. If your objective is primarily classification within the Genre Spectrum, where you input a movie description to determine its relevant genres, you can establish a threshold value for the multi-label softmax output. This allows you to select the top 'n' genres with the highest probabilities.

Here's the NN class, hyperparameter settings, and the corresponding training loop for training our model.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extracting unique genre labels from the valid movies
genres_list = []
for i in valid_indices:
    row = movie_data.loc[i]
    genres = [genre['name'] for genre in eval(row['genres'])]
    genres_list.append(genres)

# Using a MultiLableBinarier to encode the genre_labels with each movie
mlb = MultiLabelBinarizer()
genre_labels = mlb.fit_transform(genres_list)

# Creating the Training and Testing data for the model training
embeddings = []
for i in valid_indices:
    embeddings.append(doc2vec_model.dv[str(i)])
X_train, X_test, y_train, y_test = train_test_split(embeddings, genre_labels, test_size=0.2, random_state=42)

X_train_np = np.array(X_train, dtype=np.float32)
y_train_np = np.array(y_train, dtype=np.float32)
X_test_np = np.array(X_test, dtype=np.float32)
y_test_np = np.array(y_test, dtype=np.float32)

X_train_tensor = torch.tensor(X_train_np)
y_train_tensor = torch.tensor(y_train_np)
X_test_tensor = torch.tensor(X_test_np)
y_test_tensor = torch.tensor(y_test_np)

# Genre Classifier Neural Network
class GenreClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

# Step 1: Creating the Model Instance with adequate device
model = GenreClassifier(input_size=100, output_size=len(mlb.classes_)).to(device)

# Step 2: Defining the Model Hyperparameters
epochs = 50
batch_size = 64
learning_rate = 0.001

# Step 3: Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Step 4: Creating the DataLoaders
train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Step 5: Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# Step 6: Evaluation Time
model.eval()
with torch.no_grad():
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)  # Move test data to device
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Step 7 : Save the Model
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
```

That's it! We've successfully trained a Neural network for our genre classification task. Now, our next step is to take a query and generate the three most relevant genres based on our trained model.

```python
def test_model(movie_descriptions, doc2vec_model, model, mlb):
    tagged_docs = [TaggedDocument(words=word_tokenize(desc.lower()), tags=[str(i)]) for i, desc in enumerate(movie_descriptions)]
    embeddings = [doc2vec_model.infer_vector(doc.words) for doc in tagged_docs]
    X_test_np = np.array(embeddings, dtype=np.float32)
    X_test_tensor = torch.tensor(X_test_np).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
    
    # Get top N genres with the highest probabilities
    N = 3  # Number of top genres to select
    top_n_indices = np.argsort(-outputs.cpu().numpy())[:, :3]
    predicted_genres = mlb.classes_[top_n_indices]

    return predicted_genres


# Example movie descriptions to test the model
example_movie_descriptions = [
    "A young boy discovers a magical world filled with strange creatures and danger.",
    "In a dystopian future, a group of rebels fights against an oppressive government.",
    "A comedy about two people who meet on a train and embark on a whirlwind adventure."
]

test_model(example_movie_descriptions, doc2vec_model, model, mlb)
```

I trained the network on CPU using the same hyperparameter configuration as mentioned, and here are the results I obtained.

```python
[['Drama' 'Adventure' 'Thriller']
 ['Drama' 'Action' 'Thriller']
 ['Drama' 'Comedy' 'Thriller']]
```

So far, we've developed a comprehensive architecture for classifying genres based on a given query. But brace yourself, because we're about to delve into something more substantial.

When it comes to genre_embeddings, we can extract them from the final linear layer, which serves as the penultimate layer of the neural network. This layer encapsulates the entire context for each movie. In other words, we can extract the weights from the last linear layer and create vectors from those weights. These vectors will somehow encapsulate the information that the neural network captured from our input, which was the Doc2Vec embeddings of our movies metadata. 
 
Following this, we can store these embeddings elsewhere, such as in a VectorDB, for future utilization. This setup allows us to conduct query searches and deliver better recommendations by leveraging the stored genre_embeddings.

## A movie recommendation system

To build a movie recommendation system, we'll allow users to input a movie name, and based on that, we'll return relevant recommendations. To achieve this, we'll save the genre_embeddings in a Vector Database. When a user inputs a query in the form of a new movie, we'll first locate its genre_embeddings in our vector database. Once we have this, we'll find 'n' number of movies whose genre_embeddings are similar to ours. We can assess the similarity using various search algorithms like cosine similarity or by finding the least Euclidean distance.

For simplification, I've opted to use [LanceDB](https://lancedb.com/), an open-source vector database known for its blazing speed, high-level security (since our data remains local), versioning, and built-in search capabilities. We'll organize all the data into lists, with each list representing an individual movie. Subsequently, we'll create a CSV file containing relevant data.

```python
def extract_genre_embeddings(model, X_data):
    model.eval()
    with torch.no_grad():
        embeddings = model.fc3(model.relu(model.fc2(model.relu(model.fc1(X_data.to(device))))))
    return embeddings.cpu().numpy()

train_embeddings = extract_genre_embeddings(model, X_train_tensor)
test_embeddings = extract_genre_embeddings(model, X_test_tensor)

# Combine training and test data
all_indices = valid_indices[:len(X_train_tensor)] + valid_indices[len(X_train_tensor):]
all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
all_genres = np.concatenate((y_train_np, y_test_np), axis=0)

# Create a dataframe
movie_embeddings_df = pd.DataFrame({
    'movie_index': all_indices,
    'title': [movie_data.loc[idx, 'title'] for idx in all_indices],
    'genre_embeddings': [list(embeddings) for embeddings in all_embeddings],  # Convert each array of embeddings to a list
    'genre_labels': [mlb.classes_[labels.nonzero()[0]] for labels in all_genres]
})

# Save the data as a csv file
movie_embeddings_df.to_csv("movie_embeddings.csv", index=False)
```

And now, introducing LanceDB!

```python
import ast
import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector


data = pd.read_csv("movie_embeddings.csv")
data.drop(columns=["movie_index"], inplace=True)

movie_data = []
for index, row in data.iterrows():
    embedding_vector = ast.literal_eval(row["genre_embeddings"])
    movie_data.append(
        {
            "title": row['title'],
            "embeddings": embedding_vector,
            "genre_labels": row['genre_labels']
        }
    )

# Define LanceDB model
class Movie(LanceModel):
    title: str
    embeddings: Vector(128)  
    genre_labels: str

# Create LanceDB connection
db = lancedb.connect("./db")
movie_table = db.create_table(
    "movies", 
    schema=Movie,
    mode="Overwrite")
movie_table.add(movie_data)
```

The code might seem overwhelming at first glance, but it's not as complex as it appears. Essentially, we start by reading the essential data from our CSV file. Then, we establish a connection to LanceDB to set up our table and add our movie data

Each row in the table corresponds to a single movie, with columns storing data such as title, genres, overview, and embeddings. This columnar format offers a convenient way to store and retrieve information for a variety of tasks related to embeddings. 

## Using Genre Spectrum Embeddings to get the relevant recommendations.

And now, after all the groundwork, we've arrived at the final piece of the puzzle. Let's generate some relevant recommendations using `genre_embeddings` and LanceDB VectorDB.

```python
movie_data_pd = pd.DataFrame(movie_data)
def get_recommendation(title):
    result = (
        movie_table.search(movie_data_pd[movie_data_pd["title"] == title]["embeddings"].values[0]).metric('cosine')
        .limit(5)
        .to_pandas()
    )
    return result

get_recommendation("Toy Story")
```

![getrecommendation]()

Let me break down the things for you.
 
First, we initialize our LanceDB table containing our relevant data. Then, for a given movie title, we check if we have this title in our dataset. If it exists, we perform a cosine similarity search on all other movies, returning the top 5 most relevant movie titles.

And just like that, you've crafted an impressive movie recommendation system that can swiftly provide you with the best selection of movies. This is made possible by leveraging genre_embeddings created with the assistance of a neural network and storing them in LanceDB vector database.

Just to clarify, I've utilized a basic deep learning model architecture to demonstrate how we can enhance our recommendation system. We might not see the best results in terms of the movie recommendations, However, the folks at Tubi would be the right folks who can provide you with the precise model architecture they've employed to develop this.. 

Well keeping the things aside, Here's an overview of the entire training code..

```python
import pandas as pd
import numpy as np

import lance
import tqdm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Read data from CSV file
movie_data = pd.read_csv('movies_metadata.csv', low_memory=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(movie_descriptions, doc2vec_model, model, mlb):
    tagged_docs = [TaggedDocument(words=word_tokenize(desc.lower()), tags=[str(i)]) for i, desc in enumerate(movie_descriptions)]
    embeddings = [doc2vec_model.infer_vector(doc.words) for doc in tagged_docs]
    X_test_np = np.array(embeddings, dtype=np.float32)
    X_test_tensor = torch.tensor(X_test_np).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
    
    # Get top N genres with the highest probabilities
    N = 3  # Number of top genres to select
    top_n_indices = np.argsort(-outputs.cpu().numpy())[:, :N]
    predicted_genres = mlb.classes_[top_n_indices]

    return predicted_genres

def preprocess_data(movie_data_chunk):
    tagged_docs = []
    valid_indices = []

    # Wrap your loop with tqdm
    for i, row in tqdm(movie_data_chunk.iterrows(), total=len(movie_data_chunk)):
        try:
            # Constructing movie text
            movies_text = ''
            #movies_text += "Overview: " + row['overview'] + '\n'
            genres = ', '.join([genre['name'] for genre in eval(row['genres'])])
            movies_text += "Genres: " + genres + '\n'
            movies_text += "Title: " + row['title'] + '\n'
            tagged_docs.append(TaggedDocument(words=word_tokenize(movies_text.lower()), tags=[str(i)]))
            valid_indices.append(i)
        except Exception as e:
            continue

    return tagged_docs, valid_indices

from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

def train_doc2vec_model(tagged_data, num_epochs=10):
    # Initialize Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, min_count=2, epochs=num_epochs)
    doc2vec_model.build_vocab(tqdm(tagged_data, desc="Building Vocabulary"))
    for epoch in range(num_epochs):
        doc2vec_model.train(tqdm(tagged_data, desc=f"Epoch {epoch+1}"), total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    
    return doc2vec_model


# Preprocess data and extract genres for the first 1000 movies
chunk_size = 1000
tagged_data = []
valid_indices = []
for chunk_start in range(0, len(movie_data), chunk_size):
    movie_data_chunk = movie_data.iloc[chunk_start:chunk_start+chunk_size]
    chunk_tagged_data, chunk_valid_indices = preprocess_data(movie_data_chunk)
    tagged_data.extend(chunk_tagged_data)
    valid_indices.extend(chunk_valid_indices)

doc2vec_model = train_doc2vec_model(tagged_data)
doc2vec_model.save("doc2vec_model")

# Extract genre labels for the valid indices
genres_list = []
for i in valid_indices:
    row = movie_data.loc[i]
    genres = [genre['name'] for genre in eval(row['genres'])]
    genres_list.append(genres)

mlb = MultiLabelBinarizer()
genre_labels = mlb.fit_transform(genres_list)

embeddings = []
for i in valid_indices:
    embeddings.append(doc2vec_model.dv[str(i)])
X_train, X_test, y_train, y_test = train_test_split(embeddings, genre_labels, test_size=0.2, random_state=42)

X_train_np = np.array(X_train, dtype=np.float32)
y_train_np = np.array(y_train, dtype=np.float32)
X_test_np = np.array(X_test, dtype=np.float32)
y_test_np = np.array(y_test, dtype=np.float32)

X_train_tensor = torch.tensor(X_train_np)
y_train_tensor = torch.tensor(y_train_np)
X_test_tensor = torch.tensor(X_test_np)
y_test_tensor = torch.tensor(y_test_np)

class GenreClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

# Move model to the selected device
model = GenreClassifier(input_size=100, output_size=len(mlb.classes_)).to(device)

# Step 2: Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Step 4: Training loop
epochs = 10
batch_size = 64

train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

model.eval()
with torch.no_grad():
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)  # Move test data to device
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Saving the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Example movie descriptions to test the model
example_movie_descriptions = [
    "A boy discovers some powers and ready to conquer the world",
    "Ailens comes from the future and start destorying the earth",
    "A romantic comedy"
]

# Test the model
predicted_genres = test_model(example_movie_descriptions, doc2vec_model, model, mlb)
print(predicted_genres)


### Saving the relevant embeddings with the movie_index, title, genre_embeddings and the genre_labels to a dataframe
def extract_genre_embeddings(model, X_data):
    model.eval()
    with torch.no_grad():
        embeddings = model.fc3(model.relu(model.fc2(model.relu(model.fc1(X_data.to(device))))))
    return embeddings.cpu().numpy()

train_embeddings = extract_genre_embeddings(model, X_train_tensor)
test_embeddings = extract_genre_embeddings(model, X_test_tensor)

# Combine training and test data
all_indices = valid_indices[:len(X_train_tensor)] + valid_indices[len(X_train_tensor):]
all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
all_genres = np.concatenate((y_train_np, y_test_np), axis=0)

# Create a dataframe
movie_embeddings_df = pd.DataFrame({
    'movie_index': all_indices,
    'title': [movie_data.loc[idx, 'title'] for idx in all_indices],
    'genre_embeddings': [list(embeddings) for embeddings in all_embeddings],  # Convert each array of embeddings to a list
    'genre_labels': [mlb.classes_[labels.nonzero()[0]] for labels in all_genres]
})

# Save the data as a csv file
movie_embeddings_df.to_csv("movie_embeddings.csv", index=False)
```

Here is the ![collab]() for your reference..