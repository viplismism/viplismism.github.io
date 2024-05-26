---
layout: post
title: Create a Movie Recommendation System using RAG
description: This post gives a detailed view on how you can use the embeddings to create a Movie Recommendation sys
tags: [LLM, RAG]
version: Published
release: 17-05-2024
---

This article provides a comprehensive guide on creating a movie recommendation system by leveraging the power of  Embeddings and VectorDB. We'll explore how combining these two techniques can significantly enhance the recommendation experience, addressing key challenges faced by traditional systems. 

Here's what we cover below:

1. Data Ingestion and preprocessing techniques for movie metadata
2. Generating embeddings using Doc2Vec 
3. Extracting the unique genre labels
4. Training and Testing a Neural Network for genre classification task
5. A movie recommendation system
6. Using Doc2Vec to get the relevant recommendations
7. Get the relevant recommendation for a given Movie

Let's get started!

### Why use Embeddings for Recommendation System?
Scrolling through streaming platforms can be frustrating when the movie suggestions don't match our interests. Building recommendation systems is a complex task as there isn't one metric that can measure the quality of recommendations. To improve this, we can combine Embeddings and VectorDB for better recommendations.

These embeddings serve dual purposes: they can either be directly inputted into a classification model for genre classification or stored in a VectorDB. By storing embeddings in a VectorDB, efficient retrieval and query search for recommendations become possible at a later stage. This architecture offers a holistic understanding of the underlying processes involved.

This architecture offers a holistic understanding of the underlying processes involved.

![image](https://github.com/vipul-maheshwari/vipul-maheshwari.github.io/blob/main/images/movie-recommendation-using-rag/doc2vec_final.png?raw=true)

### Data Ingestion and preprocessing techniques for movie metadata
Our initial task involves gathering and organizing information about movies. This includes gathering extensive details such as the movie's type, plot summary, genres, audience ratings, and more. 

Thankfully, we have access to a robust dataset on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) containing information from various sources for approximately 45,000 movies. Please download the data from the Kaggle and place it inside your working directory. Watch for a file named `movies_metadata.csv`. 

If you require additional data, you can supplement the dataset by extracting information from platforms like Rotten Tomatoes, IMDb, or even box-office records. 

Our next step is to extract the core details from this dataset and generate a universal summary for each movie. Initially, I'll combine the movie's title, genre, and overview into a single textual string. Then, this text will be tagged to create TaggedDocument instances which will be utilized to train the Doc2Vec model later on.

Before moving forward, let's install the relevant libraries to make our life easier..

```python
pip install torch scikit-learn pylance lancedb nltk gensim scipy==1.12
```

Next, we'll proceed with the ingestion and preprocessing of the data. To simplify the process, we'll work with chunks of 1000 movies at a time. For clarity, we'll only include movie indices with non-null values for genres, accurate titles, and complete overviews. This approach ensures that we're working with high-quality, relevant data for our analysis.

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import nltk
nltk.download('punkt')

# Read data from CSV file
movie_data = pd.read_csv('movies_metadata.csv', low_memory=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(movie_data_chunk):
    tagged_docs = []
    valid_indices = []
    movie_info = []

    # Wrap your loop with tqdm
    for i, row in tqdm(movie_data_chunk.iterrows(), total=len(movie_data_chunk)):
        try:
            # Constructing movie text
            movies_text = ''
            genres = ', '.join([genre['name'] for genre in eval(row['genres'])])
            movies_text += "Overview: " + row['overview'] + '\n' 
            movies_text += "Genres: " + genres + '\n'
            movies_text += "Title: " + row['title'] + '\n'
            tagged_docs.append(TaggedDocument(words=word_tokenize(movies_text.lower()), tags=[str(i)]))
            valid_indices.append(i)
            movie_info.append((row['title'], genres))
        except Exception as e:
            continue

    return tagged_docs, valid_indices, movie_info

# Preprocess data and extract genres for the first 1000 movies
chunk_size = 1000
tagged_data = []
valid_indices = []
movie_info = []
for chunk_start in range(0, len(movie_data), chunk_size):
    movie_data_chunk = movie_data.iloc[chunk_start:chunk_start+chunk_size]
    chunk_tagged_data, chunk_valid_indices, chunk_movie_info = preprocess_data(movie_data_chunk)
    tagged_data.extend(chunk_tagged_data)
    valid_indices.extend(chunk_valid_indices)
    movie_info.extend(chunk_movie_info)
```


### Generating embeddings using Doc2Vec

Next, we'll utilize the Doc2Vec model to generate embeddings for each movie based on the preprocessed text. We'll allow the Doc2Vec model to train for several epochs to capture the essence of the various movies and their metadata in the multidimensional latent space. This process will help us represent each movie in a way that captures its unique characteristics and context.

```python
def train_doc2vec_model(tagged_data, num_epochs=10):
    doc2vec_model = Doc2Vec(vector_size=100, min_count=2, epochs=num_epochs)
    doc2vec_model.build_vocab(tqdm(tagged_data, desc="Building Vocabulary"))
    for epoch in range(num_epochs):
        doc2vec_model.train(tqdm(tagged_data, desc=f"Epoch {epoch+1}"), total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    
    return doc2vec_model

doc2vec_model = train_doc2vec_model(tagged_data)
doc2vec_model.save("doc2vec_model")
```

The `train_doc2vec_model` function trains a Doc2Vec model on the tagged movie data, producing 100-dimensional embeddings for each movie. These embeddings act as input features for the Neural network. With our current training setup we are sure that the movies with identical genres and similar kind of overviews will be positioned closer to each other in the latent space, reflecting their thematic and content similarities.

### Extracting the unique genre labels

Next, our focus shifts to compiling the names of relevant movies along with their genres. Now we are going to use MultiLabelBinarizer to transform these genres into a more comprehensible format.

To illustrate this, let's consider a movie with three genres: 'drama', 'comedy', and 'horror'. Using the MultiLabelBinarizer, we'll represent these genres with lists of 0s and 1s. If a movie belongs to a particular genre, it will be assigned a 1, and if it doesn't, it will receive a 0. Now each row in our dataset will indicate which genres are associated with a specific movie. This approach simplifies the genre representation for easier analysis.

Let's take the movie "Top Gun Maverick" as a reference. We'll associate its genres using binary encoding. Suppose this movie is categorized only under 'drama', not 'comedy' or 'horror'. When we apply the MultiLabelBinarizer the representation would be: Drama: 1, Comedy: 0, Horror: 0. This signifies that "Top Gun Maverick" is classified as a drama but not as a comedy or horror. We'll replicate this process for all the movies in our dataset to identify the unique genre labels present in our data.
Training a Neural Network for genre classification task

We'll define a class called GenreClassifier, which uses a neural network consisting of four linear layers with ReLU activations. The final layer utilizes softmax activation to generate probability scores for various genres. If your objective is primarily classification within the Genre Spectrum, where you input a movie description to determine its relevant genres, you can establish a threshold value for the multi-label softmax output. This allows you to select the top 'n' genres with the highest probabilities.

Here's the Neural Network class, Hyperparameter settings, and the corresponding training loop for training our model.

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
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Adjust the dropout rate as needed

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Move model to the selected device
model = GenreClassifier(input_size=100, output_size=len(mlb.classes_)).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
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
```

That's it! We've successfully trained a Neural network for our genre classification task. Let's test how our model is performing for the genre classification task

```python
from sklearn.metrics import f1_score

model.eval()
with torch.no_grad():
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)  # Move test data to device
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')


thresholds = [0.1] * len(mlb.classes_) 
thresholds_tensor = torch.tensor(thresholds, device=device).unsqueeze(0)

# Convert the outputs to binary predictions using varying thresholds
predicted_labels = (outputs > thresholds_tensor).cpu().numpy()

# Convert binary predictions and actual labels to multi-label format
predicted_multilabels = mlb.inverse_transform(predicted_labels)
actual_multilabels = mlb.inverse_transform(y_test_np)

# Print the Predicted and Actual Labels for each movie
for i, (predicted, actual) in enumerate(zip(predicted_multilabels, actual_multilabels)):
    print(f'Movie {i+1}:')
    print(f'    Predicted Labels: {predicted}')
    print(f'    Actual Labels: {actual}')


# Compute F1-score
f1 = f1_score(y_test_np, predicted_labels, average='micro')
print(f'F1-score: {f1:.4f}')

# Saving the trained model
torch.save(model.state_dict(), 'trained_model.pth')
```

I got an approximate F1 score of 0.94 which is pretty reasonable. 

### A movie recommendation system

To build a movie recommendation system, we'll allow users to input a movie name, and based on that, we'll return relevant recommendations. To achieve this, we'll save our Doc2Vec embeddings in a Vector Database. When a user inputs a query in the form of a new movie, we'll first locate its embeddings in our vector database. Once we have this, we'll find 'n' number of movies whose embeddings are similar to our query movie.  We can assess the similarity using various search algorithms like cosine similarity or by finding the least Euclidean distance.

For simplification, I've opted to use LanceDB, an open-source vector database known for its blazing speed, high-level security (since our data remains local), versioning, and built-in search capabilities. We'll organize the data and the embeddings into a csv file.

```python
# Save movie titles, genres, and embeddings in a single CSV file
data = []
for i in valid_indices:
    embedding = doc2vec_model.dv[str(i)]
    title, genres = movie_info[valid_indices.index(i)]
    data.append([title, genres, embedding.tolist()])

df = pd.DataFrame(data, columns=['title', 'genres', 'embeddings'])
df.to_csv('movie_embeddings.csv', index=False)
```

### And now, introducing LanceDB!

```python
import lancedb

db = lancedb.connect(".db")
tbl = db.create_table("doc2vec_embeddings", data, mode="Overwrite")
db["doc2vec_embeddings"].head()

def get_recommendations(title):
    pd_data = pd.DataFrame(data)
    result = (
        tbl.search(pd_data[pd_data["title"] == title]["vector"].values[0]).metric("cosine")
        .limit(10)
        .to_pandas()
    )
    return result[["title", "genres"]]
```

The code might seem overwhelming at first glance, but it's not as complex as it appears. Essentially, we establish a connection to LanceDB to set up our table and add our movie data to it. 

Each row in the table corresponds to a single movie, with columns storing data such as title, genres, overview, and embeddings. And for each given movie title, we check if we have this title in our dataset or not. If it exists, we perform a cosine similarity search on all other movies, returning the top 10 most relevant movie titles. This columnar format offers a convenient way to store and retrieve information for a variety of tasks related to embeddings. 
Using Doc2Vec Embeddings to get the relevant recommendations.

And now, after all the groundwork, we've arrived at the final piece of the puzzle. Let's generate some relevant recommendations using embeddings and LanceDB.

```python
get_recommendations("Vertical Limit")
```

![movie_recommendation](https://github.com/vipul-maheshwari/vipul-maheshwari.github.io/blob/main/images/movie-recommendation-using-rag/movie_recommendation.png?raw=true)

Some of the movies are even the closest match and just like that, we've crafted an impressive movie recommendation system that can swiftly provide us with the best selection of movies. This is made possible by leveraging embeddings and a Vectordb. 

That being said, here is the [colab](https://colab.research.google.com/drive/1B6I5SEXzuuEVaHcy4IwaJlrMy8wJfPSx?authuser=1#scrollTo=BV1mjqD93Mud) link for the reference.