---
layout: post
title: Creating a Restaurant Recommendation System
description: This post gives an adequate understanding of how to use RAG to create a restaurant recommendation system.
tags: [RAG, LanceDB, Recommendation System]
version: Released
release: 12-28-2024
---

In 2024, I did spend significant amount of time playing with RAG and Vector Databases. Typically, LanceDB was at the core and it was one amazing hell of a ride. As the year is ending, I tried to make a simple restaurant recommendation system but it's going to be slightly different. The idea is to use this [Restaurant Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/swiggy-restuarant-dataset) and to answer the queries like "Is there any cafe nearby which serves Punjabi food?" or "What are the top 5 restaurants in the city which serve Chinese food?". The twist is, we need some form of geospatial data to mould the data points into a form that can be used for queries which are related to the distance.

So if you want to name this stuff, it can be called "Geospatial Recommendation System". Sounds cool.. Ok let's get started.

### How to start?

Well, ok so the first thing is to sort out some relevant features that are required for the recommendation system. So generally, folks who are searching for any recommendation for a restaurant, they are generally looking out for things like restaurant's name (obviously lol), location maybe, and what kind of food they serve, or other details like ratings or maybe how much time it takes for a restaurant to deliver the food or what's the average order prices. So, we need to extract these features from the dataset.

```python
import pandas as pd

restaurant_data = pd.read_csv("data.csv")
restaurant_data = restaurant_data[restaurant_data.columns[1:]]
restaurant_data.dropna(inplace=True)
restaurant_data.drop_duplicates(inplace=True)
restaurant_data.head()
```

![image-showing-the-table-and-the-data](../images/creating-a-restaurant-bot/data-reference.png)

To keep things simple, we’re just going to focus on a few key details for now: the type of food a restaurant serves, how customers rate it on average, and where it's located. By sticking to these basics, we can quickly give people great recommendations without making things too complicated.

```python
data_points_vectors = []

for _, row in restaurant_data.iterrows():
    filter_cols = ['Food type', 'Avg ratings', 'Address']
    data_point = "#".join(f"{col}/{row[col]}" for col in filter_cols)
    data_points_vectors.append(data_point)

# Add the new column to the DataFrame
restaurant_data["query_string"] = data_points_vectors
```

You can see that I've used `#` to separate different sections and `/` for splitting up the key-value pairs. Just a heads up, you can pick different separators and delimiters if you like, but since I'm using FTS (full text search) from LanceDB, a few are reserved for internal representations. If you need to, you can use a backslash as a prefix to support the reserved ones and still use them. This is how a single query string might look

```python
'Food type/Biryani,Chinese,North Indian,South Indian#Avg ratings/4.4#Address/5Th Block'
```

Ok, this looks good! Next, we need to turn our query string into a vector. You can choose any embedding model that fits your needs, but I'll be using the paraphrase-MiniLM-L6-v2 model for now. Basically, all we have to do is encode our query strings into vectors and then load up the payload with the relevant information.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

list_of_payloads = []

for index, row in restaurant_data.iterrows():
    encoded_vector = model.encode(row['query_string'])
    payload = {
        'Area': row['Area'],
        'City': row['City'],
        'Restaurant': row['Restaurant'],
        'Price': row['Price'],
        'Avg_ratings': row['Avg ratings'],
        'Total_ratings': row['Total ratings'],
        'Food_type': row['Food type'],
        'Address': row['Address'],
        'Delivery_time': row['Delivery time'],
        'query_string': row['query_string'],
        'vector': encoded_vector
    }

    list_of_payloads.append(payload)
```

### Setting up the Vector Database

So, we've got our `list_of_payloads` that includes all the relevant data we're going to store in our vector database. Let’s get LanceDB set up here:

```python
import lancedb

# Connect to the LanceDB instance
uri = "data"
db = lancedb.connect(uri)

lancedb_table = db.create_table("restaurant-geocoding-app", data=list_of_payloads)
lancedb_df = lancedb_table.to_pandas()
lancedb_df.head()
```

![image-showing-the-lancedb-table-with-the-data](../images/creating-a-restaurant-bot/lancedb-table-reference.png)


### Setting up the Geospatial Reference

Now that our vector database is ready, the next step is to convert user queries into our specific query format. Essentially, what we will do here is to carefully extract key details from each user query to form a structured dictionary. This structured data will then be reformatted to match the pattern of our query strings. To achieve this, I’ll just use a LLM to decipher the user queries and identify the crucial entities we need.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

query_string = "Hi, I am looking for a casual dining restaurant where Indian or Italian food is served near the HSR Bangalore"

# Helper prompt to extract structured data from ip_prompt
total_prompt = f"""Query String: {query_string}\n\n\
Now from the query string above extract these following entities pinpoints:
1. Food type : Extract the food type 
2. Avg ratings : Extract the average ratings
3. Address : Extract the current exact location, don't consider the fillers like "near" or "nearby".

NOTE : For the Current location, try to understand the pin point location in the query string. Do not give any extra information. If you make the mistakes, bad things
will happen.

Finally return a python dictionary using those points as keys and don't write the markdown of python. If value of a key is not mentioned, then set it as None.
"""

# Make a request to OpenAI's API
completion = client.chat.completions.create(
    model="gpt-4o",  # Use the appropriate model
    store=True,
    messages=[
        {"role": "user", "content": total_prompt}
    ]
)

# Extract the generated text
content = completion.choices[0].message.content
print(content)
```

```python
{
  "Food type": "Indian or Italian",
  "Avg ratings": None,
  "Address": "HSR Bangalore"
}
```

Now, all we need to do is process this output.

```python
import ast

# Convert the string content to a dictionary
try:
    response_dict = ast.literal_eval(content)
except (ValueError, SyntaxError) as e:
    print("Error parsing the response:", e)
    response_dict = {}


filter_cols = ['Food type', 'Avg ratings', 'Address']
query_string_parts = [f"{col}/{response_dict.get(col)}" for col in filter_cols if response_dict.get(col)]

query_string = "#".join(query_string_parts)
print((query_string))
```

Well, now the user query looks like this:

```python
Food type/Indian or Italian#Address/HSR Bangalore
```

Well, this user query is now formatted exactly like our `query_strings`. We can go ahead and search through the vector database to find the top restaurants that best match this query.

### Searching like a pro.

I'll be using the Full Text Search (FTS) feature from LanceDB to run the search. You can read more about what’s happening behind the scenes [here](https://lancedb.github.io/lancedb/fts/#example).

```python
# Create the FTS index and search
lancedb_table.create_fts_index("query_string", replace=True)
results = lancedb_table.search(query_string).to_pandas()
results.head()
```

![image-showing-the-results](../images/creating-a-restaurant-bot/search-results.png)


### Using the Geospatial data

So basically when someone searches for nearby restaurants, maybe they're craving a specific type of cuisine or looking for highly-rated places, we first search up the places that fills there requirements. Now after identifying potential options, we use the [Geospatial API](https://developers.google.com/maps/documentation/geocoding/overview) to pinpoint their exact locations. The Google Maps API is perfect for this—it grabs the latitude and longitude so we know precisely where each restaurant is. With these coordinates, we can then easily figure out which places are closest to the user’s location, by just calculating the distance. 

If you didn't get that, Bear with me, this is going to be super cool.. So first thing we need to do is to setup our Geospatial function which takes a place and return the coordinates:

```python
import requests
import math

def get_google_geocoding(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "OK":
            latitude = result["results"][0]["geometry"]["location"]["lat"]
            longitude = result["results"][0]["geometry"]["location"]["lng"]
            return (latitude, longitude)
        else:
            print(f"Google API: No results found for address: {address}")
            return None
    else:
        print(f"Google API: Request failed for address: {address}")
        return None
```

For the distance calculation, there's this thing called the `Haversine formula`. It uses the coordinates of two points and basically draws an imaginary straight line between them across the earth to measure how far they are from each other. There's a bit of math involved in how this formula works, but we can skip that part for now. Here’s what the formula looks like:

```python
def haversine(coord1, coord2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
```

Well, everything seems solid now, the only thing left is the current location. Here’s what we can do: if a user asks about restaurants near a specific area like "nearby HSR layout," we can easily pull the current location from the preprocessing we did earlier. If not, for now, we can just input the current location manually. Then, we’ll check which restaurants from our vector database match our query and see how far they are based on the current location. 

Well let's see what we get:

```python
def process_top_restaurants(data, current_location, api_key, top_n=5):
    current_coords = get_google_geocoding(current_location, api_key)
    if not current_coords:
        return

    for index, row in data.head(top_n).iterrows():
        complete_address = f"{row['Restaurant']}, {row['City']}"
        restaurant_coords = get_google_geocoding(complete_address, api_key)
        if restaurant_coords:
            distance = haversine(current_coords, restaurant_coords)
            print(f"Restaurant Name: {row['Restaurant']}")
            print(f"Distance: {distance:.2f} km")
            print(f"Area: {row['Area']}")
            print(f"Price: {row['Price']}")
            print(f"Coordinates: {restaurant_coords}")
            print(f"Cuisines Type: {row['Food_type']}")
            print("-" * 40)

# Example usage
api_key = os.getenv('GOOGLE_MAPS_API')
current_location = 'HSR Layout, Bangalore'
process_top_restaurants(results, current_location, api_key, top_n=5)
```

```python
Restaurant Name: Brooks And Bonds Brewery
Distance: 3.36 km
Area: Koramangala
Price: 200.0
Coordinates: (12.9341801, 77.62334249999999)
Cuisines Type: Indian
----------------------------------------
Restaurant Name: Cafe Azzure
Distance: 8.06 km
Area: Ashok Nagar
Price: 1000.0
Coordinates: (12.975012, 77.6076558)
Cuisines Type: American,Italian
----------------------------------------
Restaurant Name: Tottos Pizza
Distance: 7.92 km
Area: Central Bangalore
Price: 500.0
Coordinates: (12.9731935, 77.607012)
Cuisines Type: Continental,Italian
----------------------------------------
Restaurant Name: Holy Doh!
Distance: 4.15 km
Area: Central Bangalore
Price: 600.0
Coordinates: (12.9346188, 77.6139914)
Cuisines Type: Pizzas,Italian
----------------------------------------
Restaurant Name: Bakery By Foodhall
Distance: 7.31 km
Area: Ulsoor
Price: 300.0
Coordinates: (12.9734944, 77.62038629999999)
Cuisines Type: Bakery,Italian,Desserts
----------------------------------------
```

Here we go.. This is cool :)
