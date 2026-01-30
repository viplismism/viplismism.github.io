---
layout: post
title: Effortlessly Loading and Processing Images with Lance
description: How you can use the lance format to work with big sized data
tags: [LanceDB, Data Engineering, Computer Vision]
version: Published
release: 29-03-2024
---


Working with large image datasets in machine learning can be challenging, often requiring significant computational resources and efficient data-handling techniques. While widely used for image storage, traditional file formats like JPEG or PNG are not optimized for efficient data loading and processing in Machine learning workflows. This is where the Lance format shines, offering a modern, columnar data storage solution designed specifically for machine learning applications.

![meme_for_ml_workloads](https://github.com/vipul-maheshwari/vipul-maheshwari.github.io/blob/main/images/loading_and_processing_image_with_lance/image.png?raw=true)

The Lance format stores data in a compressed columnar format, enabling efficient storage, fast data loading, and fast random access to data subsets. Additionally, the Lance format is maintained on disk, which provides a couple of advantages: It will persist through a system failure and doesn’t rely on keeping everything in memory, which can run out. This also lends itself to enhanced data privacy and security, as the data doesn’t need to be transferred over a network.

One of the other key advantages of the Lance format is its ability to store diverse data types, such as images, text, and numerical data, in a unified format. Imagine having a data lake where each kind of data can be stored seamlessly without separating underlying data types. This flexibility is particularly valuable in machine learning pipelines, where different data types often need to be processed together. This unparalleled flexibility is a game-changer in machine learning pipelines, where different modalities of data often need to be processed together for tasks like multimodal learning, audio-visual analysis, or natural language processing with visual inputs.

With Lance, you can effortlessly consider all kinds of data, from images to videos and audio files to text data and numerical values, all within the same columnar storage format. This means you can have a single, streamlined data pipeline that can handle any combination of data types without the need for complex data transformations or conversions. Lance easily handles it without worrying about compatibility issues or dealing with separate storage formats for different data types. And the best part? You can store and retrieve all these diverse data types within the same column.

In contrast, while efficient for tabular data, traditional formats like Parquet may need to handle diverse data types better. By converting all data into a single, unified format using Lance, you can retrieve and process any type of data without dealing with multiple formats or complex data structures.
In this article, I'll walk through a Python code example that demonstrates how to convert a dataset of GTA5 images into the Lance format and subsequently load them into a Pandas DataFrame for further processing.

```python
import os
import pandas as pd
import pyarrow as pa
import lance
import time
from tqdm import tqdm
```

We start by importing the necessary libraries, including os for directory handling, pandas for data manipulation, pyarrow for working with Arrow data formats, lance for interacting with the Lance format, and tqdm for displaying progress bars.

```python
def process_images():
    current_dir = os.getcwd()
    images_folder = os.path.join(current_dir, "./image")

    # Define schema for RecordBatch
    schema = pa.schema([('image', pa.binary())])

    image_files = [filename for filename in os.listdir(images_folder)
                if filename.endswith((".png", ".jpg", ".jpeg"))]

    # Iterate over all images in the folder with tqdm
    for filename in tqdm(image_files, desc="Processing Images"):
        # Construct the full path to the image
        image_path = os.path.join(images_folder, filename)

        # Read and convert the image to a binary format
        with open(image_path, 'rb') as f:
            binary_data = f.read()

        image_array = pa.array([binary_data], type=pa.binary())

        # Yield RecordBatch for each image
        yield pa.RecordBatch.from_arrays([image_array], schema=schema)
```

The process_images function is responsible for iterating over all image files in a specified directory and converting them into PyArrow RecordBatch objects. It first defines the schema for the RecordBatch, specifying that each batch will contain a single binary column named 'image'. 

It then iterates over all image files in the directory, reads each image's binary data, and yields a RecordBatch containing that image's binary data.

```python
def write_to_lance():
    schema = pa.schema([
        pa.field("image", pa.binary())
    ])

    reader = pa.RecordBatchReader.from_batches(schema, process_images())
    lance.write_dataset(
        reader,
        "image_dataset.lance",
        schema,
    )
```

The write_to_lance function creates a RecordBatchReader from the process_images generator and writes the resulting data to a Lance dataset named "image_dataset.lance". This step converts the image data into the efficient, columnar Lance format, optimizing it for fast data loading and random access.

```python
def loading_into_pandas():
    uri = "image_dataset.lance"
    ds = lance.dataset(uri)

    # Accumulate data from batches into a list
    data = []
    for batch in ds.to_batches(columns=["image"], batch_size=10):
        tbl = batch.to_pandas()
        data.append(tbl)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(data, ignore_index=True)
    print("Pandas DataFrame is ready")
    print("Total Rows: ", df.shape[0])
```

The loading_into_pandas function demonstrates how to load the image data from the Lance dataset into a Pandas DataFrame. It first creates a Lance dataset object from the "image_dataset.lance" file. Then, it iterates over batches of data, converting each batch into a Pandas DataFrame and appending it to a list. Finally, it concatenates all the DataFrames in the list into a single DataFrame, making the image data accessible for further processing or analysis.

```python
if __name__ == "__main__":
    start = time.time()
    write_to_lance()
    loading_into_pandas()
    end = time.time()
    print(f"Time(sec): {end - start}")
```

The central part of the script calls the write_to_lance and loading_into_pandas functions, measuring the total execution time for the entire process.
By leveraging the Lance format, this code demonstrates how to efficiently store and load large image datasets for machine learning applications. The columnar storage and compression techniques Lance uses result in reduced storage requirements and faster data loading times, making it an ideal choice for working with large-scale image data.

Moreover, the random access capabilities of Lance allow for selective loading of specific data subsets, enabling efficient data augmentation techniques and custom data loading strategies tailored to your machine learning workflow.

TLDR: Lance format provides a powerful and efficient solution for handling multimodal data in machine learning pipelines, streamlining data storage, loading, and processing tasks. By adopting Lance, we can improve our machine learning projects' overall performance and resource efficiency while also benefiting from the ability to store diverse data types in a unified format and maintain data locality and privacy. Here is the whole script for your reference.

```python
import os
import pandas as pd
import pyarrow as pa
import lance
import time
from tqdm import tqdm

def process_images():
    current_dir = os.getcwd()
    images_folder = os.path.join(current_dir, "./image")

    # Define schema for RecordBatch
    schema = pa.schema([('image', pa.binary())])

    image_files = [filename for filename in os.listdir(images_folder)
                if filename.endswith((".png", ".jpg", ".jpeg"))]

    # Iterate over all images in the folder with tqdm
    for filename in tqdm(image_files, desc="Processing Images"):
        # Construct the full path to the image
        image_path = os.path.join(images_folder, filename)

        # Read and convert the image to a binary format
        with open(image_path, 'rb') as f:
            binary_data = f.read()

        image_array = pa.array([binary_data], type=pa.binary())

        # Yield RecordBatch for each image
        yield pa.RecordBatch.from_arrays([image_array], schema=schema)

def write_to_lance():
    
    schema = pa.schema([
        pa.field("image", pa.binary())
    ])

    reader = pa.RecordBatchReader.from_batches(schema, process_images())
    lance.write_dataset(
        reader,
        "image_dataset.lance",
        schema,
    )

def loading_into_pandas():

    uri = "image_dataset.lance"
    ds = lance.dataset(uri)

    # Accumulate data from batches into a list
    data = []
    for batch in ds.to_batches(columns=["image"], batch_size=10):
        tbl = batch.to_pandas()
        data.append(tbl)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(data, ignore_index=True)
    print("Pandas DataFrame is ready")
    print("Total Rows: ", df.shape[0])


if __name__ == "__main__":
    start = time.time()
    write_to_lance()
    loading_into_pandas()
    end = time.time()
    print(f"Time(sec): {end - start}")

```

Imagine using Lance-formatted image data to make machine learning and deep learning projects faster. Something big is coming up, stay tuned.