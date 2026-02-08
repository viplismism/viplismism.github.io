---
layout: post
title: Python Package to convert image datasets to lance type
description: This post gives a detailed overview of how we can convert any image dataset to lance using a python package
tags: [LanceDB, Python, Open Source]
version: Released
release: 11-12-2024
---

![cli-lancify-title-image](/images/cli-for-lance-converter/lancify-title-image.png)

A few months ago, I wrote two articles on how the Lance format can supercharge your machine learning workflows. In the [first](https://viplismism.github.io/2024/03/29/effortlessly-loading-and-processing-images-with-lance-a-code-walkthrough), I showed how Lance’s columnar storage can make handling large image datasets much more efficient for the ML workflows. Then, I followed up with a [guide](https://viplismism.github.io/2024/04/09/convert-any-image-dataset-to-lance) on converting datasets like [cinic](https://www.kaggle.com/datasets/vipulmaheshwarii/cinic-10-lance-dataset?ref=blog.lancedb.com) and [mini-imagenet](https://www.kaggle.com/datasets/vipulmaheshwarii/mini-imagenet-lance-dataset?ref=blog.lancedb.com) into Lance format using a custom Python script in [Google Colab](https://colab.research.google.com/drive/12RjdHmp6m9_Lx7YMRiat4_fYWZ2g63gx?usp=sharing&ref=blog.lancedb.com). While that worked well, it was a bit manual.

Well, Some of my friends are lazy as `fuxcks` but excited enough to run the colab and use the Lance formatted datatype for some of their experiments. Being a good friend, I’m excited to share a much easier solution: the `lancify` Python package.

I mean it's literally just running a one command, and boom — your image datasets are in lance format, ready to go. And, just between us, it makes my life a lot easier too.

### Installing the Package

Before diving into the conversion process, let’s start by installing the `lancify` package. You can easily install it via pip:

```bash
pip install lancify
```

### Converting Your Image Dataset to Lance

Once you’ve installed the package, converting any image dataset to the Lance format is as simple as running the following Python code. The `lancify` package abstracts away the complexity of running the colab notebooks manually;

```python
from lancify.converter import convert_dataset

# Define the path to your image dataset
image_dataset_path = 'cards-classification'
resize_dimensions = (256, 256)  # Example resize dimensions
splits = ["train", "test", "valid"]

convert_dataset(
    dataset_path=image_dataset_path,
    batch_size=10,  # You can adjust the batch size as needed
    resize=resize_dimensions,  # Pass resize dimensions if you want to resize images
    splits=splits
)
```

For this demonstration, I have used this [dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification), which provides flexibility in terms of image resizing and dataset splits. The image resizing is optional; by default, the images are processed with their original dimensions. However, if needed, you can specify a target size, such as 256x256, by passing the desired dimensions. If you prefer to keep the original size, simply pass None for the resize parameter. Regarding dataset splits, if the dataset includes predefined divisions like training, testing, and validation sets, you can pass a list specifying the relevant splits.

For [datasets](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) that do not have predefined splits, the images are organized by classification labels. In such cases, you only need to provide the dataset path, and a single lance file will be generated, containing all the images with their corresponding labels. This makes sure that the various kinds of image datasets are handled properly whether they include splits or not.

```python
from lancify.converter import convert_dataset

image_dataset_path = 'weather-classification-data'

convert_dataset(
    dataset_path=image_dataset_path,
    batch_size=10,  # You can adjust the batch size as needed
)
```

The `convert_dataset` function automatically handles the following:
1. **Reading the image data** – It reads image files and their metadata (filename, category, data split).
2. **Converting to Lance** – The images are converted into the Lance format with proper schema.
3. **Saving the Lance files** – Lance files are saved for each dataset split (train, test, validation) if there are splits in the dataset, if not then a single lance file is saved with the combined data with an adequate schema to segregate the data with the respective labels.

This method is far more concise than manually iterating over directories, creating schemas, and writing to lance files as we did in the previous version using raw colab.

### CLI SDK

In addition to using the `lancify` package programmatically through the imported function, you can also leverage the CLI SDK to convert your image datasets. The SDK offers a CLI for the `lancify`.  

To use the CLI, all you need to do is install the package with `pip install lancify` and then run the `lancify` command in your terminal and follow the args.

![image-cli](/images/cli-for-lance-converter/cli-lancify.png)

### What's Happened Behind the Scenes?

To give you a better understanding, here's a brief overview of what happens when you use `lancify`:

- **Image Data**: The package reads images from your dataset directory and converts them into a binary format.
- **Metadata Extraction**: Metadata such as the image’s filename, category (label), and data split (train/test/validation) are automatically extracted.
- **PyArrow RecordBatch**: The image data and metadata are packaged into a PyArrow `RecordBatch` for efficient columnar storage.
- **Lance Dataset Creation**: These `RecordBatch` objects are then written to Lance datasets, which are optimized for performance and storage.

This process mirrors the manual steps we previously took but in a much more user-friendly manner, significantly reducing the boilerplate code that was necessary before when you have to manually handle the [colab](https://colab.research.google.com/drive/12RjdHmp6m9_Lx7YMRiat4_fYWZ2g63gx?usp=sharing#scrollTo=93qlCg6TpcW-)

### Loading Your Dataset into Pandas

Once your image dataset has been converted into the lance format, you can seamlessly load it into pandas data frames for doing all kind of stuff. Here’s how to do it for the `card-classification` training lance file..

```python
import lance
import pandas as pd

# Load Lance dataset
ds = lance.dataset('cards-classification/cards-classification_train.lance')
table = ds.to_table()

# Convert Lance table to Pandas dataframe
df = table.to_pandas()
print(df.head())
```

![lance-training](/images/cli-for-lance-converter/lance-training.png)

This is a simple and efficient way to convert your image datasets to the lance format using the `lancify` package, and it integrates smoothly into your deep learning projects.

Switching to the Lance format makes your data pipelines faster and more efficient, especially when dealing with large image datasets. All you need to do is install the package and run the conversion script on your datasets—no hassle.

![meme-maker](/images/cli-for-lance-converter/child-meme-lancify.png)

Just this small change can really speed up your machine learning workflows—data loading and processing become much quicker, which means your models train faster.  If you need a reference, this is a [quickie](https://viplismism.github.io/2024/06/26/train-a-cnn-with-lancedataset) on how to use the lance formatted image datasets for training you deep learning models. And if you're looking for more ideas, there are plenty of other [deep learning recipes](https://github.com/lancedb/lance-deeplearning-recipes) built on lance.

Trust me, it’s worth it! 🤗