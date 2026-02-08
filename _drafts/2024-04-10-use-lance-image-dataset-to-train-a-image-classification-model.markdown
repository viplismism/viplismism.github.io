---
layout: post
title: Use a Lance Image Dataset to train a Image Classification model
description: This post gives a detailed view on how you can use the Lance Image Dataset to train a classification model
tags: [Lance, Dataset]
version: Draft
release: 19-04-2024
---

![image](https://raw.githubusercontent.com/viplismism/viplismism.github.io/15418ff16a807a748797dba2983ec39990ea85d0/images/training-with-lance-data/less-time.png)

In a [previous](https://viplismism.github.io/2024/04/09/convert-any-image-dataset-to-lance) post, I showed you how you can convert any Image Dataset to Lance format for faster retrieval and faster I/O operations. But can we use the same Lance formatted image dataset to train a image classification model? Well here it comes...

### LANCE: A Saga for Efficient Image Datasets
Convolutional Neural Networks (CNNs) have become the go-to architecture for a wide range of image-related tasks, including image classification, object detection, and semantic segmentation. The ability of CNNs to automatically learn relevant features from the input data, combined with their strong performance on complex visual tasks, makes them a natural choice for image classification models.

When working with large-scale image datasets, the management and processing of data can become a significant challenge. Traditional file formats like JPEG and PNG, while widely used, are not optimized for the unique requirements of machine learning workflows. This is where the LANCE (Lightweight Annotated Neural Classification Encoding) format shines, providing a game-changing solution for managing and leveraging image datasets.

The LANCE format offers several key advantages that make it a powerful choice for machine learning applications:

1. Columnar Storage: LANCE stores data in a compressed columnar format, enabling efficient storage, fast data loading, and quick random access to subsets of the data. This is particularly beneficial when working with large-scale image datasets.

2. Multimodal Data Handling: LANCE is designed to handle diverse data types, including images, text, and numerical data, within a unified format. This flexibility is a game-changer in machine learning pipelines, where different modalities of data often need to be processed together.

3. Data Persistence and Privacy: LANCE maintains data on disk, ensuring that the data persists through system failures and doesn't need to be constantly transferred over a network. This also enhances data privacy and security, as the data can be stored and accessed locally without relying on external data sources.

### Integrating LANCE and Convolutional Neural Networks
The structured and efficient data organization of the LANCE format allows for seamless integration with Convolutional Neural Network (CNN) architectures, streamlining the data loading and preprocessing steps. By leveraging the LANCE format, you can overcome the challenges of working with large-scale image datasets and focus more on the core aspects of model development and optimization.

In our previous article, we discussed the process of converting various image datasets, such as CINIC-10 and mini-ImageNet, into the LANCE format. Now, we will explore how we can utilize this LANCE-formatted data to train a CNN-based image classification model.

Before diving into the model training, let's take a closer look at the structure of the LANCE-formatted data we created earlier. Referring to the script from the previous article, we can see how the LANCE dataset is created:

```python
import os
import argparse
import pandas as pd
import pyarrow as pa
import lance
import time
from tqdm import tqdm

def process_images(images_folder, split, schema):

    # Iterate over the categories within each data type
    label_folder = os.path.join(images_folder, split)
    for label in os.listdir(label_folder):
        label_folder = os.path.join(images_folder, split, label)
        
        # Iterate over the images within each label
        for filename in tqdm(os.listdir(label_folder), desc=f"Processing {split} - {label}"):
            # Construct the full path to the image
            image_path = os.path.join(label_folder, filename)

            # Read and convert the image to a binary format
            with open(image_path, 'rb') as f:
                binary_data = f.read()

            image_array = pa.array([binary_data], type=pa.binary())
            filename_array = pa.array([filename], type=pa.string())
            label_array = pa.array([label], type=pa.string())
            split_array = pa.array([split], type=pa.string())

            # Yield RecordBatch for each image
            yield pa.RecordBatch.from_arrays(
                [image_array, filename_array, label_array, split_array],
                schema=schema
            )

# Function to write PyArrow Table to Lance dataset
def write_to_lance(images_folder, dataset_name, schema):
    for split in ['test', 'train', 'val']:
        lance_file_path = os.path.join(images_folder, f"{dataset_name}_{split}.lance")
        
        reader = pa.RecordBatchReader.from_batches(schema, process_images(images_folder, split, schema))
        lance.write_dataset(
            reader,
            lance_file_path,
            schema,
        )

def loading_into_pandas(images_folder, dataset_name):
    data_frames = {}  # Dictionary to store DataFrames for each data type
    
    batch_size = args.batch_size
    for split in ['test', 'train', 'val']:
        uri = os.path.join(images_folder, f"{dataset_name}_{split}.lance")

        ds = lance.dataset(uri)

        # Accumulate data from batches into a list
        data = []
        for batch in tqdm(ds.to_batches(columns=["image", "filename", "label", "split"], batch_size=batch_size), desc=f"Loading {split} batches"):
            tbl = batch.to_pandas()
            data.append(tbl)

        # Concatenate all DataFrames into a single DataFrame
        df = pd.concat(data, ignore_index=True)
        
        # Store the DataFrame in the dictionary
        data_frames[split] = df
        
        print(f"Pandas DataFrame for {split} is ready")
        print("Total Rows: ", df.shape[0])
    
    return data_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image dataset.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--dataset', type=str, help='Path to the image dataset folder')
    
    try:
        args = parser.parse_args()
        dataset_path = args.dataset
        if dataset_path is None:
            raise ValueError("Please provide the path to the image dataset folder using the --dataset argument.")
        
        # Extract dataset name
        dataset_name = os.path.basename(dataset_path)

        start = time.time()
        schema = pa.schema([
            pa.field("image", pa.binary()),
            pa.field("filename", pa.string()),
            pa.field("label", pa.string()),
            pa.field("split", pa.string())
        ])
        write_to_lance(dataset_path, dataset_name, schema)
        data_frames = loading_into_pandas(dataset_path, dataset_name)
        end = time.time()
        print(f"Time(sec): {end - start:.2f}")

    except ValueError as e:
        print(e)
        print("Example:")
        print("python3 convert-any-image-dataset-to-lance.py --batch_size 10 --dataset image_dataset_folder")
        exit(1)

    except FileNotFoundError:
        print("The provided dataset path does not exist.")
        exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
```

The script above creates LANCE-formatted files for the training, testing, and validation sets of the CINIC-10 dataset

```python
# Load the LANCE-formatted datasets
train = data_frames['train']
test = data_frames['test'] 
val = data_frames['val']
```

here the `data_frames` dictionary contains the Pandas DataFrames for the training, testing, and validation sets, where each DataFrame has the following structure:

```markdown
   image                                              filename                    label    split
0  b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\...  n02130308_1836.png           cat     train
1  b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\...  cifar10-train-21103.png      cat     train
2  b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\...  cifar10-train-44957.png      cat     train
3  b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\...  n02129604_14997.png          cat     train
4  b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\...  n02123045_1463.png           cat     train
```

Now that we have the LANCE-formatted data ready, we can use it to train a Convolutional Neural Network (CNN) for image classification. To do this, we'll first need to create Dataset and DataLoader objects from the LANCE-formatted files.

## Load the lance files to create the dataloaders

When working with LANCE-formatted image data, we need to consider that the images are stored in a binary format. To use this data with a Convolutional Neural Network (CNN), we need to convert the binary data back into a format that the CNN can process, such as PIL Image objects. 

Here's how we we'll do that:
1. Retrieve the binary image data: The image data is stored in binary format within the lance files. We need to extract the relevant image data first.
2. Convert to PIL Image: Once we have the binary data, we'll convert it into a PIL Image object. This will give us a readable image format that we can work with.
3. Handle grayscale images: Some of the images might be in grayscale mode. We'll need to convert these to RGB format so they're compatible with a CNN that expects 3-channel color images.
4. Apply transformations: Before feeding the images to the CNN, we may need to apply some transformations, like resizing or normalization. We can do this using the provided transform function.
4. Determine the labels: Each image has a label or class associated with it. We'll look up the class index for the image's label in the provided list of classes.
5. Return the data: Finally, we'll return the transformed image and its corresponding label, which can be used to train the CNN model.

To accomplish this, we'll create a custom Dataset class that can handle the all the heavy lifting and give us this sweet dataset to work with

```python
from torch.utils import data
from PIL import Image
import io

# Define the custom dataset class
class CustomImageDataset(data.Dataset):
    def __init__(self, table, classes, transform=None):
        self.table = table
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_data = self.table["image"][idx].as_py()
        label = self.table["label"][idx].as_py()

        img = Image.open(io.BytesIO(img_data))

        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.classes.index(label)
        return img, label
```

This custom dataset class handles all the necessary steps to prepare our LANCE-formatted data for use with a CNN model. By using this class, you can easily integrate the LANCE data into your PyTorch-based training pipeline. Btw, if you are confused what `table` means here, it is simply converting the `lance` files to a more readable form, have a look in their [docs](https://lancedb.github.io/lance/read_and_write.html) and you will know why. 

But TLDR, it's just iterating over the `lance` files to get the relevant images and labels. Now we are ready to roll for training our model.

### Training a CNN

So we have written our custom dataset class, we will just import it in our main CNN script and utilize it's functionalities, as it's pretty easy now

First we will load our lance files, then we will extract our images from it and then finally the customimagedataset to utilize the binary data of the images to give us this image dataset.. 

That's it, other than that, it's simple CNN network doing the training.. btw, I have used ResNet-34 to make things smoother and better for us, afterall who doesn't want to spice up some accuracy! 

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

import io
import tqdm
import lance
import wandb

from PIL import Image
from tqdm import tqdm

import time
import warnings
warnings.simplefilter('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead")

lr = 1e-3
momentum = 0.9
number_of_epochs = 50
train_dataset_path = "cinic/cinic_train.lance"
test_dataset_path = "cinic/cinic_test.lance"
validation_dataset_path = "cinic/cinic_val.lance"
model_batch_size = 64
batches_to_train = 256

# Define the image classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# transformation function 
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Define the custom dataset class
class CustomImageDataset(data.Dataset):
    def __init__(self, table, classes, transform=None):
        self.table = table
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_data = self.table["image"][idx].as_py()
        label = self.table["label"][idx].as_py()

        img = Image.open(io.BytesIO(img_data))

        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.classes.index(label)
        return img, label

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs, batch_to_train):
    model.train()
    total_start = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batch_start = time.time()

        with tqdm(enumerate(train_loader), total=batch_to_train, desc=f"Epoch {epoch+1}") as pbar_epoch:
            for i, data in pbar_epoch:
                if i >= batch_to_train:
                    break

                optimizer.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0:
                    pbar_epoch.set_postfix({'Loss': loss.item()})
                    pbar_epoch.update(10)

        per_epoch_time = time.time() - total_batch_start
        avg_loss = running_loss / batch_to_train
        print(f'Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Time: {per_epoch_time:.4f} sec')
        wandb.log({"Loss": loss.item()})
        wandb.log({"Epoch Duration": per_epoch_time})

    total_training_time = (time.time() - total_start) / 60
    print(f"Total Training Time: {total_training_time:.4f} mins")


    # Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for data in val_loader:
            images_val, labels_val = data[0].to(device), data[1].to(device)
            outputs_val = model(images_val)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    wandb.log({"Validation Accuracy": val_accuracy})
    print('Finished Training')
    return model

train_ds = lance.dataset(train_dataset_path)
test_ds = lance.dataset(test_dataset_path)
val_ds = lance.dataset(validation_dataset_path)

train_ds_table = train_ds.to_table()
test_ds_table = test_ds.to_table()
val_ds_table = val_ds.to_table()

train_dataset = CustomImageDataset(train_ds_table, classes, transform=transform_train)
test_dataset = CustomImageDataset(test_ds_table, classes, transform=transform_test)
val_dataset = CustomImageDataset(val_ds_table, classes, transform=transform_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=model_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model_batch_size, shuffle=True)

wandb.init(project="cinic")

net = Net(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

trained_model = train_model(train_loader, val_loader, net, criterion, optimizer, device, number_of_epochs, batches_to_train)

PATH = '/cinic/cinic_resnet.pth'
torch.save(trained_model.state_dict(), PATH)

def test_model(test_loader, model, device):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in test_loader:
            images_test, labels_test = data[0].to(device), data[1].to(device)
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.2f}%')

test_model(test_loader, trained_model, device)
```

We've just wrapped up training a Convolutional Neural Network (CNN) with a dataset of lance images with just a single script. 

And if you are Curious about why Lance-backed training outperforms our vanilla approaches? I've got a [report](https://wandb.ai/vipulmaheshwari/cinic/reports/Training-a-ResNet-with-Lance-VS-Vanilla-Dataloader--Vmlldzo3NjgzNzI0?accessToken=3y8hveyqv9i0g34if82pj3vwhpi4dver7qptvd6q9kh7jaxxbq2gmoca140ht1ao) for you, showcasing how Lance-backed training delivers faster epoch durations compared to standard methods while sticking to the same CNN setting.

By the way, you can dive into various deep learning techniques that utilize lance-formatted data on this [repository](https://github.com/lancedb/lance-deeplearning-recipes).

Here is the [colab](https://colab.research.google.com/drive/1EepmyICbOnFTXtjof4_NMyL-sF0M_e1t?usp=sharing) for your reference. Adios Amigos