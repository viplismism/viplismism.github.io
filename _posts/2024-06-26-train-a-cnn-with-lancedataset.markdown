---
layout: post
title: Train a CNN classification model with Lance Dataset
description: This post shows how you can train a CNN model with Lance Dataset
tags: [LanceDB, Deep Learning, Computer Vision]
version: Published
release: 26-06-2024
---


In this [previous](https://vipul-maheshwari.github.io/2024/04/09/convert-any-image-dataset-to-lance) post, I showed you how you can convert any Image Dataset to Lance format for faster retrieval and faster I/O operations. But can we use the same Lance formatted image dataset to train an image classification model? Well here it comes...

![front-image](/images/train-a-cnn-with-lance-dataset/training_a_cnn.png)

### Lance Format: Saga for efficient image datasets

CNNs are widely used for the image related tasks in AI world. They're great at figuring out what's in a picture, spotting objects, and even breaking down images into meaningful parts. What makes them so useful is how they can learn important visual clues on their own, without needing humans intervention.

But when we're dealing with massive image collections, just handling all that data can be a real headache. That's where Lance file format comes in - it provides a clever new way to package up image data perfectly in the deep learning ecosystem for all our needs. The Lance format offers several key advantages that make it a powerful choice for machine learning applications, some of them are:

1. Lance uses a compressed columnar format, offering efficient storage, fast data loading, and quick random access, making it ideal for large-scale image datasets.
2. It supports diverse data types, including images, and text facilitating the processing of different modalities in machine learning pipelines.
3. Lance stores data on disk, ensuring persistence through system failures and enhancing privacy and security by allowing local storage and access.
4. It provides high-performance random access, up to 100 times faster than Parquet.
5. Lance enables vector search, finding nearest neighbors in under 1 millisecond, and integrates OLAP queries with vector search.
6. It features zero-copy, automatic versioning, which manages data versions automatically and reduces redundancy.

### Integrating Lance and Convolutional Neural Networks

If you've been working with Convolutional Neural Networks (CNNs) for image classification, you know that data loading and preprocessing can be a real headache. But what if I told you there's a way to make this process smoother and faster? Enter the Lance format.

In my previous post, I walked through the process of converting popular image datasets like [cinic-10](https://www.kaggle.com/datasets/vipulmaheshwarii/cinic-10-lance-dataset?ref=blog.lancedb.com) and [mini-imagenet](https://www.kaggle.com/datasets/vipulmaheshwarii/mini-imagenet-lance-dataset) to the Lance format. If you haven't read that yet, I highly recommend you do so before continuing here. It'll give you the foundation you need to fully appreciate what we're about to dive into.

Now, let's take the next step: using Lance-formatted data to train a CNN for image classification. We'll use the cinic-10 dataset as our example, but the principles apply to other datasets as well.

Before we jump in, it's important to understand how the Lance + PyTorch approach differs from the standard PyTorch method. Traditionally, PyTorch users rely on Torchvision's ImageFolder to handle image and label loading. The Lance approach, however, requires us to create a custom dataset class. This class is designed to load binary-format images and their corresponding labels directly from the Lance dataset for creating the dataloaders.

You might be wondering, "Is it worth the effort to switch?" The answer is a resounding yes, especially if you're dealing with large datasets or need faster training times. Lance's secret weapon is its lightning-fast random access capability. This means that Lance dataloaders can feed data to your CNN much quicker than standard PyTorch dataloaders, potentially shaving hours off your training time.

In the following sections, we'll dive into the details of implementing the Lance + PyTorch approach. By the end, you'll have a powerful new tool in your deep learning toolkit that can significantly streamline your image classification workflows and reduce your model training time.

### Load the Lance files to create the dataloaders

Lance-formatted image data is stored in binary format, which isn't directly usable by Convolutional Neural Networks (CNNs). We need to convert this data into a format CNNs can process, such as PIL Image objects. Here's the process we'll follow:

1. Retrieve the binary image data: Extract the relevant image data from the Lance files.
2. Convert to PIL Image: Transform the binary data into a PIL Image object, creating a readable image format.
3. Handle grayscale images: Convert any grayscale images to RGB format for compatibility with CNNs that expect 3-channel color images.
4. Apply transformations: Use the provided transform function to apply necessary transformations like resizing or normalization.
5. Determine the labels: Look up the class index for each image's label in the provided list of classes.
6. Return the data: Provide the transformed image and its corresponding label for CNN training.

To streamline this process, we'll create a custom dataset class. This class will handle all these steps efficiently, preparing the Lance-formatted data for use with a CNN.

This custom dataset class manages all the necessary steps to prepare our Lance-formatted data for use with a CNN model. It essentially iterates over the dataset to retrieve the relevant images and labels. By using this class, we can easily integrate the Lance data into your PyTorch-based training pipeline.

```python
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
```

```python
# Define the custom dataset class
class CustomImageDataset(data.Dataset):
    def __init__(self, classes, lance_dataset, transform=None):
        self.classes = classes
        self.ds = lance.dataset(lance_dataset)
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        raw_data = self.ds.take([idx], columns=['image', 'label']).to_pydict()
        img_data, label = raw_data['image'][0], raw_data['label'][0]

        img = Image.open(io.BytesIO(img_data))

        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.classes.index(label)
        return img, label
```

Now that we have our custom Dataset class set up, we're ready to proceed with training our model using the Lance dataset.

### Using Lance dataset with CNNs: Putting It All Together

Now that we've created our custom dataset class, integrating Lance dataset into our CNN training process becomes straightforward. Here's how it all comes together:

1. Import the custom dataset class into our CNN script.
2. Load the lance dataset and create lance dataloaders.
3. Use the lance dataloaders instead of the standard dataloaders to train our model.

From this point, the process follows a standard CNN training workflow. For our example, I've chosen to use ResNet-34 as our CNN architecture to enhance accuracy.

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
```

With this setup, we can train a CNN on our Lance dataset using just a single script. 

One key advantage of using Lance-backed training is its performance. Compared to traditional methods, Lance-formatted data offers significant improvements in training speed. Here is the result when I compared the training time for 3 epochs with Lance vs Vanilla dataloaders 

![epoch_duration](/images/train-a-cnn-with-lance-dataset/epoch_duration.png)

This shows an extensive improvement in training time for the Lance dataloaders as compared to the Vanilla ones. 

![shocking_cat](/images/train-a-cnn-with-lance-dataset/shocking_cat.png)

Here is the complete [notebook](https://github.com/lancedb/lance-deeplearning-recipes/blob/main/community-examples/cnn-model-with-lance-dataset.ipynb) for the reference. For those wanting to explore further, there's a [repository](https://github.com/lancedb/lance-deeplearning-recipes) showcasing various deep learning techniques that utilize Lance-formatted data. This resource can be valuable for expanding your understanding and application of Lance file format in different machine learning contexts. 