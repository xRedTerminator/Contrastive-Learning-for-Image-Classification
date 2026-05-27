# Contrastive Learning for Image Classification

This project implements and evaluates self-supervised contrastive learning methods for image representation learning on CIFAR-10. The goal is to train visual encoders without using class labels during pretraining, then test whether the learned embeddings are useful for downstream image classification through linear probing.

The project compares two contrastive objectives, Triplet Loss and SimCLR-style NT-Xent Loss, across convolutional and transformer-based image encoders. It also visualizes the learned feature space using UMAP to compare representation quality against raw pixel-space baselines.

## Overview

Modern image classifiers typically require labeled data, but self-supervised learning can learn useful representations from unlabeled images by constructing training signals from data augmentation. Contrastive learning does this by encouraging two augmented views of the same image to have similar embeddings while pushing embeddings from different images apart.

This project explores that idea through the following pipeline:

1. Load CIFAR-10 image data
2. Generate positive image pairs using random augmentations
3. Train encoders with contrastive objectives
4. Extract learned image embeddings
5. Evaluate embeddings with a linear classifier
6. Compare learned representations against raw pixel baselines
7. Visualize feature structure using dimensionality reduction

## Goals

The main goals of this project are to:

- Implement contrastive representation learning from scratch in PyTorch
- Compare Triplet Loss and NT-Xent Loss for self-supervised image learning
- Train both ResNet-style and Vision Transformer-style encoders
- Evaluate learned embeddings using linear probing
- Visualize whether contrastive pretraining creates more separable class structure
- Understand how self-supervised features compare to raw pixel-space representations

## Dataset

The project uses the CIFAR-10 dataset loaded through Hugging Face Datasets.

CIFAR-10 contains 60,000 color images across 10 object classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The training pipeline does not use class labels during contrastive pretraining. Labels are only used afterward for downstream linear evaluation.

## Data Augmentation

Contrastive learning depends heavily on augmentation. For each training image, the dataset returns two independently augmented views of the same image. These two views form a positive pair.

The augmentation pipeline includes:

- Random resized crop
- Random horizontal flip
- RandAugment
- Tensor conversion
- Image normalization

At test time, deterministic preprocessing is used:

- Resize
- Center crop
- Tensor conversion
- Image normalization

This setup encourages the model to learn features that are invariant to visual transformations while still preserving semantic information.

## Contrastive Learning Objectives

### Triplet Loss

The Triplet Loss objective treats two augmented views of the same image as positive pairs. Other images in the batch act as negatives.

The model is trained so that:

- embeddings from the same image are close together
- embeddings from different images are separated by a margin

This provides a simple contrastive baseline for learning image representations.

### NT-Xent Loss

The NT-Xent loss is the standard SimCLR-style contrastive objective.

For each batch, two augmented views are created for every image. The model computes pairwise similarities between all augmented examples and learns to identify the matching augmented view among many negatives.

This objective uses:

- cosine similarity
- temperature scaling
- in-batch negatives
- cross-entropy classification over positive pairs

Compared with Triplet Loss, NT-Xent typically provides a stronger learning signal because every example in the batch contributes to the contrastive classification task.

## Model Architectures

### ResNet Encoder

The project implements a custom ResNet-style convolutional encoder using residual blocks. The model extracts image features through convolutional layers and average pooling, then maps them into a projection space for contrastive learning.

The ResNet encoder is used for both:

- Triplet Loss training
- SimCLR / NT-Xent training

During evaluation, embeddings are extracted before the projection head and used as feature vectors for linear probing.

### Vision Transformer Encoder

The project also implements a custom Vision Transformer model.

The ViT pipeline includes:

- image patching
- patch embeddings
- sinusoidal 2D positional embeddings
- transformer encoder blocks
- representation extraction for contrastive learning

This allows comparison between convolutional inductive bias and transformer-based representation learning under the same contrastive framework.

## Training Pipeline

The training loop follows a standard PyTorch workflow:

1. Sample a batch of images
2. Generate two augmented views per image
3. Pass both views through the encoder
4. Compute contrastive loss between the two embedding sets
5. Backpropagate the loss
6. Update model parameters with AdamW
7. Apply cosine learning rate scheduling

The project trains models for a fixed number of epochs and compares the learned representations after pretraining.

## Evaluation Pipeline

After contrastive pretraining, the encoder is frozen and used to extract image embeddings.

The evaluation process is:

1. Extract embeddings from the training split
2. Train a logistic regression classifier on a subset of embeddings
3. Extract embeddings from the test split
4. Evaluate classification accuracy on CIFAR-10 test labels

This is known as linear probing. It measures whether the learned embedding space is linearly separable with respect to downstream class labels.

The project also trains a logistic regression classifier directly on raw pixels to provide a baseline comparison.

## Visualization

The project uses UMAP to visualize learned feature embeddings.

Two visualizations are compared:

1. Contrastive feature space
2. Raw pixel space

The goal is to inspect whether contrastive learning produces more clustered and semantically meaningful representations than raw image pixels.

If the contrastive encoder learns useful features, examples from the same class should form more coherent clusters in the reduced-dimensionality plot.

## Key Components

The project includes:

- Custom Triplet Loss implementation
- Custom NT-Xent / SimCLR loss implementation
- Custom ResNet encoder
- Custom Vision Transformer encoder
- CIFAR-10 loading through Hugging Face Datasets
- SimCLR-style data augmentation pipeline
- PyTorch training loop
- Linear probing with logistic regression
- Pixel-space baseline classifier
- UMAP feature visualization

## Technologies Used

- Python
- PyTorch
- Torchvision
- Hugging Face Datasets
- Transformers learning rate scheduler
- Scikit-learn
- UMAP
- NumPy
- Pandas
- Matplotlib
- Einops

## Project Structure

```text
.
├── contrastive_learning.ipynb   # Main notebook containing training, evaluation, and visualization
├── README.md                    # Project documentation
└── .gitignore                   # Files ignored by Git