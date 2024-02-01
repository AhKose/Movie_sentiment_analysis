# Movie_sentiment_analysis
---
## Overview

This repository contains the implementation of an RNN-based Natural Language Processing model for Multi-Class Sentiment Classification. GloVe is used to convert words to vectors. It includes the process of loading and processing data, vectorizing text using GloVe embeddings, building and training an RNN model, and evaluating its performance on sentiment classification tasks.

Note: The project utilizes tree structure data from Stanford Sentiment Treebank v2, requiring adjustments for compatibility with different data types.

## Features

Data Preprocessing: Load and preprocess sentiment treebank data, preparing it for model training.
GloVe Embeddings: Utilize GloVe for effective word vectorization, enhancing model understanding of text semantics.
RNN Model: Implement a Recurrent Neural Network model to capture sequential data dependencies, crucial for sentiment analysis.
Training and Evaluation: Train the model on sentiment data and evaluate its performance across multiple classes, aiming for high accuracy in sentiment classification.

## Project Structure

This project includes the following files and directories:

README.md: This file.
data_preprocessing.py: Script for data processing.
model_definition.py: Script for model definition.
train_evaluate.py: Script for training and evaluating the model.
main.py: Main script for executing the project.
data/: Directory for storing datasets. Ensure to replace placeholder paths with actual paths to your datasets.

## Usage

Data Source
This project utilizes the Stanford Sentiment Treebank v2, available on Kaggle: [Stanford Sentiment Treebank v2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2). Before running the project, please download this dataset and ensure the paths in the main.py file are correctly set to where you've stored the data.

Download the GloVe embeddings:
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
Ensure the GloVe .txt files are placed in the root directory of the project.

To define and train the model, and then evaluate its performance, execute:
```
python main.py
```

## Dependencies

- NLTK
- Gensim
- NumPy
- PyTorch

## License
Distributed under the Apache License. See LICENSE for more information.

## Data Samples
The images below showcase random data samples from the Stanford Sentiment Treebank v2, illustrating the sentiment scale originally used in the dataset. For clarity: while the original dataset showcases sentiment on a scale of 1-25, our project utilizes a updated version of the data, where sentiments are classified on a 0-4 scale for training and evaluation purposes.

![image](https://github.com/ahk19/Movie_sentiment_analysis/assets/48156018/fc763bd4-62a4-4d6e-ba0d-790e4737cd3e)
![image](https://github.com/ahk19/Movie_sentiment_analysis/assets/48156018/775fa220-91b6-4bc4-89eb-5ba2be186dd1)
