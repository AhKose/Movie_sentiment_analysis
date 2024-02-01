import random
import nltk
from gensim.models import KeyedVectors
from nltk.tree import Tree
import numpy as np

# Function to parse a line into a tree structure
def parse_tree(line):
    return Tree.fromstring(line)

# Function to vectorize a token using GloVe
def vectorize_glove(token, glove_model, vector_size=300):
    return glove_model[token] if token in glove_model else np.zeros(vector_size)

# Function to traverse a tree and vectorize its nodes
def traverse_and_vectorize(tree, glove_model, is_root=True):
    results = []

    if is_root:  # Root node
        sentence = " ".join(tree.leaves())
        vector = np.mean([vectorize_glove(word) for word in tree.leaves()], axis=0)
        root_sentiment = int(tree.label())
        results.append((sentence, vector, root_sentiment, True))

    if isinstance(tree[0], str):  # Leaf node
        word = tree[0]
        vector = vectorize_glove(word)
        sentiment = int(tree.label())
        results.append((word, vector, sentiment, False))
    else:
        for child in tree:
            results.extend(traverse_and_vectorize(child, is_root=False))

    return results

# Function to load and process data
def load_and_process_data(filename, glove_model):
    processed_data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                tree = parse_tree(line)
                processed_data.extend(traverse_and_vectorize(tree))
            except ValueError:
                print(f"Skipping problematic line: {line.strip()}")
                continue
    return processed_data

# Function to print random samples with their labels for root and non-root
def print_random_samples(data, num_samples=5):
    print(f"\nDisplaying {num_samples} random non-root samples and {num_samples} root samples from the data:")

    non_root_samples = [sample for sample in data if not sample[3]]
    root_samples = [sample for sample in data if sample[3]]

    print("\nNon-Root Samples:")
    for _ in range(num_samples):
        sample = random.choice(non_root_samples)
        print(f"Word: {sample[0]}, Is Root: {sample[3]}, Sentiment: {sample[2]}")

    print("\nRoot Samples:")
    if root_samples:
        for _ in range(num_samples):
            sample = random.choice(root_samples)
            print(f"Word: {sample[0]}, Is Root: {sample[3]}, Sentiment: {sample[2]}")
    else:
        print("No root samples found.")
