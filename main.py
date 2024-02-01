from data_processing import download_nltk_resources, load_glove_model, load_and_process_data, print_random_samples, convert_to_tensors, traverse_and_vectorize, vectorize_glove, parse_tree,
from model_definition import SimpleRNN
from train_eval import train, evaluate
import torch
import torch.optim as optim
import torch.nn as nn


def main():
    # Download resources and load GloVe model
    download_nltk_resources()
    glove_path = 'glove.6B.300d.txt'
    glove_model = load_glove_model(glove_path)

    # Paths for your datasets
    train_path = '/path/to/train.txt'
    val_path = '/path/to/dev.txt'
    test_path = '/path/to/test.txt'

    # Load and process data
    train_data = load_and_process_data(train_path, glove_model)
    val_data = load_and_process_data(val_path, glove_model)
    test_data = load_and_process_data(test_path, glove_model)

    # Print random samples from the training data
    print_random_samples(train_data)
  
    # Convert data to tensors
    train_x, train_y, train_is_root = convert_to_tensors(train_data)
    val_x, val_y, val_is_root = convert_to_tensors(val_data)
    test_x, test_y, test_is_root = convert_to_tensors(test_data)
    
    model = SimpleRNN(input_size=300, hidden_size=50, num_classes=5)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and evaluation
    for epoch in range(10):  # Example: 10 epochs
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_function, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

if __name__ == "__main__":
    main()
