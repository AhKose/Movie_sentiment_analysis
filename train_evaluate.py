import torch
import torch.optim as optim
import torch.nn as nn
from model_definition import SimpleRNN, convert_to_tensors
from data_preprocessing import load_and_process_data, traverse_and_vectorize, vectorize_glove, parse_tree, print_random_samples  
from model_definition import SimpleRNN, convert_to_tensors


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load GloVe model
    glove_path = 'glove.6B.300d.txt'  # Adjust path as necessary
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

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

    train_x, train_y, train_is_root = convert_to_tensors(train_data)
    val_x, val_y, val_is_root = convert_to_tensors(val_data)
    test_x, test_y, test_is_root = convert_to_tensors(test_data)

    # Placeholder for model instantiation and example usage
    input_size = 300  # Size of GloVe vectors
    hidden_size = 50  # RNN hidden layer size
    num_classes = 5   # Assuming 5 sentiment classes for classification

    model = SimpleRNN(input_size, hidden_size, num_classes)
    print("Model structure: ", model)


    model = SimpleRNN(input_size, hidden_size, num_classes)
    print("Model structure: ", model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model and evaluate on the validation set
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss, total_correct_all, total_correct_root, total_all, total_root = 0, 0, 0, 0, 0
        for i in range(len(train_x)):
            # Forward pass
            outputs = model(train_x[i].unsqueeze(0), train_is_root[i].unsqueeze(0))
            loss = loss_function(outputs, train_y[i].unsqueeze(0))
            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            total_correct_all += (predicted == train_y[i]).item()
            if train_is_root[i]:
                total_correct_root += (predicted == train_y[i]).item()

            total_all += 1
            if train_is_root[i]:
                total_root += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = total_loss / len(train_x)
        train_acc_all = (total_correct_all / total_all) * 100
        train_acc_root = (total_correct_root / total_root) * 100 if total_root > 0 else 0

        # Validation
        model.eval()
        val_loss, val_correct_all, val_correct_root, val_total_all, val_total_root = 0, 0, 0, 0, 0
        with torch.no_grad():
            for i in range(len(val_x)):
                outputs = model(val_x[i].unsqueeze(0), val_is_root[i].unsqueeze(0))
                loss = loss_function(outputs, val_y[i].unsqueeze(0))
                val_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(outputs, dim=1)
                val_correct_all += (predicted == val_y[i]).item()
                if val_is_root[i]:
                    val_correct_root += (predicted == val_y[i]).item()

                val_total_all += 1
                if val_is_root[i]:
                    val_total_root += 1

        val_loss /= len(val_x)
        val_acc_all = (val_correct_all / val_total_all) * 100
        val_acc_root = (val_correct_root / val_total_root) * 100 if val_total_root > 0 else 0

        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc (All): {train_acc_all:.2f}%, Train Acc (Root): {train_acc_root:.2f}%')
        print(f'           Validation Loss: {val_loss:.4f}, Validation Acc (All): {val_acc_all:.2f}%, Validation Acc (Root): {val_acc_root:.2f}%')
