import torch
import torch.nn as nn
from data_preprocessing import load_and_process_data, vectorize_glove  

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + 1, num_classes)  # +1 for is_root flag

    def forward(self, x, is_root):
        x, hidden = self.rnn(x)
        x = x[:, -1, :]  # Assuming the last timestep is what we're interested in
        x = torch.cat((x, is_root.unsqueeze(-1).float()), dim=1)  # Concatenate is_root flag
        x = self.fc(x)
        return x

def convert_to_tensors(data):
    """
    Converts data into tensors for model input.
    :param data: List of tuples (sentence, vector, sentiment, is_root).
    :return: Tensors for model input (x, y, is_root).
    """
    x = torch.tensor([item[1] for item in data], dtype=torch.float32)
    y = torch.tensor([item[2] for item in data], dtype=torch.long)
    is_root = torch.tensor([item[3] for item in data], dtype=torch.float32)
    return x, y, is_root

if __name__ == "__main__":
    # Placeholder for model instantiation and example usage
    input_size = 300  # Size of GloVe vectors
    hidden_size = 50  # RNN hidden layer size
    num_classes = 5   # Assuming 5 sentiment classes for classification

    model = SimpleRNN(input_size, hidden_size, num_classes)
    print("Model structure: ", model)


