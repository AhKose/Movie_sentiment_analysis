import torch
import torch.nn as nn
from data_preprocessing import load_and_process_data, vectorize_glove  

# Convert data to torch tensors
def convert_to_tensors(data):
    x = torch.tensor([item[1] for item in data], dtype=torch.float32)
    y = torch.tensor([item[2] for item in data])
    is_root = torch.tensor([item[3] for item in data])
    return x, y, is_root

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size + 1, num_classes)  # +1 for root flag

    def forward(self, x, is_root):
        x, _ = self.rnn(x)
        # Concatenate is_root flag to the RNN output
        x = torch.cat((x, is_root.unsqueeze(-1).float()), dim=-1)
        x = self.fc(x)
        return x
