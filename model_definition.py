import torch
import torch.nn as nn

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
