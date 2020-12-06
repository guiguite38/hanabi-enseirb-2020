import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        # We begin with a thre layer fully connected neural network
        input_size = 1000
        self.ln1 = nn.Linear(input_size, 254)
        self.dr1 = nn.Dropout(0.3)
        self.ln2 = nn.Linear(254, 254)
        self.dr2 = nn.Dropout(0.3)
        self.ln3 = nn.Linear(254, 254)
        self.dr3 = nn.Dropout(0.3)
        self.ln4 = nn.Linear(254, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.dr1(self.ln1(x)))
        x = F.relu(self.dr2(self.ln2(x)))
        x = F.relu(self.dr3(self.ln3(x)))
        return self.ln4(x)
