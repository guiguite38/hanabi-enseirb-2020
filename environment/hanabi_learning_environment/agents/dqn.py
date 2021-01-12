import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        # We begin with a thre layer fully connected neural network
        self.ln1 = nn.Linear(input_size, 254)
        self.bn1 = nn.BatchNorm1d(254)
        self.ln2 = nn.Linear(254, 254)
        self.bn2 = nn.BatchNorm1d(254)
        self.ln3 = nn.Linear(254, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.ln4 = nn.Linear(128, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.bn1(F.relu(self.ln1(x)))
        x = self.bn2(F.relu(self.ln2(x)))
        x = self.bn3(F.relu(self.ln3(x)))
        return self.ln4(x)
