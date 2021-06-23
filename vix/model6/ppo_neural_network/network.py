import torch
import torch.nn as nn
from torch.nn.functional import one_hot, log_softmax, softmax, normalize

class Agent(nn.Module):

    '''
    This class defines a neural network as an agent. 
    The agent takes in observation and returns an action.
    '''

    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        torch.manual_seed(1)

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            # nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            # nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.clamp(x,0.001,1.0)
        return x