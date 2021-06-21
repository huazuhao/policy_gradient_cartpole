import torch
import torch.nn as nn
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
import numpy as np
from torch.distributions import MultivariateNormal

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
        x = torch.clamp(x,-2.0,2.0)
        return x

    def getLogProbabilityDensity(self,states,actions,cov_matrix):

        # Convert observation to tensor if it's a numpy array
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float)

        #query the actor network for a mean action
        mean_action = self.forward(states)

        #create a distribution with the mean_action and some noise for exploration
        dist = MultivariateNormal(mean_action, cov_matrix)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(actions)

        return log_prob

    def meanKlDivergence(self, states, actions, logProbablityOld, cov_matrix):
        """
        Parameters:
        states (torch.Tensor): N_state x N_sample | The states of the samples
        actions (torch.Tensor): N_action x N_sample | The action taken for this samples
        logProbablityOld (torch.Tensor): N_sample |  Log probablility of the action, note that
            this should be detached from the gradient.

        Returns:
        torch.Tensor: Scalar | the mean of KL-divergence
        """
        logProbabilityNew = self.getLogProbabilityDensity(states,actions,cov_matrix)
        return (torch.exp(logProbablityOld)
                * (logProbablityOld - logProbabilityNew)).mean(); #Tensor kl.mean()