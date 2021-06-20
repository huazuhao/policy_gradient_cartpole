from torch.distributions import MultivariateNormal
import torch
import numpy as np

def next_action_during_training(model,cov_matrix,state):

    # Convert observation to tensor if it's a numpy array
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float)

    #query the actor network for a mean action
    mean_action = model(state)

    print('before')
    print('cov_matrix',cov_matrix)
    print('mean_action',mean_action)
    #create a distribution with the mean_action and some noise for exploration
    dist = MultivariateNormal(mean_action, cov_matrix)

    print('after')

    # Sample an action from the distribution
    action = dist.sample()

    # Calculate the log probability for that action
    log_prob = dist.log_prob(action)

    # Return the sampled action and the log probability of that action in our distribution
    return action.detach().numpy(), log_prob.detach()