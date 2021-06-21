from torch.distributions import MultivariateNormal
import torch
import numpy as np
import config as C

def next_action_during_training(model,cov_matrix,state):

    # Convert observation to tensor if it's a numpy array
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float)

    #query the actor network for a mean action
    mean_action = model(state)

    #create a distribution with the mean_action and some noise for exploration
    dist = MultivariateNormal(mean_action, cov_matrix)

    # Sample an action from the distribution
    action = dist.sample()

    # Calculate the log probability for that action
    log_prob = dist.log_prob(action)

    # Return the sampled action and the log probability of that action in our distribution
    return action.detach().numpy(), log_prob.detach()


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

    return model


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad