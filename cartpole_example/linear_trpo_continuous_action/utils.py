from sklearn.kernel_approximation import RBFSampler
from torch.distributions import MultivariateNormal
import numpy as np
import config as C
import torch

rbf_feature = RBFSampler(gamma=1, n_components = C.extracted_feature_size, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions
    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_action_distribution(theta, phis, mode):
    """ compute probability distrubtion over actions
    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    """

    mean_action = np.transpose(theta)@phis

    if mode == 'testing':
        return mean_action

    variance = torch.full(size=(C.output_dim,), fill_value=C.variance_for_exploration)
    cov_matrix = torch.diag(variance)

    mean_action = torch.tensor(mean_action, dtype=torch.float)
    #create a distribution with the mean_action and some noise for exploration
    dist = MultivariateNormal(mean_action, cov_matrix)
    
    # Sample an action from the distribution
    action = dist.sample()

    return action.detach().numpy()


def compute_log_grad(theta, phis, action):
    """ computes the log grad
    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action: action taken
    :return: log softmax gradient (shape d x 1)
    """

    #convert action to a vector
    action = np.asarray(action)
    action = np.reshape(action,(C.output_dim,1))

    #the derivative is du/dtheta * dlog(gaussian)/du, where u is the mean vector of the gaussian distribution
    mean_action = np.transpose(theta)@phis

    inv_cov_matrix_diag = np.ones(C.output_dim)*(1.0/C.variance_for_exploration)
    inv_cov_matrix = np.diag(inv_cov_matrix_diag)

    dlog_gaussian_du = inv_cov_matrix@(action-mean_action)
    du_dtheta = phis

    grad = du_dtheta@dlog_gaussian_du

    return grad


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards
    :param grads: list of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    assert len(grads)>0
    assert len(rewards)>0

    baseline_b = 0
    for sample_index in range(0,len(rewards)):
        total_reward_for_one_trajectory = np.sum(rewards[sample_index])
        baseline_b += total_reward_for_one_trajectory
    baseline_b = baseline_b/len(rewards)


    total_grad = np.zeros((grads[0][0].shape))

    for sample_index in range(0,len(grads)):
        total_grad_sum_one_trajectory = np.zeros((grads[0][0].shape))
        current_grad_trajectory = grads[sample_index]
        current_reward_trajectory = rewards[sample_index]
        current_total_reward = np.sum(current_reward_trajectory)

        for time_index_h in range(0,len(current_grad_trajectory)):
            
            #computing reward to go
            # current_reward = 0
            # for time_index_t in range(time_index_h,len(current_reward_trajectory)):
            #     current_reward += current_reward_trajectory[time_index_t]
            # adjusted_reward = current_reward-baseline_b


            total_grad_sum_one_trajectory += current_grad_trajectory[time_index_h]*(current_total_reward-baseline_b)

        total_grad += total_grad_sum_one_trajectory/len(current_grad_trajectory)

    return total_grad/len(grads)


def compute_fisher_matrix(grads):
    """ computes the fisher information matrix using the sampled trajectories gradients
    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :return: fisher information matrix (shape d x d)
    Note: don't forget to take into account that trajectories might have different lengths
    """

    lambda_regularizer = 1e-3

    assert len(grads) > 0 

    fisher_matrix = np.zeros((grads[0][0].shape[0],grads[0][0].shape[0]))

    for sample_index in range(0,len(grads)):
        one_grads = grads[sample_index]
        
        summand = np.zeros((one_grads[0].shape[0],one_grads[0].shape[0]))
        for time_step in range(0,len(one_grads)):
            summand += one_grads[time_step]@np.transpose(one_grads[time_step])
        summand = summand/len(one_grads)

        fisher_matrix += summand
    
    fisher_matrix = fisher_matrix/len(grads)+lambda_regularizer*np.eye(grads[0][0].shape[0])
    return fisher_matrix



def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent
    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    regularizer = 1e-6
    denominator = np.transpose(v_grad)@np.linalg.inv(fisher)@v_grad+regularizer
    return (delta/denominator)**(1/2)