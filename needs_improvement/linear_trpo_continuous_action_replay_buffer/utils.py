from sklearn.kernel_approximation import RBFSampler
from torch.distributions import MultivariateNormal
import numpy as np
import config as C
import torch
import random

rbf_feature = RBFSampler(gamma=1, n_components = C.extracted_feature_size, random_state=12345)
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

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


def compute_action_distribution(theta, phis, cov_matrix, mode):
    """ compute probability distrubtion over actions
    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    """

    mean_action = np.transpose(theta)@phis

    if mode == 'testing':
        return mean_action

    mean_action = torch.tensor(mean_action, dtype=torch.float)
    #create a distribution with the mean_action and some noise for exploration
    dist = MultivariateNormal(mean_action, cov_matrix)
    
    # Sample an action from the distribution
    action = dist.sample()

    #compute the log prob of this action-state pair
    log_prob = dist.log_prob(action)

    return action.detach().numpy(), log_prob.detach()


def compute_log_grad(theta, phis, action, inv_cov_matrix):
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

    dlog_gaussian_du = inv_cov_matrix@(action-mean_action)
    du_dtheta = phis

    grad = du_dtheta@dlog_gaussian_du

    return grad


def compute_value_gradient(sampled_off_line_data, theta,cov_matrix,inv_cov_matrix):

    '''
    theta is the current model parameters
    '''
    total_grad = np.zeros((C.extracted_feature_size,1))
    grads_for_fisher_matrix = []

    baseline_b = []
    for sample_dict in sampled_off_line_data:
        baseline_b.append(np.sum(sample_dict['rewards']))
    baseline_b = np.mean(baseline_b)

    for sample_dict in sampled_off_line_data:
        this_trajectory_rewards = sample_dict['rewards']
        this_trajectory_states = sample_dict['states'] #the states are sampled features from the RBFSampler
        this_trajectory_actions = sample_dict['actions']
        this_trajectory_log_prob_action_states = sample_dict['log_prob']


        actor_log_prob_mean = np.squeeze(np.asarray(this_trajectory_states))@theta
        actor_log_prob_mean_tensor = torch.tensor(actor_log_prob_mean, dtype=torch.float)

        dist = MultivariateNormal(actor_log_prob_mean_tensor, cov_matrix)

        #the shape of actor_log_prob should be torch.size([trajectory_length])
        actor_log_prob = dist.log_prob(torch.tensor(this_trajectory_actions))
        
        #calculate the ratio for adjusting off-line data
        ratios = torch.exp(actor_log_prob - torch.tensor(this_trajectory_log_prob_action_states))
        ratio = torch.prod(ratios).numpy()

        #print('the ratio is',ratio)

        current_total_reward = np.sum(this_trajectory_rewards)
        grads_for_fisher_matrix_one_trajectory = []

        total_grad_sum_one_trajectory = np.zeros((C.extracted_feature_size,1))
        for time_index_h in range(0,len(this_trajectory_states)):
            state_at_time_h = this_trajectory_states[time_index_h]
            action_at_time_h = this_trajectory_actions[time_index_h]
            grad_log_at_time_h = compute_log_grad(theta,state_at_time_h,action_at_time_h,inv_cov_matrix)
            

            grads_for_fisher_matrix_one_trajectory.append(grad_log_at_time_h*np.sqrt(ratio))
            #grads_for_fisher_matrix_one_trajectory.append(grad_log_at_time_h*ratio)
            #grads_for_fisher_matrix_one_trajectory.append(grad_log_at_time_h)
            total_grad_sum_one_trajectory += ratio*grad_log_at_time_h*(current_total_reward-baseline_b)

        total_grad += total_grad_sum_one_trajectory
        grads_for_fisher_matrix.append(grads_for_fisher_matrix_one_trajectory)

    return total_grad/len(sampled_off_line_data),grads_for_fisher_matrix


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