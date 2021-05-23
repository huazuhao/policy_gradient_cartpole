from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


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


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    logits_after_subtracting_max = logits-max(logits)
    r = np.exp(logits_after_subtracting_max)/sum(np.exp(logits_after_subtracting_max))
    return r.reshape(-1,1)


def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    logits = np.sum(theta*phis,axis = 0)
    r = compute_softmax(logits,axis = 0)
    return r.reshape(1,-1)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """
    denominator_in_softmax = np.sum(np.exp(np.sum(theta*phis,axis = 0)))
    
    numerator_in_softmax = np.exp(phis[:,action_idx]@theta)

    derivative_of_exp_theta_phis = np.exp(np.sum(theta*phis,axis = 0))*phis 
    #each column is a derivative of the numerator in softmax with respect to theta
    #i can choose which column to use with action_idx

    first_term = derivative_of_exp_theta_phis[:,action_idx]/denominator_in_softmax
    second_term = -1*numerator_in_softmax/denominator_in_softmax**2*np.sum(derivative_of_exp_theta_phis,axis = 1)

    r = denominator_in_softmax/numerator_in_softmax*(first_term+second_term)
    return r.reshape(-1,1)




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
        total_reward_for_one_trajectory = 0
        current_reward_trajectory = rewards[sample_index]

        for time_index in range(0,len(current_reward_trajectory)):
            total_reward_for_one_trajectory += current_reward_trajectory[time_index]
        baseline_b += total_reward_for_one_trajectory

    baseline_b = baseline_b/len(rewards)

    total_grad = np.zeros((grads[0][0].shape))

    for sample_index in range(0,len(grads)):
        total_grad_sum_one_trajectory = np.zeros((grads[0][0].shape))
        current_grad_trajectory = grads[sample_index]
        current_reward_trajectory = rewards[sample_index]
        

        for time_index_h in range(0,len(current_grad_trajectory)):
            

            current_reward = 0
            for time_index_t in range(time_index_h,len(current_reward_trajectory)):
                current_reward += current_reward_trajectory[time_index_t]
            adjusted_reward = current_reward-baseline_b

            total_grad_sum_one_trajectory += current_grad_trajectory[time_index_h]*adjusted_reward

        total_grad += total_grad_sum_one_trajectory/len(current_grad_trajectory)

    return total_grad/len(grads)
    

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

