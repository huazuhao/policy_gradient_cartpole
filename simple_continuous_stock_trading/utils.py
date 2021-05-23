from sklearn.kernel_approximation import RBFSampler
import numpy as np
from scipy.stats import uniform
import tensorflow_probability as tfp
tfd = tfp.distributions

rbf_n_components = 100
rbf_feature = RBFSampler(gamma=1, n_components = rbf_n_components, random_state=12345)


def extract_features(state):
    """ This function computes the RFF features for a state for all the discrete actions
    param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    num_actions = 1
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_action(theta,phis):
    '''
    phis: RFF features of the state and actions (shape d x 1)
    theta:[theta1,theta2]. theta has shape 2d x 1.
    theta1 has shape d x 1. theta2 has shape d x 1.

    The underlying distribution of the model is Kumaraswamy
    Kumaraswamy's a is given by theta1'@phis
    Kumaraswamy's b is given by theta2'@phis

    return: action drawn according to the Kumaraswamy distribution
    '''

    theta1 = theta[0:rbf_n_components,:]
    theta2 = theta[rbf_n_components:,:]
    a = (np.transpose(theta1)@phis)[0][0]**2+1e-6
    b = (np.transpose(theta2)@phis)[0][0]**2+1e-6

    dist = tfd.Kumaraswamy(a, b)
    sample = dist.sample(1).numpy()
    action = sample[0]

    return action


def compute_log_policy_grad(theta,phis,x):
    '''
    param theta: model parameter (shape 2d x 1)
    param phis: RFF features of the state and actions (shape d x 1)
    param x: a single scalar. Here, x means the percentage of total portfolio value we want in stock
    return:log policy grad (shape 2d x 1)
    '''

    theta1 = theta[0:rbf_n_components,:]
    theta2 = theta[rbf_n_components:,:]
    a = (np.transpose(theta1)@phis)[0][0]**2+1e-6
    b = (np.transpose(theta2)@phis)[0][0]**2+1e-6

    #adjust x
    if x >=1-1e-4:
        x -= 1e-4
    if x <= 0+1e-4:
        x += 1e-4
        

    grad_log_Kumaraswamy_a_1 = a*np.log(x)*(b*x**a-1)+x**a-1
    grad_log_Kumaraswamy_a_2 = a*(x**a-1)
    grad_log_Kumaraswamy_theta1 = grad_log_Kumaraswamy_a_1/grad_log_Kumaraswamy_a_2**2.0*phis


    # grad_log_Kumaraswamy_a_1 = 1.0/(a*b)*x**(1.0-a)*(1.0-x**a)**(1.0-b)




    # grad_log_Kumaraswamy_a_2 = b*x**(a-1.0)*(1.0-x**a)**(b-1.0)
    # grad_log_Kumaraswamy_a_3 = a*b*x**(a-1.0)*np.log(x)*(1.0-x**a)**(b-1.0)
    # grad_log_Kumaraswamy_a_4 = -1.0*a*(b-1.0)*b*x**(2.0*a-1.0)*np.log(x)*(1.0-x**a)**(b-2.0)
    # grad_log_Kumaraswamy_a = grad_log_Kumaraswamy_a_1*(grad_log_Kumaraswamy_a_2+\
    #                                                    grad_log_Kumaraswamy_a_3+grad_log_Kumaraswamy_a_4)
    # grad_log_Kumaraswamy_theta1 = grad_log_Kumaraswamy_a*2.0*phis

    grad_log_Kumaraswamy_b = np.log(1.0-x**a)+1.0/b
    grad_log_Kumaraswamy_theta2 = grad_log_Kumaraswamy_b*2.0*phis

    grad = np.concatenate((grad_log_Kumaraswamy_theta1, grad_log_Kumaraswamy_theta2), axis=0)
    
    return grad


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

    try:
        (delta/denominator)**(1/2)
    except:
        print('inside compute eta')
        print('delta is',delta)
        print('denominator is',denominator)
        print('with no regularizer is',np.transpose(v_grad)@np.linalg.inv(fisher)@v_grad)


    return (delta/denominator)**(1/2)