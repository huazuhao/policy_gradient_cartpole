import numpy as np
import config as C
import utils
import os
import jsonpickle

def sample_one_trajectory(theta,env):
    this_trajectory_reward = []
    this_trajectory_grads = []

    #first, initialize the observation
    observation = env.reset()
    current_feature = utils.extract_features(observation, C.output_dim)
    
    for time_index in range(0,200):

        #compute an action given current observation
        action = utils.compute_action_distribution(theta, current_feature, mode = 'train')

        #apply the action to the environment
        observation, reward, done, info = env.step(action[0])

        #record reward and grad
        this_trajectory_reward.append(reward)
        computed_grad_log_state_action = utils.compute_log_grad(theta, current_feature, action)
        this_trajectory_grads.append(computed_grad_log_state_action)

        current_feature = utils.extract_features(observation, C.output_dim)

        if done:
            break

    return this_trajectory_reward,this_trajectory_grads


def sample(theta, env, N):
    """ samples N trajectories using the current policy
    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout
    Note: the maximum trajectory length is 200 steps
    """

    total_rewards = []
    total_grads = []

    for _ in range(0,N):
        one_trajectory_reward,one_trajectory_grads = sample_one_trajectory(theta,env)

        total_rewards.append(one_trajectory_reward)
        total_grads.append(one_trajectory_grads)

    return total_grads, total_rewards


def train(N, T, delta,env):
    """
    param N: number of trajectories to sample in each time step
    param T: number of iterations to train the model
    param delta: trust region size
    param env: the environment for the policy to learn
    
    return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(C.extracted_feature_size,1)

    episode_rewards = [] #record the reward during the training process

    for iteration_index in range(0,T):
        #first, I sample the grads and the rewards
        trajectories_grads,trajectories_rewards = sample(theta, env, N)

        total_reward = 0
        for trajectory_index in range(0,len(trajectories_rewards)):
            current_reward = trajectories_rewards[trajectory_index]
            total_reward += np.sum(current_reward)
        total_reward = total_reward/len(trajectories_rewards)
        print('total reward is',total_reward,'and this is training epoch',iteration_index)

        episode_rewards.append(total_reward)

        #gradient of the value function
        value_function_gradient = utils.compute_value_gradient(trajectories_grads, trajectories_rewards)

        #fisher matrix
        fisher_matrix = utils.compute_fisher_matrix(trajectories_grads)

        #step size
        step_size = utils.compute_eta(delta, fisher_matrix, value_function_gradient)

        #update theta
        theta += step_size*np.linalg.inv(fisher_matrix)@value_function_gradient

        #save the learned parameter theta
        learned_parameter_theta = {}
        learned_parameter_theta['learned_parameter_theta'] = theta
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'learned_parameter_theta.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(learned_parameter_theta))


    return theta, episode_rewards