import numpy as np
import config as C
import utils
import os
import jsonpickle
import random
import torch

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

def sample_one_trajectory(theta,env,cov_matrix):
    this_trajectory_rewards = []
    this_trajectory_states = []
    this_trajectory_actions = []
    this_trajectory_log_prob_action_states = []


    #first, initialize the observation
    observation = env.reset()
    current_feature = utils.extract_features(observation, C.output_dim)
    
    for time_index in range(0,200):

        #compute an action given current observation
        action, log_prob = utils.compute_action_distribution(theta, current_feature, cov_matrix, mode = 'train')

        #apply the action to the environment
        observation, reward, done, info = env.step(action)

        #record reward and grad
        this_trajectory_rewards.append(reward)
        this_trajectory_states.append(current_feature)
        this_trajectory_actions.append(action[0])
        this_trajectory_log_prob_action_states.append(log_prob)

        #next time step
        current_feature = utils.extract_features(observation, C.output_dim)

        if done:
            break

    sample_dict = {}
    sample_dict['rewards'] = this_trajectory_rewards
    sample_dict['states'] = this_trajectory_states
    sample_dict['actions'] = this_trajectory_actions
    sample_dict['log_prob'] = this_trajectory_log_prob_action_states

    total_reward = np.sum(this_trajectory_rewards)

    return sample_dict, total_reward


def sample(theta, env, N, replay_buffer, replay_buffer_rewards,cov_matrix):
    """ samples N trajectories using the current policy
    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout
    Note: the maximum trajectory length is 200 steps
    """
    #replay_buffer_rewards = []
    #replay_buffer = []

    total_reward = 0

    for _ in range(0,N):
        one_trajectory_data,one_trajectory_total_reward = sample_one_trajectory(theta,env,cov_matrix)

        #drop old simulation experience
        if len(replay_buffer) > C.replay_buffer_size:
            drop_index = np.argmin(replay_buffer_rewards)
            replay_buffer.pop(drop_index)
            replay_buffer_rewards.pop(drop_index)

        replay_buffer.append(one_trajectory_data)
        replay_buffer_rewards.append(one_trajectory_total_reward)

  
        total_reward += one_trajectory_total_reward


    current_batch_reward = total_reward/N


    return replay_buffer, replay_buffer_rewards, current_batch_reward


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

    #cov matrix for the exploration part of sampling
    variance = torch.full(size=(C.output_dim,), fill_value=C.variance_for_exploration)
    cov_matrix = torch.diag(variance)

    #inv_cov_matrix for computing log grad of action distribution
    inv_cov_matrix_diag = np.ones(C.output_dim)*(1.0/C.variance_for_exploration)
    inv_cov_matrix = np.diag(inv_cov_matrix_diag)

    replay_buffer = []
    replay_buffer_rewards = []
    optimization_history_list = []

    for iteration_index in range(0,T):
        #first, I sample the grads and the rewards
        replay_buffer,replay_buffer_rewards,current_batch_reward = sample(theta, env, N, replay_buffer, replay_buffer_rewards,cov_matrix)

        #record the optimization process
        optimization_history_list.append(current_batch_reward)
        optimization_history = {}
        optimization_history['objective_history'] = optimization_history_list
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'optimization_history.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history))

        print('this is training epoch',iteration_index)
        print('the current reward is',current_batch_reward)

        for _ in range(0,C.max_offline_training):

            #sample experience from the replay buffer for training
            # new_replay_buffer_rewards = []
            # for entry in replay_buffer_rewards:
            #     new_replay_buffer_rewards.append(np.log(entry*-1)*-1) #because the reward is negative here
            # sample_probability = (np.exp(new_replay_buffer_rewards))/np.sum(np.exp(new_replay_buffer_rewards)) #apply softmax to the total_reward list
            sampled_off_line_data = []
            for sample_counter in range(0,C.batch_size):
                #sampled_index = np.random.choice(np.arange(0, len(replay_buffer)), p=sample_probability.tolist())
                sampled_index = random.randint(0,len(replay_buffer)-1)
                sampled_off_line_data.append(replay_buffer[sampled_index])

            #update model

            #gradient of the value function
            value_function_gradient, grads_for_fisher_matrix = utils.compute_value_gradient(sampled_off_line_data,theta, cov_matrix, inv_cov_matrix)

            #fisher matrix
            fisher_matrix = utils.compute_fisher_matrix(grads_for_fisher_matrix)

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


    return theta