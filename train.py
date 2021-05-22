import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample_one_reward(theta,env,num_actions):
    this_trajectory_reward = []
    this_trajectory_grads = []

    #first, initialize the observation
    observation = env.reset()
    current_feature = utils.extract_features(observation, num_actions)
    
    for time_index in range(0,200):
        #compute an action given current observation
        action_distribution = utils.compute_action_distribution(theta, current_feature)
        #print("the action distribution is",action_distribution)
        #action = np.argmax(action_distribution) This is not correct
        #action = np.random.binomial(1,action_distribution[0][1],1)[0]
        action = np.random.choice(num_actions,1,p = action_distribution[0])[0]
        #print("the action is",action)
        #apply the action to the environment
        observation, reward, done, info = env.step(action)

        this_trajectory_reward.append(reward)
        log_softmax_grad = utils.compute_log_softmax_grad(theta, current_feature, action)
        this_trajectory_grads.append(log_softmax_grad)

        current_feature = utils.extract_features(observation, num_actions)

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
    num_actions = 2

    total_rewards = []
    total_grads = []

    for _ in range(0,N):
        one_trajectory_reward,one_trajectory_grads = sample_one_reward(theta,env,num_actions)

        total_grads.append(one_trajectory_grads)
        total_rewards.append(one_trajectory_reward)

    return total_grads, total_rewards


def train(N, T, delta):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')
    env.seed(12345)

    episode_rewards = []

    for iteration_index in range(0,T):
        #first, I sample the grads and the rewards
        trajectories_grads,trajectories_reward = sample(theta, env, N)

        total_reward = 0
        for trajectory_index in range(0,len(trajectories_reward)):
            current_reward = trajectories_reward[trajectory_index]
            total_reward += np.sum(current_reward)
        total_reward = total_reward/len(trajectories_reward)
        print('total reward is',total_reward)

        episode_rewards.append(total_reward)

        #gradient of the value function
        value_function_gradient = utils.compute_value_gradient(trajectories_grads, trajectories_reward)

        #fisher matrix
        fisher_matrix = utils.compute_fisher_matrix(trajectories_grads)

        #step size
        step_size = utils.compute_eta(delta, fisher_matrix, value_function_gradient)

        #update theta
        theta += step_size*np.linalg.inv(fisher_matrix)@value_function_gradient


    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)

    # env = gym.make('CartPole-v0')
    # env.seed(12345)
    # observation = env.reset()
    # num_actions = 2
    # current_feature = utils.extract_features(observation, num_actions)
    # for t in range(200):
    #     env.render()

    #     #compute an action given current observation
    #     action_distribution = utils.compute_action_distribution(theta, current_feature)
    #     #action = np.argmax(action_distribution) This is not correct
    #     action = np.random.binomial(1,action_distribution[0][1],1)[0]
    #     #apply the action to the environment
    #     observation, reward, done, info = env.step(action)
    #     #compute the next feature vector
    #     current_feature = utils.extract_features(observation, num_actions)

    #     if done:
    #         print("Episode finished after {} timesteps".format(t+1))
    #         break


    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
