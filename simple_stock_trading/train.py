import numpy as np
import utils
import matplotlib.pyplot as plt
import simple_buy_sell_spy
import gym

def sample_one_reward(theta,env,num_actions):
    this_trajectory_reward = []
    this_trajectory_grads = []

    #first, initialize the observation
    observation = env.reset()
    current_feature = utils.extract_features(observation, num_actions)
    
    for time_index in range(0,200):

        #cartpole
        # #compute an action given current observation
        # action_distribution = utils.compute_action_distribution(theta, current_feature)
        # #action = np.argmax(action_distribution) This is not correct
        # action = np.random.choice(num_actions,1,p = action_distribution[0])[0]
        # #apply the action to the environment
        # observation, reward, done, info = env.step(action)

        # this_trajectory_reward.append(reward)
        # log_softmax_grad = utils.compute_log_softmax_grad(theta, current_feature, action)
        # this_trajectory_grads.append(log_softmax_grad)

        # current_feature = utils.extract_features(observation, num_actions)

        # if done:
        #     break

        #stock

        #compute an action given current observation
        action_distribution = utils.compute_action_distribution(theta, current_feature)
        action = np.random.choice(num_actions,1,p = action_distribution[0])[0]
        #apply the action to the environment
        observation, reward = env.step(action)

        this_trajectory_reward.append(reward)
        log_softmax_grad = utils.compute_log_softmax_grad(theta, current_feature, action)
        this_trajectory_grads.append(log_softmax_grad)

        current_feature = utils.extract_features(observation, num_actions)

    #final step
    action_distribution = utils.compute_action_distribution(theta, current_feature)
    action = np.random.choice(num_actions,1,p = action_distribution[0])[0]
    #apply the action to the environment
    observation, reward = env.step(action,final_step=True)
    this_trajectory_reward.append(reward)
    log_softmax_grad = utils.compute_log_softmax_grad(theta, current_feature, action)
    this_trajectory_grads.append(log_softmax_grad)


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
    num_actions = 3 #2 for cartpole, 3 for stock

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
    #env = gym.make('CartPole-v0')
    #env.seed(12345)

    env = simple_buy_sell_spy.simple_buy_sell_spy()

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


    return theta, max(episode_rewards)

if __name__ == '__main__':
    np.random.seed(1234)


    best_theta_so_far = None
    best_episode_reward = float('-inf')

    for _ in range(0,1):
        theta, episode_rewards = train(N=100, T=100, delta=1e-2)
        if episode_rewards >= best_episode_reward:
            best_episode_reward = episode_rewards
            best_theta_so_far = theta
            print('update best_theta_so_far')


    theta = best_theta_so_far

    visualize_time_length = 200

    price_history = []
    action_history = np.zeros((1,visualize_time_length+1))

    env = simple_buy_sell_spy.simple_buy_sell_spy()
    price,observation = env.reset(return_price = True)

    price_history.append(price)
    
    num_actions = 3
    current_feature = utils.extract_features(observation, num_actions)
    for t in range(visualize_time_length):
        #compute an action given current observation
        action_distribution = utils.compute_action_distribution(theta, current_feature)
        #action = np.argmax(action_distribution) This is not correct
        action = np.random.choice(num_actions,1,p = action_distribution[0])[0]
        #apply the action to the environment
        price, observation,execute_action = env.step(action,return_price=True)
        #compute the next feature vector
        current_feature = utils.extract_features(observation, num_actions)

        price_history.append(price)
        if execute_action:
            action_history[0,t+1] = action

    

    already_plotted_sell_legend = False
    already_plotted_buy_legend = False
    print('begin to plot')
    plt.plot(price_history)
    for time_index in range(0,len(action_history[0])):

        if action_history[0,time_index]==2:
            print('the buy price is',price_history[time_index])
            if already_plotted_sell_legend == False:
                plt.scatter(time_index,price_history[time_index],color = 'b',label = 'buy')
                already_plotted_sell_legend = True
            else:
                plt.scatter(time_index,price_history[time_index],color = 'b')

        elif action_history[0,time_index]==1:
            print('the sell price is',price_history[time_index])
            print(' ')
            if already_plotted_buy_legend == False:
                plt.scatter(time_index,price_history[time_index],color = 'r',label = 'sell')
                already_plotted_buy_legend = True
            else:
                plt.scatter(time_index,price_history[time_index],color = 'r')
    plt.legend()
    plt.show()

    #visualize
    # plt.plot(episode_rewards)
    # plt.title("avg rewards per timestep")
    # plt.xlabel("training iteration")
    # plt.ylabel("avg rewards")
    # plt.show()
