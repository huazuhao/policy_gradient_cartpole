import numpy as np
import utils
import matplotlib.pyplot as plt
import simple_continuous_buy_sell_spy


def sample_one_reward(theta,env):
    this_trajectory_reward = []
    this_trajectory_grads = []

    #first, initialize the observation
    observation = env.reset()
    current_feature = utils.extract_features(observation)
    
    for time_index in range(0,200):


        #compute an action given current observation
        action = utils.compute_action(theta, current_feature)
        #print('the action is',action)

        #apply the action to the environment
        observation, reward = env.step(action)
        #print('the obs is',observation)
        #print('the reward is',reward)
        #print(' ')

        this_trajectory_reward.append(reward)
        log_policy_grad = utils.compute_log_policy_grad(theta, current_feature, action)
        this_trajectory_grads.append(log_policy_grad)

        current_feature = utils.extract_features(observation)

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

    #env.reset_sample()

    total_rewards = []
    total_grads = []

    for _ in range(0,N):
        one_trajectory_reward,one_trajectory_grads = sample_one_reward(theta,env)


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
    theta = np.random.rand(200,1)

    env = simple_continuous_buy_sell_spy.simple_continuous_buy_sell_spy(N)

    episode_rewards = []

    for iteration_index in range(0,T):
        #first, I sample the grads and the rewards
        trajectories_grads,trajectories_reward = sample(theta, env, N)

        total_reward = 0
        for trajectory_index in range(0,len(trajectories_reward)):
            current_reward = trajectories_reward[trajectory_index]
            total_reward += np.sum(current_reward)
        total_reward = total_reward/len(trajectories_reward)
        print('total_reward.append(',total_reward,')')

        episode_rewards.append(total_reward)

        #gradient of the value function
        value_function_gradient = utils.compute_value_gradient(trajectories_grads, trajectories_reward)

        #fisher matrix
        fisher_matrix = utils.compute_fisher_matrix(trajectories_grads)

        #step size
        step_size = utils.compute_eta(delta, fisher_matrix, value_function_gradient)

        #update theta
        update_theta = step_size*np.linalg.inv(fisher_matrix)@value_function_gradient
        if np.isfinite(update_theta).all() == True:
            theta += update_theta
        else:
            theta

    return theta, episode_rewards




if __name__ == '__main__':
    np.random.seed(123)


    # best_theta_so_far = None
    # best_episode_reward = float('-inf')

    # for _ in range(0,1):
    #     theta, episode_rewards = train(N=100, T=10, delta=1e-2)
    #     if episode_rewards >= best_episode_reward:
    #         best_episode_reward = episode_rewards
    #         best_theta_so_far = theta
    #         print('update best_theta_so_far')

    # theta = best_theta_so_far

    theta, episode_rewards = train(N=50, T=1000, delta=1e-3)

    visualize_time_length = 50

    price_history = []
    action_history1 = np.zeros((1,visualize_time_length+1))
    action_history2 = np.zeros((1,visualize_time_length+1))
    

    env = simple_continuous_buy_sell_spy.simple_continuous_buy_sell_spy(1)
    price,observation = env.reset(return_price = True)

    price_history.append(price)
    

    current_feature = utils.extract_features(observation)

    for t in range(visualize_time_length):
        #compute an action given current observation
        action = utils.compute_action(theta, current_feature)
        #apply the action to the environment
        price, observation,execute_action,need_to_buy,need_to_sell = env.step(action,return_price=True)
        #compute the next feature vector
        current_feature = utils.extract_features(observation)

        price_history.append(price)
        if execute_action:
            if need_to_buy:
                action_history1[0,t+1] = 2
                #print('record buy action')
            if need_to_sell:
                action_history1[0,t+1] = 1
                #print('record sell action')

        action_history2[0,t+1] = action            

    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('price history', color=color)
    ax1.plot(price_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('action', color=color)  # we already handled the x-label with ax1
    ax2.plot(action_history2[0,:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    already_plotted_sell_legend = False
    already_plotted_buy_legend = False
    print('begin to plot')
    plt.plot(price_history)
    for time_index in range(0,len(action_history1[0])):

        if action_history1[0,time_index]==2:
            #print('the buy price is',price_history[time_index])
            if already_plotted_sell_legend == False:
                plt.scatter(time_index,price_history[time_index],color = 'b',label = 'buy')
                already_plotted_sell_legend = True
            else:
                plt.scatter(time_index,price_history[time_index],color = 'b')

        elif action_history1[0,time_index]==1:
            #print('the sell price is',price_history[time_index])
            #print(' ')
            if already_plotted_buy_legend == False:
                plt.scatter(time_index,price_history[time_index],color = 'r',label = 'sell')
                already_plotted_buy_legend = True
            else:
                plt.scatter(time_index,price_history[time_index],color = 'r')
    plt.legend()
    plt.show()
