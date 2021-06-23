import config as C
import numpy as np
import gym
import utils
import matplotlib.pyplot as plt
import jsonpickle
import train
import os

if __name__ == '__main__':

    np.random.seed(1234)
    env = gym.make('BipedalWalker-v3')
    env.seed(12345)

    theta, episode_rewards = train.train(N=C.batch_size, T=C.training_epoch, delta=1e-2,env=env)
    '''
    param N: number of trajectories to sample in each time step
    param T: number of iterations to train the model
    param delta: trust region size, because we are using trpo
    param env: the environment for the policy to learn
    '''

    #test the training result
    observation = env.reset()
    current_feature = utils.extract_features(observation,C.output_dim)
    for t in range(200):

        env.render()

        #compute an action given current observation
        action = utils.compute_action_distribution(theta, current_feature, mode = 'test')

        #apply the action to the environment
        observation, reward, done, info = env.step(action)

        #compute the next feature vector
        current_feature = utils.extract_features(observation, C.output_dim)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


    #plot the training history
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()