import utils
import numpy as np
import gym
from gym import wrappers
import policy
import train
import config as C


if __name__ == '__main__':

    #work_dir = utils.mkdir('exp', 'brs')
    #monitor_dir = utils.mkdir(work_dir, 'monitor')

    np.random.seed(0)
    env = gym.make('BipedalWalker-v3')
    env.seed(0)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    #policy_inputs_size = env.observation_space.shape[0]
    policy_outputs_size = env.action_space.shape[0]
    policy = policy.Policy(C.extracted_feature_size, policy_outputs_size)
    normalizer = utils.Normalizer(C.extracted_feature_size)
    train.train(env, policy, normalizer)