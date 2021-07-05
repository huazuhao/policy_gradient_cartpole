# the standard implementation of ARS

# Importing the libraries
import datetime
import os
import numpy as np
import gym
from gym import wrappers
from sklearn.kernel_approximation import RBFSampler
import config as C

rbf_feature = RBFSampler(gamma=1, random_state=12345, n_components = C.extracted_feature_size)


def extract_features(state):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    num_actions = 1 #one continuous action output
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return np.reshape(feats,(-1,))

# Setting the Hyper Parameters

class Hp():

    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.005
        self.nb_directions = 100
        self.nb_best_directions = 80
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'Pendulum-v0'


# Normalizing the states

class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        print('inside normalizer and n is',self.n)
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


# Building the AI

class Policy():

    def __init__(self, output_size):
        self.theta = np.zeros((output_size, C.extracted_feature_size))

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        #print('new theta is',self.theta)

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction=None, delta=None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while True:
        state = extract_features(state)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        #reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
        if done:
            break

    return sum_rewards


# Training the AI

def train(env, policy, normalizer, hp):
    for step in range(hp.nb_steps):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction="positive", delta=deltas[k])

        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction="negative", delta=deltas[k])

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        test_reward = [0] * hp.nb_directions
        for k in range(hp.nb_directions):
            test_reward[k] = explore(env, normalizer, policy)
        print('Step:', step, 'Reward:', np.mean(test_reward))


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Running the main code
if __name__ == '__main__':

    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')

    hp = Hp()
    np.random.seed(hp.seed)
    env = gym.make(hp.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    #nb_inputs = env.observation_space.shape[0]
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_outputs)
    normalizer = Normalizer(C.extracted_feature_size)
    train(env, policy, normalizer, hp)
