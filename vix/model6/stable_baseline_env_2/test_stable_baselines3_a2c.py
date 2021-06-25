import gym
import numpy as np
from stable_baselines3 import A2C

from stable_baselines3.common.vec_env import SubprocVecEnv
import trading_vix_env

import custom_call_back
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env: (str) the environment ID for gym environment
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init

def main():
    #env_id = "CartPole-v1"
    vix_env = trading_vix_env.trading_vix_env()
    num_cpu = 20  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(vix_env, i) for i in range(num_cpu)])

    # Create log dir
    log_dir = './a2c_data'
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)
    callback = custom_call_back.CustomCallback(check_freq = 1000,log_dir = log_dir)

    model = A2C('MlpPolicy', env, verbose=1,n_steps=5)
    model.learn(total_timesteps=2500000000,callback = callback)

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    main()