import gym
import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
import trading_vix_env

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

    model = PPO('MlpPolicy', env, verbose=1,n_steps=500,batch_size = 10000)
    model.learn(total_timesteps=2500000000)

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    main()