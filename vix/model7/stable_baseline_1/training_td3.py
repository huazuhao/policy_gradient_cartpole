import trading_vix_env
import os
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
import custom_call_back


def main():
    # Create log dir
    log_dir = './td3_data'
    os.makedirs(log_dir, exist_ok=True)

    vix_env = trading_vix_env.trading_vix_env()
    env = Monitor(vix_env, log_dir)

    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # Create the callback: check every 20000 steps
    callback = custom_call_back.CustomCallback(check_freq = 20000,log_dir = log_dir)
    # Create RL model
    model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=2,batch_size = 10000)
    # Train the agent
    model.learn(total_timesteps=int(5e9), callback=callback)


if __name__ == '__main__':
    main()