import gym
import network
import config as C
import torch.optim as optim
import ppo_learn
import torch

if __name__ == '__main__':
    

    #initialize the environment
    env = gym.make('Pendulum-v0')
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]

    HIDDEN_SIZE = C.hidden_size
    DEVICE = C.device

    #first, we need to initialize the policy neural network model
    # the agent driven by a neural network architecture
    model = network.Agent(observation_space_size=observation_space_size,
                        action_space_size=action_space_size,
                        hidden_size=HIDDEN_SIZE).to(DEVICE)

    model_optim = optim.Adam(params=model.parameters(), lr=C.alpha)

    #initialize other variables

    replay_buffer = []
    replay_buffer_reward = []
    variance = torch.full(size=(action_space_size,), fill_value=C.variance_for_exploration)
    cov_matrix = torch.diag(variance)



    ppo_learn.ppo_learn(replay_buffer,replay_buffer_reward,env,model,cov_matrix,model_optim)
