import config as C
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
import torch.optim as optim
from collections import deque
import agent
import play_episode
import utils
import numpy as np
import os

class PolicyGradient:
    def __init__(self, problem: str = "CartPole", use_cuda: bool = False):

        self.NUM_EPOCHS = C.NUM_EPOCHS
        self.ALPHA = C.ALPHA
        self.BATCH_SIZE = C.BATCH_SIZE
        self.GAMMA = C.GAMMA
        self.HIDDEN_SIZE = C.HIDDEN_SIZE
        self.BETA = C.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        torch.manual_seed(0)

        # instantiate the tensorboard writer
        self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'NH={self.HIDDEN_SIZE},'
                                            f'BETA={self.BETA}')

        # create the environment
        #self.env = gym.make('CartPole-v1') if problem == "CartPole" else gym.make('LunarLander-v2')

        if problem == "CartPole":
            self.env = gym.make('CartPole-v1')
        else:
            raise Exception("gym environment is not correct")
        
        self.action_space_size = self.env.action_space.n

        # the agent driven by a neural network architecture
        self.agent = agent.Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)

        self.total_rewards = deque([], maxlen=self.BATCH_SIZE)


    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        episode = 0
        epoch = 0

        # init the epoch arrays
        # used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while epoch<self.NUM_EPOCHS:

            # play an episode of the environment
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             episode) = play_episode.play_episode(self.env,self.DEVICE,self.action_space_size,self.agent,self.GAMMA,episode)

            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # if the epoch is over - we have epoch trajectories to perform the policy gradient
            if episode > self.BATCH_SIZE:

                # reset the rendering flag
                self.finished_rendering_this_epoch = False

                # reset the episode count
                episode = 0

                # increment the epoch
                epoch += 1 

                # calculate the loss
                loss, entropy = utils.calculate_loss(self.BETA, epoch_logits=epoch_logits,
                                                    weighted_log_probs=epoch_weighted_log_probs)

                # zero the gradient
                self.adam.zero_grad()

                # backprop
                loss.backward()

                # update the parameters
                self.adam.step()

                # feedback
                #print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                #      end="",
                #      flush=True)
                print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                      flush=True)

                self.writer.add_scalar(tag='Average Return over 100 episodes',
                                       scalar_value=np.mean(self.total_rewards),
                                       global_step=epoch)

                self.writer.add_scalar(tag='Entropy',
                                       scalar_value=entropy,
                                       global_step=epoch)

                # reset the epoch arrays
                # used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

                # check if solved
                if np.mean(self.total_rewards) > 200:
                    print('\nSolved!')
                    print("\nSaving the final neural network")

                    #save the neural network in the end
                    cwd = os.getcwd()
                    parameter_file = 'cartpole_nn_trained_model.pt'
                    cwd = os.path.join(cwd,parameter_file)
                    torch.save(self.agent.state_dict(),cwd)

                    break

        # close the environment
        self.env.close()

        # close the writer
        self.writer.close()
