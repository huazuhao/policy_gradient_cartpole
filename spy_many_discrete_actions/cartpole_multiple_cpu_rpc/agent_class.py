import gym
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote

import config as C
import policy
import random
import observer_class
import utils

class agent_class():

    def __init__(self,world_size):

        OBSERVER_NAME = "observer{}"

        self.NUM_EPOCHS = C.NUM_EPOCHS
        self.ALPHA = C.ALPHA
        self.BATCH_SIZE = C.BATCH_SIZE
        self.GAMMA = C.GAMMA
        self.HIDDEN_SIZE = C.HIDDEN_SIZE
        self.BETA = C.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and C.USE_GPU else 'cpu')

        self.env = gym.make('CartPole-v1') #just to get the dimension of the observation space and the action space

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.env.seed(0)

        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.total_rewards = deque([], maxlen=self.BATCH_SIZE)
        self.epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        self.epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        self.episode_counter = 0
        self.epoch_counter = 0

        # the agent driven by a neural network architecture
        self.agent = policy.policy(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)

        for observer_rank in range(1,world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(observer_rank))
            self.ob_rrefs.append(remote(ob_info, observer_class.observer_class))


    def report_reward(self,ob_id,sum_weighted_log_probs,episode_logits,sum_of_episode_rewards):

        print('before moving to cuda',sum_weighted_log_probs)

        sum_weighted_log_probs = sum_weighted_log_probs.to(self.DEVICE)
        episode_logits = episode_logits.to(self.DEVICE)

        # after each episode append the sum of total rewards to the deque
        self.total_rewards.append(sum_of_episode_rewards)

        # append the weighted log-probabilities of actions
        self.epoch_weighted_log_probs = torch.cat((self.epoch_weighted_log_probs, sum_weighted_log_probs),
                                                dim=0)

        # append the logits - needed for the entropy bonus calculation
        self.epoch_logits = torch.cat((self.epoch_logits, episode_logits), dim=0)


    def get_agent_policy(self,ob_id):
        return self.agent.to('cpu')

    
    def run_episodes(self):

        self.episode_counter = 0
        
        if self.BATCH_SIZE % len(self.ob_rrefs) == 0:
            max_iteration = int(self.BATCH_SIZE/len(self.ob_rrefs))
        else:
            max_iteration = int(self.BATCH_SIZE/len(self.ob_rrefs)+1)


        for iteration_index in range(0,max_iteration):
            
            futs = []
            for ob_rref in self.ob_rrefs:

                if self.episode_counter<self.BATCH_SIZE:
                    futs.append(
                        rpc_async(
                            ob_rref.owner(),
                            ob_rref.rpc_sync().run_one_episode,
                            args = (self.agent_rref,)
                        )
                    )
                    self.episode_counter += 1

                else:
                    pass

            #wait until all observers have finished this episode
            for fut in futs:
                fut.wait()


    def finish_episode(self):

        # increment the epoch
        self.epoch_counter += 1 

        # calculate the loss
        loss = utils.calculate_loss(self.BETA, epoch_logits=self.epoch_logits,
                                    weighted_log_probs=self.epoch_weighted_log_probs)

        # zero the gradient
        self.adam.zero_grad()

        print('loss is',loss)

        # backprop
        loss.backward()

        print(utils.getBack(loss.grad_fn))

        print('finished printing get back')

        for param in self.agent.parameters():
            print(param)
            print(param.grad.norm())
            break

        # update the parameters
        self.adam.step()

        # print current average policy reward
        print("\r", f"Epoch: {self.epoch_counter}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                      flush=True)

        # reset the epoch arrays
        self.epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        self.epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)


        return self.total_rewards