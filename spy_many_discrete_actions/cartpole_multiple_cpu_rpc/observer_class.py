import gym
import torch.distributed.rpc as rpc
import torch
import numpy as np
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax
import utils
import config as C

class observer_class():

    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.seed(0)
        
        self.action_space_size = self.env.action_space.n
        #self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DEVICE = 'cpu'

        self.gamma = C.GAMMA

    def run_one_episode(self,agent_rref):

        self.agent_policy = agent_rref.rpc_sync().get_agent_policy(self.id).to(self.DEVICE)

        # reset the environment to a random initial state every epoch
        state = self.env.reset()

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.action_space_size), device=self.DEVICE)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)


        # episode loop
        while True:

                
            # get the action logits from the agent - (preferences)
            action_logits = self.agent_policy(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))
            
            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state
            state, reward, done, _ = self.env.step(action=action.cpu().item())

            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            average_rewards = np.concatenate((average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)

            # the episode is over
            if done:

                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = utils.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.gamma)
                discounted_rewards_to_go -= average_rewards  # baseline - state specific average

                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.action_space_size)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                #in the simplest case of policy gradient, all we need to track is
                #sum_weighted_log_probs. This is all we need for backprop of the neural network

                agent_rref.rpc_sync().report_reward(self.id,sum_weighted_log_probs,episode_logits,sum_of_rewards)

                break