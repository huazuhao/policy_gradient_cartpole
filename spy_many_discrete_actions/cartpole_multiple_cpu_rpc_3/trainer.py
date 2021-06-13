
import torch
import torch.distributed.rpc as rpc
import numpy as np
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax

import gym
import trading_spy_with_many_discrete_actions_reward_type_3 as stock_env
import config as C
import utils
import BatchUpdateParameterServer as bups


class trainer():

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.m = None
        self.DEVICE = 'cpu'
        self.action_space_size = C.action_space_size

    def get_next_batch(self,env):

        for _ in range(C.NUM_EPOCHS):

            epoch_logits = torch.empty(size=(0, self.action_space_size), device=self.DEVICE)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

            # reset the environment to a random initial state every epoch
            state = env.reset()

            # initialize the episode arrays
            episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
            episode_logits = torch.empty(size=(0, C.action_space_size), device=self.DEVICE)
            average_rewards = np.empty(shape=(0,), dtype=np.float)
            episode_rewards = np.empty(shape=(0,), dtype=np.float)

            # episode loop
            for step_index in range(0,C.max_simulation_length):

                # get the action logits from the agent - (preferences)
                state['price_history'].append(state['current_stock_ratio'])
                nn_input = np.asarray(state['price_history'])
                action_logits = self.m(torch.tensor(nn_input).float().unsqueeze(dim=0).to(self.DEVICE))
                #action_logits = self.m(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))


                # append the logits to the episode logits list
                episode_logits = torch.cat((episode_logits, action_logits), dim=0)

                # sample an action according to the action distribution
                action = Categorical(logits=action_logits).sample()
                
                # append the action to the episode action list to obtain the trajectory
                # we need to store the actions and logits so we could calculate the gradient of the performance
                episode_actions = torch.cat((episode_actions, action), dim=0)


                # take the chosen action, observe the reward and the next state
                state, reward, execute_action = env.step(action=action.cpu().item())
                #state, reward, done, _ = env.step(action=action.cpu().item())


                # append the reward to the rewards pool that we collect during the episode
                # we need the rewards so we can calculate the weights for the policy gradient
                # and the baseline of average
                episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

                # here the average reward is state specific
                average_rewards = np.concatenate((average_rewards,
                                                np.expand_dims(np.mean(episode_rewards), axis=0)),
                                                axis=0)

            # turn the rewards we accumulated during the episode into the rewards-to-go:
            # earlier actions are responsible for more rewards than the later taken actions
            discounted_rewards_to_go = utils.get_discounted_rewards(rewards=episode_rewards,
                                                                                gamma=C.GAMMA)
            discounted_rewards_to_go -= average_rewards  # baseline - state specific average

            # # calculate the sum of the rewards for the running average metric
            sum_of_rewards = np.sum(episode_rewards)

            # set the mask for the actions taken in the episode
            mask = one_hot(episode_actions, num_classes=C.action_space_size)

            # calculate the log-probabilities of the taken actions
            # mask is needed to filter out log-probabilities of not related logits
            episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

            # weight the episode log-probabilities by the rewards-to-go
            episode_weighted_log_probs = episode_log_probs * \
                torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

            # calculate the sum over trajectory of the weighted log-probabilities
            sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, sum_weighted_log_probs),
                                                    dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # calculate the loss
            loss, entropy = utils.calculate_loss(C.BETA, epoch_logits=epoch_logits,
                                                weighted_log_probs=epoch_weighted_log_probs)

            yield loss, sum_of_rewards



    def train(self):

        name = rpc.get_worker_info().name
        self.m = self.ps_rref.rpc_sync().get_model()


        #now we compute the gradient based on the model m
        #we play one episode of the environment
        self.env = stock_env.trading_spy(C.max_simulation_length,C.min_history_length,C.max_position,C.init_cash_value)
        #self.env = gym.make('CartPole-v1')

        for loss, sum_of_rewards in self.get_next_batch(self.env):
            
            #utils.timed_log(f"reward is {sum_of_rewards}")

            loss.backward()
            #utils.timed_log(f"{name} reporting grads")

            self.m = rpc.rpc_sync(
                self.ps_rref.owner(),
                bups.BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in self.m.cpu().parameters()],sum_of_rewards),
            )

            #utils.timed_log(f"{name} got updated model")