import torch
import numpy as np
from torch.distributions import Categorical
import utils
from torch.nn.functional import one_hot, log_softmax


def play_episode(environment, device, action_space_size, agent, gamma, episode: int):

        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """

        # reset the environment to a random initial state every epoch
        state = environment.reset()

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=device)
        episode_logits = torch.empty(size=(0, action_space_size), device=device)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        # episode loop
        while True:

                
            # get the action logits from the agent - (preferences)
            action_logits = agent(torch.tensor(state).float().unsqueeze(dim=0).to(device))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state
            state, reward, done, _ = environment.step(action=action.cpu().item())

            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)
            

            # the episode is over
            if done:

                # increment the episode
                episode += 1

                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # simply compute the total reward as supposed to reward to go
                vanila_rewards = np.ones(episode_actions.shape)*sum_of_rewards

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=environment.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(vanila_rewards).float().to(device)

                # calculate the sum over trajectory of the weighted log-probabilities
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                sum_weighted_log_probs = sum_weighted_log_probs.to('cpu')
                episode_logits = episode_logits.to('cpu')

                sum_weighted_log_probs = sum_weighted_log_probs.to(device)
                episode_logits = episode_logits.to(device)

                return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode