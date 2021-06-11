import config as C
import torch
import gym
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
from torch.utils.tensorboard import SummaryWriter
import agent
import torch.optim as optim
from collections import deque
import os
import play_episode
import utils
import numpy as np

class PolicyGradient:
    def __init__(self, problem: str = "CartPole", use_cuda: bool = False):

        self.NUM_EPOCHS = C.NUM_EPOCHS
        self.ALPHA = C.ALPHA
        self.BATCH_SIZE = C.BATCH_SIZE
        self.GAMMA = C.GAMMA
        self.HIDDEN_SIZE = C.HIDDEN_SIZE
        self.BETA = C.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        self.max_cpu_worker = mp.cpu_count()
        if self.max_cpu_worker > 1:
            self.max_cpu_worker -= 5
        self.max_cpu_worker = 1

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


        self.agent.share_memory()
        set_start_method('spawn')
        torch.cuda.empty_cache()


    def solve_environment(self):

        # init the epoch counter
        epoch = 0

        # init the epoch arrays
        # used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)


        while epoch<self.NUM_EPOCHS:

            parameter = {}
            parameter['env'] = self.env
            parameter['device'] = self.DEVICE
            parameter['action_space_size'] = self.action_space_size
            parameter['agent'] = self.agent
            parameter['gamma'] = self.GAMMA

            q = mp.Queue(maxsize = self.max_cpu_worker)

            for iteration_index in range(0,int(self.BATCH_SIZE/self.max_cpu_worker)+1):
                p_list = []
                for cpu_worker in range(0,self.max_cpu_worker):
                    
                    episode = iteration_index*self.max_cpu_worker+cpu_worker

                    if episode<self.BATCH_SIZE:
                        print('this is episode',episode)
                        p = mp.Process(target =  play_episode.play_episode,\
                                                    args = (q,parameter))
                        p.start()
                        p_list.append(p)
                    
                    else:
                        pass

                
                for j in range(len(p_list)):
                    res = q.get()

                    # after each episode append the sum of total rewards to the deque
                    self.total_rewards.append(res[2])

                    # append the weighted log-probabilities of actions
                    epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, res[0]),
                                                        dim=0)

                    # append the logits - needed for the entropy bonus calculation
                    epoch_logits = torch.cat((epoch_logits, res[1]), dim=0)


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

            self.writer.add_scalar(tag='Average Return',
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