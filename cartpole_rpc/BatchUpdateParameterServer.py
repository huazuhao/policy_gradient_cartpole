import torch
import threading
from torch import optim
import torch.distributed.rpc as rpc
import numpy as np
import os

import policy
import config as C
import utils

class BatchUpdateParameterServer():


    def __init__(self,batch_update_size = C.BATCH_SIZE):


        self.HIDDEN_SIZE = C.HIDDEN_SIZE
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and C.use_cuda else 'cpu')
        self.DEVICE = 'cpu'

        self.model = policy.policy(observation_space_size=C.observation_space_size,
                           action_space_size=C.action_space_size,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.current_rewards = []
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=C.ALPHA)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    
    def get_model(self):
        #return self.model.cpu() #because rpc can only pass cpu data
        #return self.model.to('cpu')
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, reward):
        self = ps_rref.local_value()

        for p, g in zip(self.model.parameters(), grads):
            p.grad += g

        self.current_rewards.append(reward)

        with self.lock:
            self.curr_update_size += 1

            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size

                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                utils.timed_log(f"reward is {np.mean(self.current_rewards)}")
                utils.timed_log("PS updated model")
                self.future_model = torch.futures.Future()
                self.current_rewards = []

                #save the model
                cwd = os.getcwd()
                parameter_file = 'cartpole_rpc_trained_model.pt'
                cwd = os.path.join(cwd,parameter_file)
                torch.save(self.model.state_dict(),cwd)

        return fut