import torch
import threading
from torch import optim
import torch.distributed.rpc as rpc
import numpy as np
import os
import jsonpickle

import policy
import config as C
import utils


class BatchUpdateParameterServer():


    def __init__(self,batch_update_size = C.UPDATE_SIZE):


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
        self.optimization_history = []
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=C.ALPHA)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    
    def get_model(self):
        #return self.model.cpu() #because rpc can only pass cpu data
        #return self.model.to('cpu')
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, rewards):
        self = ps_rref.local_value()

        utils.timed_log(f"PS got {self.curr_update_size}/{self.batch_update_size} updates")

        for p, g in zip(self.model.parameters(), grads):
            p.grad += g

        for reward in rewards:
            self.current_rewards.append(reward)
            self.curr_update_size += 1

        with self.lock:

            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size

                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                utils.timed_log(f"rewards length is {len(self.current_rewards)}")
                utils.timed_log(f"average reward is {np.mean(self.current_rewards)}")
                utils.timed_log("PS updated model")
                self.future_model = torch.futures.Future()

                #save the model
                cwd = os.getcwd()
                parameter_file = C.trained_model_name
                cwd = os.path.join(cwd,parameter_file)
                torch.save(self.model.state_dict(),cwd)

                #record optimization history
                self.optimization_history.append(np.mean(self.current_rewards))
                optimization_history = {}
                optimization_history['history'] = self.optimization_history
                #store the history
                cwd = os.getcwd()
                #cwd = os.path.join(cwd, 'data_folder')
                parameter_file = 'optimization_history.json'
                cwd = os.path.join(cwd,parameter_file)
                with open(cwd, 'w') as statusFile:
                    statusFile.write(jsonpickle.encode(optimization_history))

                self.current_rewards = []

        return fut