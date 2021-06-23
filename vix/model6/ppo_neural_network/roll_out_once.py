import utils
import torch
import numpy as np
from sys import getsizeof
import time

def roll_out_once(q, env,model,cov_matrix):
    '''
    We let the agent interact with the environment for one complete episode


    The return data is
        1. observation_list 
        A list of observation of the environment at each time step. 

        2. action_list
        A list of actions taken at each time step.
        
        3. log_prob_action_list
        A list of the log(pi(Action|State)) at each time step.
        
        4. reward_list
        A list of reward at each time step. 
    '''

    observation_list = []
    action_list = []
    log_prob_action_list = []
    reward_list = []

    execute_sell = False
    null_objective = True

    state = env.reset()
    state = np.reshape(state,(-1,))

    for time_index in range(0,200):

        # Calculate action and make a step in the env. 
        # Note that rew is short for reward.
        action,log_prob = utils.next_action_during_training(model,cov_matrix,state)


        #record data
        observation_list.append(state)
        action_list.append(action)
        log_prob_action_list.append(log_prob)

        #take one more time step
        state, reward, execute_sell  = env.step(action[0])

        if execute_sell and null_objective:
            null_objective = False

        state = np.reshape(state,(-1,))

    reward = env.final()

    if null_objective:
        for _ in range(0,len(observation_list)):
            reward_list.append(-1e9)
    else:
        for _ in range(0,len(observation_list)):
            reward_list.append(reward)

    # Reshape data as tensors in the shape specified in function description, before returning
    observation_list = torch.tensor(observation_list, dtype=torch.float)
    action_list = torch.tensor(action_list, dtype=torch.float)
    log_prob_action_list = torch.tensor(log_prob_action_list, dtype=torch.float)  

    temp_dict = {}
    temp_dict['observation_list'] = observation_list
    temp_dict['action_list'] = action_list
    temp_dict['log_prob_action_list'] = log_prob_action_list
    temp_dict['reward_list'] = reward_list

    q.put([temp_dict])
    q.close()
    time.sleep(2)

