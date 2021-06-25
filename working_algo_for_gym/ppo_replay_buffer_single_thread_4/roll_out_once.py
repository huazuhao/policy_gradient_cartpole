import utils
import torch

def roll_out_once(env,model,cov_matrix):
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

    state = env.reset()
    
    for time_index in range(0,200):

        # Calculate action and make a step in the env. 
        # Note that rew is short for reward.
        action,log_prob = utils.next_action_during_training(model,cov_matrix,state)


        #record data
        observation_list.append(state)
        action_list.append(action)
        log_prob_action_list.append(log_prob)

        #take one more time step
        state, reward, done, _ = env.step(action)

        #record more data
        reward_list.append(reward)

        if done:
            break

    # Reshape data as tensors in the shape specified in function description, before returning
    observation_list = torch.tensor(observation_list, dtype=torch.float)
    action_list = torch.tensor(action_list, dtype=torch.float)
    log_prob_action_list = torch.tensor(log_prob_action_list, dtype=torch.float)  
    
    return observation_list,action_list,log_prob_action_list,reward_list


