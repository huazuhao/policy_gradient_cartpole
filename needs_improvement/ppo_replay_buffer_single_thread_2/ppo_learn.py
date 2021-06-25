import config as C
import roll_out_once
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import os
import jsonpickle

def ppo_learn(replay_buffer,replay_buffer_reward,env,model,cov_matrix,model_optim):

    np.random.seed(0)
    current_best_reward = float('-inf')
    global_iteration_counter = 0
    optimization_history_list = []

    while True:

        new_sample_reward = []

        #The first step is to add more simulation result to the replay buffer
        for episode_counter in range(0,C.max_new_episode):
            observation_list,action_list,log_prob_action_list,reward_list = \
            roll_out_once.roll_out_once(env,model,cov_matrix)

            #drop old simulation experience
            if len(replay_buffer) > C.replay_buffer_size:
                drop_index = np.argmin(replay_buffer_reward)
                replay_buffer.pop(drop_index)
                replay_buffer_reward.pop(drop_index)

            #add the new simulation result to the replay buffer
            total_reward = np.sum(reward_list)
            replay_buffer_reward.append(total_reward)
            temp_dict = {}
            temp_dict['observation_list'] = observation_list
            temp_dict['action_list'] = action_list
            temp_dict['log_prob_action_list'] = log_prob_action_list
            temp_dict['reward_list'] = reward_list
            replay_buffer.append(temp_dict)

            new_sample_reward.append(np.sum(reward_list))

        global_iteration_counter += 1
        print('this is global iteration ',global_iteration_counter)
        print('the current reward is',np.mean(new_sample_reward))
        
        #record the optimization process
        optimization_history_list.append(np.mean(new_sample_reward))
        optimization_history = {}
        optimization_history['objective_history'] = optimization_history_list
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'optimization_history.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history))

        if np.mean(new_sample_reward) > current_best_reward:
            current_best_reward = np.mean(new_sample_reward)
            #save the neural network model
            cwd = os.getcwd()
            parameter_file = 'pendulum_nn_trained_model.pt'
            cwd = os.path.join(cwd,parameter_file)
            torch.save(model.state_dict(),cwd)


        
        #we can update the model more than once because we are using off-line data
        for update_iteration in range(0,5):
            #sample experience from the replay buffer for training
            new_replay_buffer_reward = []
            for entry in replay_buffer_reward:
                new_replay_buffer_reward.append(np.log(entry*-1)*-1) #because the reward is negative here
            sample_probability = (np.exp(new_replay_buffer_reward))/np.sum(np.exp(new_replay_buffer_reward)) #apply softmax to the total_reward list
            sampled_off_line_data = []
            for sample_counter in range(0,C.training_batch_size):
                sampled_index = np.random.choice(np.arange(0, len(replay_buffer)), p=sample_probability.tolist())
                sampled_off_line_data.append(replay_buffer[sampled_index])

        
            #compute the loss and update model
            #total_loss = torch.tensor([0.0], requires_grad=True)
            total_loss = 0
            model.zero_grad()

            baseline_reward = 0
            for sample_index in range(0,len(sampled_off_line_data)):
                off_line_data = sampled_off_line_data[sample_index]
                baseline_reward += np.sum(off_line_data['reward_list'])
            baseline_reward = baseline_reward/len(sampled_off_line_data)

            for sample_index in range(0,len(sampled_off_line_data)):
                off_line_data = sampled_off_line_data[sample_index]

                actor_log_prob_mean = model(off_line_data['observation_list'])
                dist = MultivariateNormal(actor_log_prob_mean, cov_matrix)
                actor_log_prob = dist.log_prob(off_line_data['action_list'])

                #calculate the ratio for adjusting off-line data
                ratios = torch.exp(actor_log_prob - off_line_data['log_prob_action_list'])
                ratio = torch.prod(ratios)

                #vanila policy gradient loss
                #vanila_pg_loss = off_line_data['log_prob_action_list']*np.sum(off_line_data['reward_list'])
                vanila_pg_loss = torch.sum(actor_log_prob)*(np.sum(off_line_data['reward_list'])+baseline_reward)

                #compute the ppo loss
                temp_loss1 = ratio*vanila_pg_loss
                temp_loss2 = torch.clamp(ratio,1-C.ppo_clip,1+C.ppo_clip)*vanila_pg_loss
                total_loss = total_loss - torch.min(temp_loss1,temp_loss2)

            total_loss = total_loss/len(sampled_off_line_data)
            #update the model
            model.zero_grad()
            total_loss.backward()
            model_optim.step()

            # #reference here https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
            # model.zero_grad()


            # for sample_index in range(0,len(sampled_off_line_data)):
            #     off_line_data = sampled_off_line_data[sample_index]

            #     actor_log_prob_mean = model(off_line_data['observation_list'])
            #     dist = MultivariateNormal(actor_log_prob_mean, cov_matrix)
            #     actor_log_prob = dist.log_prob(off_line_data['action_list'])

            #     #calculate the ratio for adjusting off-line data
            #     ratios = (torch.exp(actor_log_prob - off_line_data['log_prob_action_list']))
            #     ratio = torch.prod(ratios)

            #     #vanila policy gradient loss
            #     #vanila_pg_loss = torch.sum(off_line_data['log_prob_action_list']*np.sum(off_line_data['reward_list']))
            #     #vanila_pg_loss = torch.sum(actor_log_prob)*np.sum(off_line_data['reward_list'])
            #     vanila_pg_loss = torch.sum(actor_log_prob*np.sum(off_line_data['reward_list']))

            #     #compute the ppo loss
            #     temp_loss1 = ratio*vanila_pg_loss
            #     temp_loss2 = torch.clamp(ratio,1-C.ppo_clip,1+C.ppo_clip)*vanila_pg_loss
            #     total_loss = -1*torch.min(temp_loss1,temp_loss2)

                

            #     total_loss.backward()

            # #total_loss = total_loss/len(sampled_off_line_data)
            # #update the model
            # model_optim.step()
            # model.zero_grad()    





