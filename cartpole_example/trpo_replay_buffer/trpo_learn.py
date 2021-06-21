import config as C
import roll_out_once
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import os
import utils
import jsonpickle

def trpo_learn(replay_buffer,replay_buffer_reward,env,model,cov_matrix):

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

        
            #concatenate the sampled experience into one long experience
            total_sampled_observation = torch.empty(size=(0,))
            total_sampled_action = torch.empty(size=(0,))
            total_sampled_log_prob_action_state = torch.empty(size=(0,))
            total_sampled_reward = np.zeros((0))

            baseline_reward = 0
            for sample_index in range(0,len(sampled_off_line_data)):
                off_line_data = sampled_off_line_data[sample_index]
                baseline_reward += np.sum(off_line_data['reward_list'])
            baseline_reward = baseline_reward/len(sampled_off_line_data)

            for sample_index in range(0,len(sampled_off_line_data)):
                off_line_data = sampled_off_line_data[sample_index]
                total_sampled_observation = torch.cat((total_sampled_observation,off_line_data['observation_list']),dim = 0)
                total_sampled_action = torch.cat((total_sampled_action,off_line_data['action_list']),dim = 0)
                total_sampled_log_prob_action_state = torch.cat((total_sampled_log_prob_action_state,off_line_data['log_prob_action_list']),dim = 0)
                total_sampled_reward  = np.concatenate((total_sampled_reward,np.asarray(off_line_data['reward_list'])-baseline_reward))
            total_sampled_reward = torch.tensor(total_sampled_reward)

            #compute loss and update model with trpo

            #this get_loss function will also be used for line search later on
            get_loss = lambda x: getSurrogateloss(model,
                                                total_sampled_observation,
                                                total_sampled_action,
                                                cov_matrix,
                                                total_sampled_log_prob_action_state,
                                                total_sampled_reward)
            loss = get_loss(model)

            #print('the loss of the model is',loss)

            grads = torch.autograd.grad(loss, model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads])
            
            #compute the direction for updating the model parameter
            Fvp = lambda v: FisherVectorProduct(v,
                                        model,
                                        total_sampled_observation,
                                        total_sampled_action,
                                        total_sampled_log_prob_action_state,
                                        C.damping,
                                        cov_matrix)

            stepdir = conjugate_gradients(Fvp, -loss_grad, 20)

            #print('the step direction is',stepdir)

            #now, I need to perform a line search for knowing how large a step I can take in the 
            #direction computed previously
            shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

            if shs > 0: #this should be positive 

                #print('shs is',shs)
                lm = torch.sqrt(shs / C.max_kl)
                #print('lm is',lm)
                fullstep = stepdir / lm[0]
                #print('fullstep is',fullstep)
                neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
                prev_params = utils.get_flat_params_from(model)
                success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                                neggdotstepdir / lm[0])

                #print('new params is',new_params)

                model = utils.set_flat_params_to(model, new_params)




#helper function for computing trpo

def getSurrogateloss(model,observations,actions,cov_matrix,logProbabilityOld,total_reward):

    actor_log_prob_mean = model(observations)
    dist = MultivariateNormal(actor_log_prob_mean, cov_matrix)
    actor_log_prob = dist.log_prob(actions)

    action_loss = total_reward*torch.exp(actor_log_prob - logProbabilityOld)

    return action_loss.mean()*-1

def FisherVectorProduct(v , model, states, actions,logProbabilityOld,damping, cov_matrix):

    #we want to be able to compute the direction of the update by inv(FisherMatrix)*Grad_of_Network
    #however, computing the inverse is too hard
    #instead, we want to solve FisherMatrix*x = Grad_of_Network
    #to solve the above equation, we can use conjugate gradient
    #this FisherVectorProduct helper function is needed for the conjugate gradient method
    #the FisherMatrix is the second derivative of the KL between the new policy distribution and the old policy distribution
    #for a detailed explanation of this part
    #please visit https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/

    kl = model.meanKlDivergence(states, actions, logProbabilityOld, cov_matrix)

    grads = torch.autograd.grad(kl, model.parameters()
                    ,retain_graph=True, create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                    for grad in grads]).data

    return flat_grad_grad_kl + v * damping


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        t= i
        if rdotr < residual_tol:
            break
    return x

def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.5):
    fval = f(model).data
    #print("\tfval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        utils.set_flat_params_to(model, xnew)
        newfval = f(model).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        #print("\ta : %6.4e /e : %6.4e /r : %6.4e "%(actual_improve.item(), expected_improve.item(), ratio.item()))

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            #print("\tfval after", newfval.item())
            #print("\tlog(std): %f"%xnew[0])
            print('update model')
            return True, xnew
    return False, x


    

#the reference below is for computing minibatch loss
#reference here https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20