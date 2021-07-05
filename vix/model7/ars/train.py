
import config as C
import numpy as np
import utils
import os
import jsonpickle
import multiprocessing


def sample_one_trajectory(q,env, normalizer, normalizer_data, policy, direction=None, delta=None):
    state = env.reset()
    sum_rewards = 0

    normalizer.n = normalizer_data['n']
    normalizer.mean = normalizer_data['mean']
    normalizer.mean_diff = normalizer_data['mean_diff']
    normalizer.var = normalizer_data['var']

    while True:
        state = utils.extract_features(state)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward
        if done:
            break
    
    new_normalizer_data = {}
    new_normalizer_data['n'] = normalizer.n
    new_normalizer_data['mean'] = normalizer.mean
    new_normalizer_data['mean_diff'] = normalizer.mean_diff
    new_normalizer_data['var'] = normalizer.var

    q.put([sum_rewards,new_normalizer_data])


def sample_trajectories(env,normalizer,normalizer_data,policy,direction=None,deltas=None):

    total_rewards = []

    #print('multiprocessing and',normalizer_data['n'])

    q = multiprocessing.Queue(maxsize = C.max_worker)
    counter = 0

    for iteration_index in range(0,int(C.max_sample_directions/C.max_worker+1)):
            p_list = []
            for worker in range(0,C.max_worker):
                try:
                    if counter < C.max_sample_directions:
                        if deltas != None:
                            delta = deltas[counter]
                        else:
                            delta = None
                        p = multiprocessing.Process(target = sample_one_trajectory,\
                                                    args = (q,env,normalizer,normalizer_data,policy,direction,delta))
                        p.start()
                        p_list.append(p)
                        counter += 1
                except:
                    pass

            for j in range(len(p_list)):
                res = q.get()
                total_rewards.append(res[0])
                new_normalizer_data = res[1]
    
    return total_rewards, new_normalizer_data




def train(env, policy, normalizer):

    optimization_history = []
    current_best_objective = float('-inf')
    new_normalizer_data = {}
    new_normalizer_data['n'] = normalizer.n
    new_normalizer_data['mean'] = normalizer.mean
    new_normalizer_data['mean_diff'] = normalizer.mean_diff
    new_normalizer_data['var'] = normalizer.var

    for step in range(C.max_epoch):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()

        # Getting the positive rewards in the positive directions
        positive_rewards, new_normalizer_data = sample_trajectories(env,normalizer,new_normalizer_data,policy,direction='positive',deltas=deltas)

        # Getting the negative rewards in the negative/opposite directions
        negative_rewards, new_normalizer_data = sample_trajectories(env,normalizer,new_normalizer_data,policy,direction='negative',deltas=deltas)

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:C.max_best_sample_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        test_rewards, new_normalizer_data = sample_trajectories(env,normalizer,new_normalizer_data,policy)
        print('Step:', step, 'Reward:', np.mean(test_rewards))

        # render policy for demonstration:
        # if step % C.render_gym_env_frequency ==0:
        #     state = env.reset()
        #     sum_rewards = 0
        #     while True:
        #         env.render()
        #         state = utils.extract_features(state)
        #         #normalizer.observe(state)
        #         #state = normalizer.normalize(state)
        #         delta = None
        #         direction = None
        #         action = policy.evaluate(state, delta, direction)
        #         state, reward, done, _ = env.step(action)
        #         #reward = max(min(reward, 1), -1)
        #         sum_rewards += reward
        #         if done:
        #             env.close()
        #             break
        #     print('the learned theta is',policy.theta)

        #save the optimization trajectory
        optimization_history.append(np.mean(test_rewards))
        optimization_history_dict = {}
        optimization_history_dict['history'] = optimization_history
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        file_name = 'optimization_history.json'
        cwd = os.path.join(cwd,file_name)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history_dict))

        #save the current best parameter
        if np.mean(test_rewards) > current_best_objective:
            current_best_objective = np.mean(test_rewards)
            learned_parameter_dict = {}
            #theta
            learned_parameter_dict['theta'] = policy.theta
            #normalizer
            learned_parameter_dict['normalizer_n'] = new_normalizer_data['n']
            learned_parameter_dict['normalizer_mean'] = new_normalizer_data['mean']
            learned_parameter_dict['normalizer_mean_diff'] = new_normalizer_data['mean_diff']
            learned_parameter_dict['normalizer_var'] = new_normalizer_data['var']
            cwd = os.getcwd()
            #cwd = os.path.join(cwd, 'data_folder')
            file_name = 'learned_parameter.json'
            cwd = os.path.join(cwd,file_name)
            with open(cwd, 'w') as statusFile:
                statusFile.write(jsonpickle.encode(learned_parameter_dict))

