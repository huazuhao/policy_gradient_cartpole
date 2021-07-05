
import config as C
import numpy as np
import utils
import os
import jsonpickle

def sample_one_trajectory(env, normalizer, policy, direction=None, delta=None):
    state = env.reset()
    sum_rewards = 0

    while True:
        state = utils.extract_features(state)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        #reward = max(min(reward, 1), -1)
        sum_rewards += reward
        if done:
            break

    return sum_rewards


def train(env, policy, normalizer):

    optimization_history = []
    current_best_objective = float('-inf')

    for step in range(C.max_epoch):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * C.max_sample_directions
        negative_rewards = [0] * C.max_sample_directions

        # Getting the positive rewards in the positive directions
        for k in range(C.max_sample_directions):
            positive_rewards[k] = sample_one_trajectory(env, normalizer, policy, direction="positive", delta=deltas[k])

        # Getting the negative rewards in the negative/opposite directions
        for k in range(C.max_sample_directions):
            negative_rewards[k] = sample_one_trajectory(env, normalizer, policy, direction="negative", delta=deltas[k])

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
        test_reward = [0] * C.max_evaluation
        for k in range(C.max_evaluation):
            test_reward[k] = sample_one_trajectory(env, normalizer, policy)
        print('Step:', step, 'Reward:', np.mean(test_reward))


        #save the optimization trajectory
        optimization_history.append(np.mean(test_reward))
        optimization_history_dict = {}
        optimization_history_dict['history'] = optimization_history
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        file_name = 'optimization_history.json'
        cwd = os.path.join(cwd,file_name)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history_dict))

        #save the current best parameter
        if np.mean(test_reward) > current_best_objective:
            current_best_objective = np.mean(test_reward)
            learned_parameter_dict = {}
            #theta
            learned_parameter_dict['theta'] = policy.theta
            #normalizer
            learned_parameter_dict['normalizer_n'] = normalizer.n
            learned_parameter_dict['normalizer_mean'] = normalizer.mean
            learned_parameter_dict['normalizer_mean_diff'] = normalizer.mean_diff
            learned_parameter_dict['normalizer_var'] = normalizer.var
            cwd = os.getcwd()
            #cwd = os.path.join(cwd, 'data_folder')
            file_name = 'learned_parameter.json'
            cwd = os.path.join(cwd,file_name)
            with open(cwd, 'w') as statusFile:
                statusFile.write(jsonpickle.encode(learned_parameter_dict))

