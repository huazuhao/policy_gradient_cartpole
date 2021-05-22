import pickle
import numpy as np
import utils


with open('test_info.pkl', 'rb') as f:
    tests_info = pickle.load(f)
test_cases = sorted(tests_info.keys())


""" ------------- testing action distribution computation ----------------"""
print('-'*10 + ' testing compute_action_distribution ' + '-'*10)
for i in test_cases:
    theta = tests_info[i]['theta']
    phis = tests_info[i]['phis']
    soln_action_dist = tests_info[i]['action_dst']
    action_dist = utils.compute_action_distribution(theta, phis)

    err = np.linalg.norm(soln_action_dist - action_dist)
    print('test {} for compute_action_distribution - error = {}'.format(i, err))

""" ------------- testing compute_log_softmax_grad ----------------"""
print('-' * 10 + ' testing compute_log_softmax_grad ' + '-' * 10)
for i in test_cases:
    theta = tests_info[i]['theta']
    phis = tests_info[i]['phis']
    action = tests_info[i]['action']
    soln_grad = tests_info[i]['grad']
    grad = utils.compute_log_softmax_grad(theta, phis, action)
    err = np.linalg.norm(soln_grad - grad)
    print('test {} for compute_log_softmax_grad - error = {}'.format(i, err))

""" ------------- testing compute_fisher_matrix ----------------"""
print('-' * 10 + ' testing compute_fisher_matrix ' + '-' * 10)
for i in test_cases:
    total_grads = tests_info[i]['total_grads']
    total_rewards = tests_info[i]['total_rewards']

    soln_fisher = tests_info[i]['fisher']
    fisher = utils.compute_fisher_matrix(total_grads)

    err = np.linalg.norm(soln_fisher - fisher)
    print('test {} for compute_fisher_matrix - error = {}'.format(i, err))

""" ------------- testing compute_value_gradient ----------------"""
print('-' * 10 + ' testing compute_value_gradient ' + '-' * 10)
for i in test_cases:
    total_grads = tests_info[i]['total_grads']
    total_rewards = tests_info[i]['total_rewards']

    soln_v_grad = tests_info[i]['v_grad']
    #print('the solution grad is',soln_v_grad.tolist())
    v_grad = utils.compute_value_gradient(total_grads, total_rewards)
    #print('the computed grad is',v_grad.tolist())

    err = np.linalg.norm(soln_v_grad - v_grad)
    print('test {} for compute_value_gradient - error = {}'.format(i, err))

""" ------------- testing compute_value_gradient ----------------"""
print('-' * 10 + ' testing compute_value_gradient ' + '-' * 10)
for i in test_cases:

    fisher = tests_info[i]['fisher']
    delta = 1e-2
    v_grad = tests_info[i]['v_grad']
    soln_eta = tests_info[i]['eta']

    eta = utils.compute_eta(delta, fisher, v_grad)

    err = np.linalg.norm(soln_eta - eta)
    print('test {} for compute_eta - error = {}'.format(i, err))


