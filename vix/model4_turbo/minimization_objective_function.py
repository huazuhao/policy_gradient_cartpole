import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import expon
from copulas.multivariate import Multivariate
import jsonpickle
import datetime
import os

import trading_vix_non_random_seed
import config as C



class minimization_objective_function():

    def __init__(self):

        self.max_worker = multiprocessing.cpu_count()
        if self.max_worker > 1:
            self.max_worker -= 5
        #self.max_worker = 2

        self.best_obj_so_far = float('-inf')
        self.best_parameters_from_experiment = None
        self.global_iteration_count = 0

        self.init_search_space_dict = {
            'cov_para_1':[-0.9,0.9],
            'cov_para_2':[-0.9,0.9],
            'cov_para_3':[-0.9,0.9],
            'cov_para_4':[-0.9,0.9],
            'cov_para_5':[-0.9,0.9],
            'cov_para_6':[-0.9,0.9],
            'cov_para_7':[-0.9,0.9],
            'cov_para_8':[-0.9,0.9],
            'cov_para_9':[-0.9,0.9],
            'cov_para_10':[-0.9,0.9],

            'beta_1a':[0,20],
            'beta_1b':[0,20],
            'beta_2a':[0,20],
            'beta_2b':[0,20],
            'beta_3a':[0,20],
            'beta_3b':[0,20],
            'beta_4a':[0,20],
            'beta_4b':[0,20],
            'beta_5a':[0,20],
            'beta_5b':[0,20],

            'lambda_expon_1':[0.001,10],
            'lambda_expon_2':[0.001,0.1],
            'lambda_expon_3':[0.001,0.1],
            'lambda_expon_4':[0.001,0.1],
        }
        
        self.var_name_list = []
        self.var_name_list.append('cov_para_1')
        self.var_name_list.append('cov_para_2')
        self.var_name_list.append('cov_para_3')
        self.var_name_list.append('cov_para_4')
        self.var_name_list.append('cov_para_5')
        self.var_name_list.append('cov_para_6')
        self.var_name_list.append('cov_para_7')
        self.var_name_list.append('cov_para_8')
        self.var_name_list.append('cov_para_9')
        self.var_name_list.append('cov_para_10')
        self.var_name_list.append('beta_1a')
        self.var_name_list.append('beta_1b')
        self.var_name_list.append('beta_2a')
        self.var_name_list.append('beta_2b')
        self.var_name_list.append('beta_3a')
        self.var_name_list.append('beta_3b')
        self.var_name_list.append('beta_4a')
        self.var_name_list.append('beta_4b')
        self.var_name_list.append('beta_5a')
        self.var_name_list.append('beta_5b')
        self.var_name_list.append('lambda_expon_1')
        self.var_name_list.append('lambda_expon_2')
        self.var_name_list.append('lambda_expon_3')
        self.var_name_list.append('lambda_expon_4')


        self.dim = len(self.init_search_space_dict)
        lower_bound_list = []
        upper_bound_list = []
        for var_name in self.var_name_list:
            [lower_bound,upper_bound] = self.init_search_space_dict[var_name]
            lower_bound_list.append(lower_bound)
            upper_bound_list.append(upper_bound)
        
        self.lb = np.asarray(lower_bound_list)
        self.ub = np.asarray(upper_bound_list)


    def get_one_sample_score(self,q,free_parameters,seed_index):


        free_para1 = free_parameters['cov_para_1']
        free_para2 = free_parameters['cov_para_2']
        free_para3 = free_parameters['cov_para_3']
        free_para4 = free_parameters['cov_para_4']
        free_para5 = free_parameters['cov_para_5']
        free_para6 = free_parameters['cov_para_6']
        free_para7 = free_parameters['cov_para_7']
        free_para8 = free_parameters['cov_para_8']
        free_para9 = free_parameters['cov_para_9']
        free_para10 = free_parameters['cov_para_10']

        diag_1 = 1
        diag_2 = np.sqrt(1-free_para1**2)
        if (1-free_para2**2-free_para3**2)<0:
            diag_3 = np.sqrt(np.abs(1-free_para2**2-free_para3**2))
        else:
            diag_3 = np.sqrt(1-free_para2**2-free_para3**2)

        if (1-free_para4**2-free_para5**2-free_para6**2)<0:
            diag_4 = np.sqrt(np.abs(1-free_para4**2-free_para5**2-free_para6**2))
        else:
            diag_4 = np.sqrt(1-free_para4**2-free_para5**2-free_para6**2)
        
        if (1-free_para7**2-free_para8**2-free_para9**2-free_para10**2)<0:
            diag_5 = np.sqrt(np.abs(1-free_para7**2-free_para8**2-free_para9**2-free_para10**2))
        else:
            diag_5 = np.sqrt(1-free_para7**2-free_para8**2-free_para9**2-free_para10**2)


        lower_triangular_matrix = np.asarray([[diag_1,0,0,0,0],
                                [free_para1,diag_2,0,0,0],
                                [free_para2,free_para3,diag_3,0,0],
                                [free_para4,free_para5,free_para6,diag_4,0],
                                [free_para7,free_para8,free_para9,free_para10,diag_5]])


        cov_matrix = lower_triangular_matrix@lower_triangular_matrix.transpose()


        #now, define the marginal distribution of the gaussian copula
        univerates = [{'loc': 0,
        'scale': 1,
        'a': free_parameters['beta_1a'],
        'b': free_parameters['beta_1b'],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': free_parameters['beta_2a'],
        'b': free_parameters['beta_2b'],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': free_parameters['beta_3a'],
        'b': free_parameters['beta_3b'],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': free_parameters['beta_4a'],
        'b': free_parameters['beta_4b'],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': free_parameters['beta_5a'],
        'b': free_parameters['beta_5b'],
        'type': 'copulas.univariate.beta.BetaUnivariate'}]


        #now, we construct the gaussian copula
        copula_parameters = {}
        copula_parameters['covariance'] = cov_matrix
        copula_parameters['univariates'] = univerates
        copula_parameters['type'] = 'copulas.multivariate.gaussian.GaussianMultivariate'
        copula_parameters['columns'] = [0,1,2,3,4]

        new_dist = Multivariate.from_dict(copula_parameters)

        #other parameters needed for transforming the features
        lambda_expon_1 = free_parameters['lambda_expon_1']
        lambda_expon_2 = free_parameters['lambda_expon_2']
        lambda_expon_3 = free_parameters['lambda_expon_3']
        lambda_expon_4 = free_parameters['lambda_expon_4']
        lambda_expons = [lambda_expon_1,lambda_expon_2,lambda_expon_3,lambda_expon_4]


        #now, we begin to simulate trading
        #first, initialize the observation
        locol_env = trading_vix_non_random_seed.trading_vix(seed_index)
        this_trajectory_reward = []
        has_at_least_sell = False
        null_objective = True
        current_feature = locol_env.reset()

        for time_index in range(0,200):
        
            #compute an action given current observation
            transformed_features = []
            for feature_index in range(len(lambda_expons)):
                transformation = expon.cdf(current_feature[feature_index,0],scale = 1.0/lambda_expons[feature_index])
                min_transformation = 0.1
                transformation = min_transformation*np.exp(np.log(1.0/min_transformation)*transformation)
                transformed_features.append(transformation)
            transformed_features = np.asarray(transformed_features)
            transformed_features = np.reshape(transformed_features,(1,-1))
            #holding_position = expit(current_feature[-1,:][0])
            holding_position = current_feature[-1,:][0]
            if holding_position<0:
                print('holding is less than 0, there is some problem and the holding position is',holding_position)
            if holding_position>1:
                print('holding is greater than 1, there is some problem and the holding position is',holding_position)
            min_transformed_holding = 0.1
            transformed_holding = min_transformed_holding*np.exp(np.log(1.0/min_transformed_holding)*holding_position)
            transformed_holding = np.reshape(transformed_holding,(1,1))
            data_point_for_df = np.concatenate((transformed_features,transformed_holding),axis = 1)

            assert data_point_for_df.shape[1] == 5
            data_point_for_copula = pd.DataFrame(data_point_for_df)
            action = new_dist.cdf(data_point_for_copula)

            #print('action in optimization trading vix is',action)

            #apply the action to the environment
            current_feature, reward, has_at_least_sell = locol_env.step(action)

            if has_at_least_sell and null_objective:
                #print('sold at least once')
                null_objective = False

            #record reward
            this_trajectory_reward.append(reward)

        #final time step,get the long term reward
        reward = locol_env.final()
        this_trajectory_reward.append(reward)

        if null_objective:
            objective = -1e9
        else:
            objective = np.sum(this_trajectory_reward)

        q.put([objective])



    def __call__(self,x):

        free_parameters = {}
        
        for index in range(self.dim):
            var_name = self.var_name_list[index]
            var_value = x[index]
            free_parameters[var_name] = var_value

       
        total_returns = []
        
        q = multiprocessing.Queue(maxsize = self.max_worker)

        for iteration_index in range(0,int(C.ROUNDS_OF_WORKER)):
            p_list = []
            for worker in range(0,self.max_worker):
                try:
                    seed_index = iteration_index*self.max_worker+worker
                    p = multiprocessing.Process(target = self.get_one_sample_score,\
                                                args = (q,free_parameters,seed_index))
                    p.start()
                    p_list.append(p)
                except:
                    pass

            for j in range(len(p_list)):
                res = q.get()
                total_returns.append(res[0])

        if np.mean(total_returns)>self.best_obj_so_far:
            self.best_obj_so_far = np.mean(total_returns)
            self.best_parameters_from_experiment = free_parameters
            print('the best objective so far is',self.best_obj_so_far)
            print('the associated best parameters are',free_parameters)
            print('the current time is',datetime.datetime.now())
            cwd = os.getcwd()
            parameter_file = 'best_parameter_from_direct_experiment.json'
            cwd = os.path.join(cwd,parameter_file)
            with open(cwd, 'w') as statusFile:
                statusFile.write(jsonpickle.encode(self.best_parameters_from_experiment))

        print('this is global iteration',self.global_iteration_count)
        print('the objective for ths iteration is',np.mean(total_returns))
        print('the current time is',datetime.datetime.now())
        print(' ')

        self.global_iteration_count += 1

        optimization_objective = np.quantile(total_returns,0.25)
        #since the turbo package assumes a minimization procedure
        #I need to minimize the negative of the true objective
        optimization_objective = optimization_objective*-1

        return optimization_objective