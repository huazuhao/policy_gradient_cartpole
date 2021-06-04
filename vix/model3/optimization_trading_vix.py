import pandas as pd
import numpy as np
import scipy as sp
import jsonpickle
import random
import datetime
import multiprocessing
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
import trading_vix
from scipy.stats import expon
from scipy.special import expit
from copulas.multivariate import Multivariate
import utils

from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    SearchSpace, 
    SimpleExperiment, 
    OutcomeConstraint, 
)

from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.factory import Models
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

import torch
import config as C


class optimization():

    def __init__(self):
        self.max_worker = multiprocessing.cpu_count()
        if self.max_worker > 1:
            self.max_worker -= 5
        #self.max_worker = 1
        
        self.best_obj_so_far = float('-inf')
        self.exp = None #define an experiment for ax
        self.best_parameters_from_experiment = None
        self.global_iteration_count = 0

    
    def evaluate_parameter(self,parameter, weight=None):
        total_returns = []
        
        q = multiprocessing.Queue(maxsize = self.max_worker)

        for iteration_index in range(0,C.ROUNDS_OF_WORKER):
            p_list = []
            for worker in range(0,self.max_worker):
                try:
                    p = multiprocessing.Process(target = self.get_one_sample_score,\
                                                args = (q,parameter))
                    p.start()
                    p_list.append(p)
                except:
                    pass

            for j in range(len(p_list)):
                res = q.get()
                total_returns.append(res[0])
                
        self.global_iteration_count += 1

        if np.mean(total_returns)>self.best_obj_so_far:
            self.best_obj_so_far = np.mean(total_returns)
            self.best_parameters_from_experiment = parameter
            print('the best objective so far is',self.best_obj_so_far)
            print('the associated best parameters are',parameter)
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

        optimization_objective = (np.mean(total_returns) - np.std(total_returns) * 0.5, np.std(total_returns))
        

        return {'adj_return': optimization_objective}


    def get_one_sample_score(self,q,free_parameters):

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
        locol_env = trading_vix.trading_vix()
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


    def init_search_space(self):
        search_space = SearchSpace(
        parameters=[
            RangeParameter(name="cov_para_1", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_2", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_3", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_4", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_5", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_6", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_7", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_8", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_9", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_10", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),

            RangeParameter(name="beta_1a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_1b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_2a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_2b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_3a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_3b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_4a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_4b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_5a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_5b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),

            RangeParameter(name="lambda_expon_1", parameter_type=ParameterType.FLOAT, lower=0.001, upper=0.1),
            RangeParameter(name="lambda_expon_2", parameter_type=ParameterType.FLOAT, lower=0.001, upper=0.1),
            RangeParameter(name="lambda_expon_3", parameter_type=ParameterType.FLOAT, lower=0.001, upper=0.1),
            RangeParameter(name="lambda_expon_4", parameter_type=ParameterType.FLOAT, lower=0.001, upper=0.1),
            ]
        )

        self.exp = SimpleExperiment(
        name="0",
        search_space=search_space,
        evaluation_function=self.evaluate_parameter,
        objective_name="adj_return"
        )


    def initialization_trials(self):
        sobol = Models.SOBOL(self.exp.search_space)
        for i in range(C.INITIALIZATION_TRIALS):
            self.exp.new_trial(generator_run=sobol.gen(1))


    def optimization_trials(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print(device)
        for i in range(C.OPTIMIZATION_TRIALS):
            print('Running GP+EI optimization trial', i+1)
            # Reinitialize GP+EI model at each step with updated data.
            #gpei = Models.GPEI(experiment=self.exp, data=self.exp.eval(), dtype=torch.float32, device=device)
            gpei = Models.GPEI(experiment=self.exp, data=self.exp.eval())
            batch = self.exp.new_trial(generator_run=gpei.gen(1))

    
    def get_best_parameters(self):
        _, trial = list(self.exp.trials.items())[-1]
        print(trial)
        trial.generator_run
        gr = trial.generator_run
        best_arm, best_arm_predictions = gr.best_arm_predictions
        best_parameters = best_arm.parameters
        self.best_parameters = best_parameters
        print('the best parameters are',best_parameters)


    def store_parameters(self):
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'best_parameter_from_multi_armed_bandit.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(self.best_parameters))

        cwd = os.getcwd()
        parameter_file = 'best_parameter_from_direct_experiment.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(self.best_parameters_from_experiment))


    
    def run_optimization(self):


        self.init_search_space()

        print('Begin initialization trials')

        self.initialization_trials()

        self.optimization_trials()
        self.get_best_parameters()
        self.store_parameters()