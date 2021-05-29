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
import trading_vix_objective_2
from scipy.stats import expon
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
from copulas.multivariate import Multivariate

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

        #the size of each gaussian copula cannot be too large
        #otherwise, the CDF will almost always be 0. 
        #the max dimension of the copula is 3
        #so, we will use a hierarchical structure of gaussian copulas
        #with each of having dimension of less than 3. 

        cov_input = [free_parameters['cov_para_1'],free_parameters['cov_para_2'],free_parameters['cov_para_3']]
        marginal_input = [free_parameters['beta_1a'],free_parameters['beta_1b'],
                 free_parameters['beta_2a'],free_parameters['beta_2b'],
                 free_parameters['beta_3a'],free_parameters['beta_3b']]
        dist_1 = utils.construct_a_copula(cov_input,marginal_input)

        cov_input = [free_parameters['cov_para_4']]
        marginal_input = [free_parameters['beta_4a'],free_parameters['beta_4b'],
                 free_parameters['beta_5a'],free_parameters['beta_5b']]
        dist_2 = utils.construct_a_copula(cov_input,marginal_input)
        


        #other parameters needed for transforming the features
        lambda_expon_1 = free_parameters['lambda_expon_1']
        lambda_expon_2 = free_parameters['lambda_expon_2']
        lambda_expon_3 = free_parameters['lambda_expon_3']
        lambda_expons = [lambda_expon_1,lambda_expon_2,lambda_expon_3]


        #now, we begin to simulate trading
        #first, initialize the observation
        locol_env = trading_vix_objective_2.trading_vix()
        this_trajectory_reward = []
        current_feature = locol_env.reset()

        for time_index in range(0,200):
        
            #compute an action given current observation
            transformed_features = []
            for feature_index in range(len(lambda_expons)):
                transformation = expon.cdf(current_feature[feature_index,0],scale = 1.0/lambda_expons[feature_index])
                transformed_features.append(transformation)
            transformed_features = np.asarray(transformed_features)
            transformed_features = np.reshape(transformed_features,(1,-1))
            data_point_for_copula_1 = pd.DataFrame(transformed_features)
            copula_1_output = dist_1.cdf(data_point_for_copula_1)

            data_point_for_copula_2 = np.asarray([copula_1_output,current_feature[-1,:][0]])
            data_point_for_copula_2 = np.reshape(data_point_for_copula_2,(1,-1))
            data_point_for_copula_2 = pd.DataFrame(data_point_for_copula_2)
            copula_2_output = dist_2.cdf(data_point_for_copula_2)

            #print('action is',copula_1_output)

            #apply the action to the environment
            current_feature, reward = locol_env.step(copula_1_output)

            #record reward
            this_trajectory_reward.append(reward)

        #final time step,get the long term reward
        reward = locol_env.final()
        this_trajectory_reward.append(reward)

        #print('the sum of the reward is',np.sum(this_trajectory_reward))
        objective = np.sum(this_trajectory_reward)
        if objective == 0:
            objective = -1e9

        q.put([objective])


    def init_search_space(self):
        search_space = SearchSpace(
        parameters=[
            RangeParameter(name="cov_para_1", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_2", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_3", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_4", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            # RangeParameter(name="cov_para_5", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            # RangeParameter(name="cov_para_6", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),

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
        parameter_file = 'index_copula_parameter.json'
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