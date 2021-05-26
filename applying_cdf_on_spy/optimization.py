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
from scipy.special import expit
import simple_continuous_buy_sell_spy

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
        self.env = None #the stock trading environment
        self.best_parameters_from_experiment = None
        self.global_iteration_count = 0

    def load_env(self):
        self.env = simple_continuous_buy_sell_spy.simple_continuous_buy_sell_spy()
    
    def evaluate_parameter(self,parameter, weight=None):
        total_returns = []
        
        q = multiprocessing.Queue(maxsize = self.max_worker)

        for iteration_index in range(0,int(C.N_SAMPLES)):
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

        print('this is iteration',self.global_iteration_count)
        print('the objective for ths iteration is',np.mean(total_returns))
        print('the current time is',datetime.datetime.now())
        print(' ')

        optimization_objective = (np.mean(total_returns) - np.std(total_returns) * 0.5, np.std(total_returns))
        

        return {'adj_return': optimization_objective}

    
    def get_one_sample_score(self,q,free_parameters):
        #construct the covariance matrix of the gaussian copula 
        #ensure the matrix is symmetric positive definite
        free_para1 = free_parameters['cov_para_1']
        free_para2 = free_parameters['cov_para_2']
        free_para3 = free_parameters['cov_para_3']

        diag_1 = 1
        diag_2 = np.sqrt(1-free_para1**2)
        if (1-free_para2**2-free_para3**2)<0:
            diag_3 = np.sqrt(np.abs(1-free_para2**2-free_para3**2))
        else:
            diag_3 = np.sqrt(1-free_para2**2-free_para3**2)

        lower_triangular_matrix = np.asarray([[diag_1,0,0],
                                [free_para1,diag_2,0],
                                [free_para2,free_para3,diag_3]])


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
        'type': 'copulas.univariate.beta.BetaUnivariate'}]


        #now, we construct the gaussian copula
        copula_parameters = {}
        copula_parameters['covariance'] = cov_matrix
        copula_parameters['univariates'] = univerates
        copula_parameters['type'] = 'copulas.multivariate.gaussian.GaussianMultivariate'
        copula_parameters['columns'] = [0,1,2]

        new_dist = Multivariate.from_dict(copula_parameters)

        #now, we begin to simulate trading
        #first, initialize the observation
        this_trajectory_reward = []
        current_feature = self.env.reset()


        for time_index in range(0,200):
        
            #compute an action given current observation
            current_feature = np.transpose(current_feature)
            logistic_transform = expit(current_feature)
            holding_position = np.reshape(current_feature[:,-1],(1,1))
            data_point_for_df = np.concatenate((logistic_transform[:,:-1],holding_position),axis = 1)
            assert data_point_for_df.shape[1] == 3
            data_point_for_copula = pd.DataFrame(data_point_for_df)
            action = new_dist.cdf(data_point_for_copula)

            #apply the action to the environment
            current_feature, reward = self.env.step(action)

            #record reward
            this_trajectory_reward.append(reward)

        #final time step
        reward = self.env.final()
        this_trajectory_reward.append(reward)

        q.put([np.sum(this_trajectory_reward)])


    def init_search_space(self):
        search_space = SearchSpace(
        parameters=[
            RangeParameter(name="cov_para_1", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_2", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),
            RangeParameter(name="cov_para_3", parameter_type=ParameterType.FLOAT, lower=-0.9, upper=0.9),

            RangeParameter(name="beta_1a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_1b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_2a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_2b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_3a", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
            RangeParameter(name="beta_3b", parameter_type=ParameterType.FLOAT, lower=0, upper=20),
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


        self.load_env()
        self.init_search_space()

        print('Begin initialization trials')

        self.initialization_trials()

        self.optimization_trials()
        self.get_best_parameters()
        self.store_parameters()