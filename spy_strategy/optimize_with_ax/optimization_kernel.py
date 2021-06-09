import multiprocessing
import os
import trading_spy_ax
import jsonpickle
import config as C
import numpy as np
import utils
import datetime

from ax import *


import torch


class optimization():

    def __init__(self):
        self.max_worker = multiprocessing.cpu_count()
        if self.max_worker > 1:
            self.max_worker -= 5
        #self.max_worker = 1

        self.data_interval_minute = 5

        self.best_obj_so_far = float('-inf')
        self.mean_as_objective = float('-inf')
        self.exp = None #define an experiment for ax
        self.best_parameters_from_experiment = None
        self.global_iteration_count = 0
        self.optimization_history_objective = []
        self.optimization_history_mean = []

    
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
                

        if np.min(total_returns)>self.best_obj_so_far:
            self.best_obj_so_far = np.min(total_returns)
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
        print('the objective for ths iteration is',np.min(total_returns))
        print('the mean of total_returns is',np.mean(total_returns))
        print('the current time is',datetime.datetime.now())
        print(' ')

        self.optimization_history_mean.append(np.mean(total_returns))
        self.optimization_history_objective.append(np.min(total_returns))
        optimization_history = {}
        optimization_history['mean_history'] = self.optimization_history_mean
        optimization_history['objective_history'] = self.optimization_history_objective
        #store the history
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'optimization_history.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history))

        #optimization_objective = (np.mean(total_returns) - np.std(total_returns) * 0.5, np.std(total_returns))
        optimization_objective = (np.min(total_returns),0)

        self.global_iteration_count += 1

        return {'adj_return': optimization_objective}


    def get_one_sample_score(self,q,free_parameters):

        max_simulation_day = 100
        num_minutes_per_trading_day = (6*2+1)*30
        max_simulation_length = max_simulation_day*num_minutes_per_trading_day/self.data_interval_minute #in unit of interval
        min_history_length = 30*num_minutes_per_trading_day/self.data_interval_minute #in unit of interval, the 30 means 30 days
        #convert window length from days into number of intervals
        inner_window_length_interval = free_parameters['inner_bound_window_length']*num_minutes_per_trading_day/self.data_interval_minute
        outer_window_length_interval = free_parameters['outer_bound_window_length']*num_minutes_per_trading_day/self.data_interval_minute

        #begin to simulate trading
        local_env = trading_spy_ax.trading_spy(max_simulation_length,min_history_length)
        this_trajectory_reward = []
        has_at_least_one_sell = False
        null_objective = True
        current_obs = local_env.reset()
        begin_simulation_index = local_env.get_begin_index()
        per_position_value = current_obs['cash']/np.floor(free_parameters['max_position'])


        for time_index in range(int(begin_simulation_index),int(begin_simulation_index+max_simulation_length)):

            action = None
            buy_dollar_value = 0

            inner_lower_bound, inner_upper_bound = utils.generate_bounds(
                current_obs['middle_price_history'],
                int(inner_window_length_interval),
                free_parameters['inner_denoise_std'],
                free_parameters['buy_1_threshold'],
                free_parameters['sell_1_threshold']
            )

            outer_lower_bound, outer_upper_bound = utils.generate_bounds(
                current_obs['middle_price_history'],
                int(outer_window_length_interval),
                free_parameters['outer_denoise_std'],
                free_parameters['buy_2_threshold'],
                free_parameters['sell_2_threshold']
            )

            #see if we can buy
            #we pay ask price when buying
            if current_obs['ask_price'] < inner_lower_bound and current_obs['ask_price'] > outer_lower_bound:
                time_since_last_buy = (time_index-current_obs['last_buy_1_time'])*\
                    self.data_interval_minute #in unit of minutes
                if time_since_last_buy > free_parameters['buy_1_wait']:
                    if current_obs['cash'] >= per_position_value:
                        action = "buy_1"
                        buy_dollar_value = per_position_value

            if current_obs['ask_price'] < inner_lower_bound and current_obs['ask_price'] <  outer_lower_bound:
                time_since_last_buy = (time_index-current_obs['last_buy_2_time'])*\
                    self.data_interval_minute #in unit of minutes
                if time_since_last_buy > free_parameters['buy_2_wait']:
                    if current_obs['cash'] >= per_position_value:
                        action = "buy_2"
                        buy_dollar_value = per_position_value

            
            #see if we can sell
            #we get to sell at the bid price
            if current_obs['first_position'] != None:
                first_position_price = current_obs['first_position']['price']

                if current_obs['bid_price'] > inner_upper_bound and current_obs['bid_price'] < outer_upper_bound:
                    time_since_last_sell = (time_index-current_obs['last_sell_1_time'])*\
                        self.data_interval_minute #in unit of minutes
                    if time_since_last_sell > free_parameters['sell_1_wait']:
                        if current_obs['bid_price'] > first_position_price*(1+free_parameters['min_profit']/100):
                            action = "sell_1"

                if current_obs['bid_price'] > inner_upper_bound and current_obs['bid_price'] > outer_upper_bound:
                    time_since_last_sell = (time_index-current_obs['last_sell_2_time'])*\
                        self.data_interval_minute #in unit of minutes
                    if time_since_last_sell > free_parameters['sell_2_wait']:
                        if current_obs['bid_price'] > first_position_price*(1+free_parameters['min_profit']/100):
                            action = "sell_2"


            #apply action to the environment
            current_obs,reward,has_at_least_one_sell = local_env.step(action,buy_dollar_value)

            if has_at_least_one_sell and null_objective:
                null_objective = False
            
            #record the reward
            this_trajectory_reward.append(reward)

        #final time step, get the long term reward
        reward = local_env.final()
        this_trajectory_reward.append(reward)

        if null_objective:
            objective = -1e9
        else:
            objective = np.sum(this_trajectory_reward)

        q.put([objective])


    def init_search_space(self):
        search_space = SearchSpace(
        parameters=[
            RangeParameter(name="inner_denoise_std", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="outer_denoise_std", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="buy_1_threshold", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="buy_2_threshold", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="sell_1_threshold", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="sell_2_threshold", parameter_type=ParameterType.FLOAT, lower=0.01, upper=4),
            RangeParameter(name="inner_bound_window_length", parameter_type=ParameterType.FLOAT, lower=1, upper=30), #in unit of days
            RangeParameter(name="outer_bound_window_length", parameter_type=ParameterType.FLOAT, lower=1, upper=30),
            RangeParameter(name="buy_1_wait", parameter_type=ParameterType.FLOAT, lower=1, upper=1e4), #in unit of minutes
            RangeParameter(name="buy_2_wait", parameter_type=ParameterType.FLOAT, lower=1, upper=1e4),
            RangeParameter(name="sell_1_wait", parameter_type=ParameterType.FLOAT, lower=1, upper=1e4),
            RangeParameter(name="sell_2_wait", parameter_type=ParameterType.FLOAT, lower=1, upper=1e4),
            RangeParameter(name="max_position", parameter_type=ParameterType.INT, lower=1, upper=10),
            RangeParameter(name="min_profit", parameter_type=ParameterType.FLOAT, lower=0.01, upper=10), #in unit of percentage
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