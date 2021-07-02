import multiprocessing
import numpy as np
import trading_vix_and_spy
import jsonpickle
import linear_trpo_utils
import trading_vix_and_spy_utils
import datetime
import os
import pandas as pd

class turbo_objective_function():

    def __init__(self):

        self.max_worker = multiprocessing.cpu_count()
        if self.max_worker > 1:
            self.max_worker -= 5
        self.max_worker = 25
        self.rounds_of_workers = 2

        self.num_data_interval_per_day = 1

        self.best_obj_so_far = float('-inf')
        self.mean_as_objective = float('-inf')
        self.best_parameters_from_experiment = None
        self.global_iteration_count = 0
        self.optimization_history_objective = []
        self.optimization_history_mean = []

        self.init_search_space_dict = {
            'inner_denoise_std':[0.01,4],
            'outer_denoise_std':[0.01,4],
            'buy_1_threshold':[0.01,4],
            'buy_2_threshold':[0.01,4],
            'sell_1_threshold':[0.01,4],
            'sell_2_threshold':[0.01,4],
            'inner_window_length_interval':[5,30], #in unit of days
            'outer_window_length_interval':[5,30],
            'buy_1_wait':[1,10], #in unit of days
            'buy_2_wait':[1,10],
            'sell_1_wait':[1,10],
            'sell_2_wait':[1,10],
            'min_profit':[1,10], #in unit of percentage
            'pct_portfolio_in_vix':[0.01,50] #in unit of percentage
        }

        self.var_name_list = []
        self.var_name_list.append('inner_denoise_std')
        self.var_name_list.append('outer_denoise_std')
        self.var_name_list.append('buy_1_threshold')
        self.var_name_list.append('buy_2_threshold')
        self.var_name_list.append('sell_1_threshold')
        self.var_name_list.append('sell_2_threshold')
        self.var_name_list.append('inner_window_length_interval')
        self.var_name_list.append('outer_window_length_interval')
        self.var_name_list.append('buy_1_wait')
        self.var_name_list.append('buy_2_wait')
        self.var_name_list.append('sell_1_wait')
        self.var_name_list.append('sell_2_wait')
        self.var_name_list.append('min_profit')
        self.var_name_list.append('pct_portfolio_in_vix')

        self.dim = len(self.init_search_space_dict)
        lower_bound_list = []
        upper_bound_list = []
        for var_name in self.var_name_list:
            [lower_bound,upper_bound] = self.init_search_space_dict[var_name]
            lower_bound_list.append(lower_bound)
            upper_bound_list.append(upper_bound)
        
        self.lb = np.asarray(lower_bound_list)
        self.ub = np.asarray(upper_bound_list)

        #parameters for trading vix
        learned_parameter_theta_file_name = 'learned_parameter_theta.json'
        learned_parameter_theta_dict = jsonpickle.decode(open(learned_parameter_theta_file_name).read())
        self.vix_theta = learned_parameter_theta_dict['learned_parameter_theta']


    def get_one_sample_score(self,q,spy_parameters,seed_index):

        local_env = trading_vix_and_spy.trading_vix_and_spy()
        local_env.seed(seed_index)

        full_observation = local_env.reset()

        #vix observation
        vix_observation = np.concatenate((full_observation[0:5],[full_observation[6]]))
        vix_observation = np.array([e for e in vix_observation], dtype=np.float32)
        vix_current_feature = linear_trpo_utils.extract_features(vix_observation,1) #1 for the output dimension

        #spy observation
        spy_observation = full_observation[5]
        #other things related to spy trading
        last_buy_1_time = -99999
        last_buy_2_time = -99999
        last_sell_1_time = -99999
        last_sell_2_time = -99999
        min_spy_purchase_price = float('inf')
        time_index = 0

        while True:
    
            #vix trading action
            vix_action = linear_trpo_utils.compute_action_distribution(self.vix_theta,vix_current_feature,mode = 'test')
            
            #spy trading action
            inner_lower_bound, inner_upper_bound = trading_vix_and_spy_utils.generate_spy_bounds(
                spy_observation,
                int(spy_parameters['inner_window_length_interval']*self.num_data_interval_per_day),
                spy_parameters['inner_denoise_std'],
                spy_parameters['buy_1_threshold'],
                spy_parameters['sell_1_threshold']
            )

            outer_lower_bound, outer_upper_bound = trading_vix_and_spy_utils.generate_spy_bounds(
                spy_observation,
                int(spy_parameters['outer_window_length_interval']*self.num_data_interval_per_day),
                spy_parameters['outer_denoise_std'],
                spy_parameters['buy_2_threshold'],
                spy_parameters['sell_2_threshold']
            )
            
            current_spy_price = spy_observation[-1]
            
            spy_action = 0.5 #do nothing spy action is 0.5
            spy_action_type = None
            #see if we can buy
            if current_spy_price < inner_lower_bound and current_spy_price > outer_lower_bound:
                time_since_last_buy = (time_index-last_buy_1_time) #time_since_last_buy is in unit of interval
                if time_since_last_buy > spy_parameters['buy_1_wait']*self.num_data_interval_per_day:
                    spy_action = 1
                    spy_action_type = 'buy_1'
            if current_spy_price < inner_lower_bound and current_spy_price < outer_lower_bound:
                time_since_last_buy = (time_index-last_buy_2_time)
                if time_since_last_buy > spy_parameters['buy_2_wait']*self.num_data_interval_per_day:
                    spy_action = 1
                    spy_action_type = 'buy_2'
            
            #see if we can sell
            if current_spy_price > inner_upper_bound and current_spy_price < outer_upper_bound:
                time_since_last_sell = (time_index-last_sell_1_time)
                if time_since_last_sell > spy_parameters['sell_1_wait']*self.num_data_interval_per_day:
                    if current_spy_price > min_spy_purchase_price*(1+spy_parameters['min_profit']/100.0):
                        spy_action = 0
                        spy_action_type = 'sell_1'
                    
            if current_spy_price > inner_upper_bound and current_spy_price > outer_upper_bound:
                time_since_last_sell = (time_index-last_sell_2_time)
                if time_since_last_sell > spy_parameters['sell_2_wait']*self.num_data_interval_per_day:
                    if current_spy_price > min_spy_purchase_price*(1+spy_parameters['min_profit']/100.0):
                        spy_action = 0
                        spy_action_type = 'sell_2'
                        
            #form the joint action
            action_df = pd.DataFrame()
            action_df['0'] = [spy_action]
            action_df['1'] = [vix_action[0][0]] 
            action_df['2'] = [spy_parameters['pct_portfolio_in_vix']/100.0]
            
            action_array = action_df.iloc[0].tolist()
            action_array = np.reshape(action_array,(-1,))
            
            full_observation, reward, done, info = local_env.step(action_array)
            
            #prepare for the next round of action
            if info['bought_spy']:
                if spy_action_type == 'buy_1':
                    last_buy_1_time = time_index
                if spy_action_type == 'buy_2':
                    last_buy_2_time = time_index
            if info['sold_spy']:
                if spy_action_type == 'sell_1':
                    last_sell_1_time = time_index
                if spy_action_type == 'sell_2':
                    last_sell_2_time = time_index
            min_spy_purchase_price = info['min_spy_purchase_price']
            time_index += 1
            
            if done:
                break
            
            spy_observation = full_observation[5]
            
            vix_observation = np.concatenate((full_observation[0:5],[full_observation[6]]))
            vix_observation = np.array([e for e in vix_observation], dtype=np.float32)
            vix_current_feature = linear_trpo_utils.extract_features(vix_observation,1) #1 for the output dimension

        q.put([reward])


    def __call__(self,x):

        spy_parameters = {}
        
        for index in range(self.dim):
            var_name = self.var_name_list[index]
            var_value = x[index]
            spy_parameters[var_name] = var_value

       
        total_returns = []
        
        q = multiprocessing.Queue(maxsize = self.max_worker)

        for iteration_index in range(0,int(self.rounds_of_workers)):
            p_list = []
            for worker in range(0,self.max_worker):
                try:
                    seed_index = iteration_index*self.max_worker+worker
                    p = multiprocessing.Process(target = self.get_one_sample_score,\
                                                args = (q,spy_parameters,seed_index))
                    p.start()
                    p_list.append(p)
                except:
                    pass

            for j in range(len(p_list)):
                res = q.get()
                total_returns.append(res[0])

        optimization_objective = np.quantile(total_returns,0.25)

        if optimization_objective>self.best_obj_so_far:
            self.best_obj_so_far = optimization_objective
            self.best_parameters_from_experiment = spy_parameters
            print('the best objective so far is',self.best_obj_so_far)
            print('the associated best parameters are',spy_parameters)
            print('the current time is',datetime.datetime.now())
            cwd = os.getcwd()
            parameter_file = 'best_parameter_from_direct_experiment.json'
            cwd = os.path.join(cwd,parameter_file)
            with open(cwd, 'w') as statusFile:
                statusFile.write(jsonpickle.encode(self.best_parameters_from_experiment))

        print('this is global iteration',self.global_iteration_count)
        print('the objective for ths iteration is',optimization_objective)
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

        self.global_iteration_count += 1

        #optimization_objective = np.quantile(total_returns,0.25)
        #optimization_objective = np.min(total_returns)
        #since the turbo package assumes a minimization procedure
        #I need to minimize the negative of the true objective
        optimization_objective = optimization_objective*-1

        return optimization_objective