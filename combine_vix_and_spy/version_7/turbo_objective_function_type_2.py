import multiprocessing
import numpy as np
import trading_vix_and_spy
import jsonpickle
import trading_vix_and_spy_utils
import datetime
import os
import pandas as pd
import ast
import pickle


class turbo_objective_function():

    def __init__(self):

        self.max_worker = multiprocessing.cpu_count()
        if self.max_worker > 1:
            self.max_worker -= 5
        self.max_worker = 25
        self.rounds_of_workers = 2

        self.interval_per_trading_day = 7

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
            'inner_window_length_interval':[5,39], #in unit of days
            'outer_window_length_interval':[5,39],
            'buy_1_wait':[1,5], #in unit of days
            'buy_2_wait':[1,5],
            'sell_1_wait':[1,5],
            'sell_2_wait':[1,5],
            'min_profit':[1,4], #in unit of percentage
            'pct_portfolio_in_vix_model':[15,50], #in unit of percentage
            'max_spy_positions':[8,15]
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
        self.var_name_list.append('pct_portfolio_in_vix_model')
        self.var_name_list.append('max_spy_positions')

        self.dim = len(self.init_search_space_dict)
        lower_bound_list = []
        upper_bound_list = []
        for var_name in self.var_name_list:
            [lower_bound,upper_bound] = self.init_search_space_dict[var_name]
            lower_bound_list.append(lower_bound)
            upper_bound_list.append(upper_bound)
        
        self.lb = np.asarray(lower_bound_list)
        self.ub = np.asarray(upper_bound_list)

        #the vix model
        with open('lifeline_model.pickle', 'rb') as f:
            self.vix_model = pickle.load(f)
        self.last_valid_data_for_vix = 52



    def get_one_sample_score(self,q,spy_parameters,seed_index):

        local_env = trading_vix_and_spy.trading_vix_and_spy()
        local_env.seed(seed_index)

        current_spy_stock_value = 0
        current_spy_cash_value = local_env.total_init_cash
        inner_lower_bound_history = []
        inner_upper_bound_history = []
        outer_lower_bound_history = []
        outer_upper_bound_history = []
        spy_buy_sell = []
        spy_price_history = None
        total_portfolio_value_history = None
        cash_history = []
        ratio_in_stock_form = []

        #other things related to spy trading
        last_buy_1_time = -99999
        last_buy_2_time = -99999
        last_sell_1_time = -99999
        last_sell_2_time = -99999
        min_spy_purchase_price = float('inf')
        time_index = 0

        full_observation = local_env.reset()
        spy_observation = ast.literal_eval(full_observation[-2])
        ratio_in_stock_form.append(full_observation[-1])

        while True:
    
            inner_lower_bound, inner_upper_bound = trading_vix_and_spy_utils.generate_spy_bounds(
            spy_observation,
            int(spy_parameters['inner_window_length_interval']*self.interval_per_trading_day),
            spy_parameters['inner_denoise_std'],
            spy_parameters['buy_1_threshold'],
            spy_parameters['sell_1_threshold']
            )

            outer_lower_bound, outer_upper_bound = trading_vix_and_spy_utils.generate_spy_bounds(
            spy_observation,
            int(spy_parameters['outer_window_length_interval']*self.interval_per_trading_day),
            spy_parameters['outer_denoise_std'],
            spy_parameters['buy_2_threshold'],
            spy_parameters['sell_2_threshold']
            )
            
            current_spy_price = spy_observation[-1]
            
            spy_action = 0.0 #do nothing spy action is 0.0
            spy_action_type = None
            #see if we can buy
            if current_spy_price < inner_lower_bound and current_spy_price > outer_lower_bound:
                time_since_last_buy = (time_index-last_buy_1_time) #time_since_last_buy is in unit of interval
                if time_since_last_buy > spy_parameters['buy_1_wait']*self.interval_per_trading_day:
                    spy_action = 1
                    spy_action_type = 'buy_1'
            if current_spy_price < inner_lower_bound and current_spy_price < outer_lower_bound:
                time_since_last_buy = (time_index-last_buy_2_time)
                if time_since_last_buy > spy_parameters['buy_2_wait']*self.interval_per_trading_day:
                    spy_action = 1
                    spy_action_type = 'buy_2'
            
            #see if we can sell
            if current_spy_price > inner_upper_bound and current_spy_price < outer_upper_bound:
                time_since_last_sell = (time_index-last_sell_1_time)
                if time_since_last_sell > spy_parameters['sell_1_wait']*self.interval_per_trading_day:
                    if current_spy_price > min_spy_purchase_price*(1+spy_parameters['min_profit']/100):
                        spy_action = -1
                        spy_action_type = 'sell_1'
                    
            if current_spy_price > inner_upper_bound and current_spy_price > outer_upper_bound:
                time_since_last_sell = (time_index-last_sell_2_time)
                if time_since_last_sell > spy_parameters['sell_2_wait']*self.interval_per_trading_day:
                    if current_spy_price > min_spy_purchase_price*(1+spy_parameters['min_profit']/100):
                        spy_action = -1
                        spy_action_type = 'sell_2'
                        
            #compute the max allowed buy dollar value
            prediction = self.vix_model.predict_survival_function(full_observation[0:self.last_valid_data_for_vix].reshape((1,-1)))
            vix_model_prediction = np.mean(prediction[0].tolist()[-10:])
            spy_buy_cap = (current_spy_stock_value+current_spy_cash_value)*(1-spy_parameters['pct_portfolio_in_vix_model']/100.0) + \
                        (current_spy_stock_value+current_spy_cash_value)*(spy_parameters['pct_portfolio_in_vix_model']/100.0)*vix_model_prediction

            #form the joint action
            action_df = pd.DataFrame()
            action_df['0'] = [spy_action]
            action_df['1'] = [spy_parameters['max_spy_positions']] 
            action_df['2'] = [spy_buy_cap]
            
            action_array = action_df.iloc[0].tolist()
            action_array = np.reshape(action_array,(-1,))
            
            #apply action
            full_observation, reward, done, info = local_env.step(action_array)
            if done != True:
                spy_observation = ast.literal_eval(full_observation[-2])
                ratio_in_stock_form.append(full_observation[-1])
            
            cash_history.append(info['spy_cash'])

    
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
            
            #record information for plotting
            inner_lower_bound_history.append(inner_lower_bound)
            inner_upper_bound_history.append(inner_upper_bound)
            outer_lower_bound_history.append(outer_lower_bound)
            outer_upper_bound_history.append(outer_upper_bound)
            
            if info['bought_spy']:
                spy_buy_sell.append(1)
            elif info['sold_spy']:
                spy_buy_sell.append(-1)
            else:
                spy_buy_sell.append(0)

            current_spy_cash_value = local_env.spy_cash
            current_spy_stock_value = trading_vix_and_spy_utils.total_value_in_spy_stock(current_spy_cash_value,info['current_spy_positions'],current_spy_price)
            
            if done:
                total_portfolio_value_history = info['total_portfolio_value']
                spy_price_history = info['spy_price_trajectory']
                break

        #outside of the while loop
        total_realized_gain = local_env.total_realized_gain
        total_realized_gain = total_realized_gain/(local_env.max_trajectory_length*1.0)

        #compute sharp ratio
        total_portfolio_return_history = np.asarray(total_portfolio_value_history)/total_portfolio_value_history[0]
        sharpe_ratio = total_portfolio_return_history[-1] - total_portfolio_return_history[0]
        if np.std(total_portfolio_return_history) <= 0.00001:
            sharpe_ratio = -99999
        else:
            sharpe_ratio = sharpe_ratio / np.std(total_portfolio_return_history)

        q.put([total_realized_gain,sharpe_ratio])



    def __call__(self,x):

        spy_parameters = {}
        
        for index in range(self.dim):
            var_name = self.var_name_list[index]
            var_value = x[index]
            spy_parameters[var_name] = var_value

       
        total_returns = []
        sharpe_ratios = []

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
                sharpe_ratios.append(res[1])

        #optimization_objective = np.min([np.quantile(sharpe_ratios,0.25),np.mean(sharpe_ratios)])
        optimization_objective = np.mean(sharpe_ratios)
        if np.isnan(optimization_objective):
            print('the optimization objective might be nan',optimization_objective)
            optimization_objective = -99999
        optimization_objective = optimization_objective + np.clip(np.mean(total_returns)/10,0,0.5)#divide by 10 when the data is hourly
        

        if optimization_objective>self.best_obj_so_far:
            self.best_obj_so_far = optimization_objective
            self.best_parameters_from_experiment = spy_parameters
            print('the best objective so far is',self.best_obj_so_far)
            print('the associated best parameters are',spy_parameters)
            print('the current time is',datetime.datetime.now())
            cwd = os.getcwd()
            parameter_file = 'best_parameter_from_direct_experiment_type_2.json'
            cwd = os.path.join(cwd,parameter_file)
            with open(cwd, 'w') as statusFile:
                statusFile.write(jsonpickle.encode(self.best_parameters_from_experiment))

        print('we are optimizing over sharpe ratio')
        print('this is global iteration',self.global_iteration_count)
        print('the objective for ths iteration is',optimization_objective)
        print('the mean of total_returns is',np.mean(total_returns))
        print('the mean of sharpe ratio is',np.mean(sharpe_ratios))
        print('the current time is',datetime.datetime.now())
        print(' ')

        self.optimization_history_mean.append(np.mean(total_returns))
        self.optimization_history_objective.append(optimization_objective)
        optimization_history = {}
        optimization_history['return_mean_history'] = self.optimization_history_mean
        optimization_history['objective_history'] = self.optimization_history_objective
        #store the history
        cwd = os.getcwd()
        #cwd = os.path.join(cwd, 'data_folder')
        parameter_file = 'optimization_history_type_2.json'
        cwd = os.path.join(cwd,parameter_file)
        with open(cwd, 'w') as statusFile:
            statusFile.write(jsonpickle.encode(optimization_history))

        self.global_iteration_count += 1

        #since the turbo package assumes a minimization procedure
        #I need to minimize the negative of the true objective
        optimization_objective = optimization_objective*-1

        return optimization_objective