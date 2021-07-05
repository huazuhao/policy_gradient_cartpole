import gym
from gym import spaces
import pandas as pd
import numpy as np
import random
from random import randrange
from sympy.solvers import solve
from sympy import Symbol

class trading_vix_env(gym.Env):

    def __init__(self):
        super(trading_vix_env,self).__init__()

        self.max_trajectory_length_in_days = 100
        self.intervals_per_day = 7
        self.max_trajectory_length = self.intervals_per_day * self.max_trajectory_length_in_days

        #load data
        self.index_feature_dataframe = pd.read_csv("full_feature_dataframe.csv")

        #observation and action space
        self.action_space = spaces.box.Box(
            low=0, #no position
            high=1, #all in stock
            shape=(1,),
            dtype=np.float32
        )

        high_observation = np.asarray([9999]*(self.index_feature_dataframe.shape[1]-6)) #i don't get to observe date,vixy bid/ask, spy bid/ask/mid
        self.observation_space = spaces.box.Box(
            low=-1*high_observation,
            high=high_observation,
            dtype=np.float32
        )

        #other variables
        self.current_time_index = None
        self.quantity = None #how many vixy shares i own
        self.cash = None
        self.min_transaction_value = None
        self.buy_and_hold_stock_quantity = None
        self.current_portfolio_value = None
        self.current_trajectory_length = None
        self.has_at_least_one_sell = None


    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)


    def reset(self):
        #pick a random starting point on the self.index_feature_dataframe
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-self.max_trajectory_length-50) #for some safety margin

        observation = self.index_feature_dataframe.iloc[self.current_time_index][2:25].to_numpy()
        observation = observation.reshape((-1,1))
        current_vixy_sell_price = self.index_feature_dataframe.iloc[self.current_time_index][25] #sell price or bid price
        current_vixy_buy_price = self.index_feature_dataframe.iloc[self.current_time_index][26] #buy price or ask price
        returned_observation = np.concatenate((observation,[[0]]),axis = 0) #[[0]] because I start off with 0 in vix
        returned_observation = returned_observation.astype('float64')

        #initialize other variables
        self.quantity = 0
        self.cash = 1e4
        self.min_transaction_value = 5e2
        self.buy_and_hold_stock_quantity = self.cash/current_vixy_buy_price
    
        value_in_stock = self.quantity*current_vixy_sell_price
        self.current_portfolio_value = self.cash + value_in_stock

        self.current_trajectory_length = 0
        self.has_at_least_one_sell = False

        return np.reshape(returned_observation,(-1,))


    def step(self,action):

        action = np.clip(action, 0, 1)[0]

        if self.current_portfolio_value == None:
            raise Exception("Please call reset first")

        execute_sell = False
        execute_buy = False
            
        current_vixy_sell_price = self.index_feature_dataframe.iloc[self.current_time_index][25] #sell price or bid price
        current_vixy_buy_price = self.index_feature_dataframe.iloc[self.current_time_index][26] #buy price or ask price
        
        value_in_stock = self.quantity*current_vixy_sell_price
        
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value
        if current_percent_value_in_stock<action:
            need_to_buy = True
            need_to_sell = False
        else:
            need_to_buy = False
            need_to_sell = True

        average_future_price = self.index_feature_dataframe.iloc[self.current_time_index:self.current_time_index+30]
        average_future_price = np.mean(average_future_price['vixy_bid_close'])
        sell_price_diff_of_new_minus_future = current_vixy_sell_price - average_future_price


        if need_to_buy:
            x = Symbol('x')
            r = solve((value_in_stock+x)/(value_in_stock+x+self.cash-x) - action,x)
            r = float(r[0])
            if r>self.min_transaction_value:
                if r > self.cash:
                    r = self.cash #cannot buy more than cash
                self.cash -= r
                bought_quantity = r/current_vixy_buy_price
                self.quantity += bought_quantity
                execute_buy = True


        if need_to_sell:
            x = Symbol('x')
            r = solve((value_in_stock-x)/(value_in_stock-x+self.cash+x) - action,x)
            r = float(r[0])
            if r>self.min_transaction_value:
                sold_quantity = r/current_vixy_sell_price
                if sold_quantity > self.quantity:
                    sold_quantity = self.quantity
                self.quantity -= sold_quantity
                self.cash += sold_quantity*current_vixy_sell_price
                execute_sell = True
                if self.has_at_least_one_sell == False:
                    self.has_at_least_one_sell = True

        
        self.current_time_index += 1
        self.current_trajectory_length += 1
        current_vixy_sell_price = self.index_feature_dataframe.iloc[self.current_time_index][25] #sell price or bid price
        current_vixy_buy_price = self.index_feature_dataframe.iloc[self.current_time_index][26] #buy price or ask price
        value_in_stock = self.quantity*current_vixy_sell_price
        self.current_portfolio_value = self.cash + value_in_stock
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value

        observation = self.index_feature_dataframe.iloc[self.current_time_index][2:25].to_numpy()
        observation = observation.reshape((-1,1))

        observation = np.concatenate((observation,[[current_percent_value_in_stock]]),axis = 0)

        reward = 0
        #when I buy and the stock subsequently rises, then I get a positive reward. 
        if execute_buy:
            reward = sell_price_diff_of_new_minus_future*-1*10
        #when I sell and the stock subsequently falls, then I get a positive reward
        if execute_sell:
            reward = sell_price_diff_of_new_minus_future*10
        reward = 0

        info = {}
        info['current_portfolio'] = self.current_portfolio_value
        info['execute_buy'] = execute_buy
        info['execute_sell'] = execute_sell
        info['current_vix_sell_price'] = current_vixy_sell_price

        if self.current_trajectory_length == self.max_trajectory_length:
            #the end of this trajectory
            done = True
            reward = (self.current_portfolio_value/current_vixy_sell_price)-self.buy_and_hold_stock_quantity
            returned_observation = np.reshape(observation,(-1,))
            returned_observation = returned_observation.astype('float64')
            print('the reward is',reward)
            if self.has_at_least_one_sell == False:
                reward = 0
            return returned_observation, reward, done, info 
            
        done = False
        returned_observation = np.reshape(observation,(-1,))
        returned_observation = returned_observation.astype('float64')

        return returned_observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass