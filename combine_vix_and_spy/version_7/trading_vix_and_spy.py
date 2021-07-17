import pandas as pd
import numpy as np
import random
from random import randrange
import trading_vix_and_spy_utils
import ast

class trading_vix_and_spy():


    def __init__(self):
        self.total_init_cash = 50000
        self.min_spy_transaction_value = 500

        #load data
        self.index_feature_dataframe = pd.read_csv('full_feature_dataframe_lifeline_testing.csv')

        #other variables
        self.current_time_index = None
        self.current_trajectory_length = None
        self.spy_positions = None #this should be a list
        self.spy_cash = None
        self.total_portfolio_value = None
        self.total_realized_gain = None
        self.spy_price_trajectory = None


    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)


    def reset(self):
        #pick a random ending point
        self.max_trading_day = random.randint(30,100)
        self.interval_per_trading_day = 7
        self.max_trajectory_length = self.max_trading_day * self.interval_per_trading_day
        #pick a random starting point
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-self.max_trajectory_length)
        self.current_trajectory_length = 0

        #initialize other variables
        self.spy_positions = []
        self.spy_cash = self.total_init_cash #i need to rebalance this (put some money into vix) during the first step call
        self.total_portfolio_value = [self.spy_cash]
        self.total_realized_gain = 0

        observation_based_on_dataframe = self.index_feature_dataframe.iloc[self.current_time_index][2:].to_numpy() 
        #[2:] because I ignore the column 'Unnamed: 0' and 'date'
        current_spy_price_list = self.index_feature_dataframe.iloc[self.current_time_index][-1]
        current_spy_price_list = ast.literal_eval(current_spy_price_list)
        current_spy_price = current_spy_price_list[-1]
        self.spy_price_trajectory = []
        self.spy_price_trajectory.append(current_spy_price)

        returned_observation = np.concatenate((observation_based_on_dataframe,[0]),axis = 0) #[[0]] because I start off with 0 position in spy

        return np.reshape(returned_observation,(-1,))
        

    def step(self,action):
        
        bought_spy = False
        sold_spy = False

        spy_buy_sell_action = action[0] #this is either 1 ro -1
        spy_max_positions = int(action[1])*1.0
        spy_buy_cap = action[2]

        current_spy_price_list = self.index_feature_dataframe.iloc[self.current_time_index][-1]
        current_spy_price_list = ast.literal_eval(current_spy_price_list)
        current_spy_price = current_spy_price_list[-1]
        total_value_in_spy_stock = trading_vix_and_spy_utils.total_value_in_spy_stock(self.spy_cash,self.spy_positions,current_spy_price)

        if spy_buy_sell_action == 1:
            #buy spy
            #first, I compute the max dollar value I can buy
            if len(self.spy_positions) < spy_max_positions:
                spy_new_position_max_value = self.spy_cash/(spy_max_positions-len(self.spy_positions))
            else:
                spy_new_position_max_value = 0

            if self.spy_cash >= self.min_spy_transaction_value and spy_new_position_max_value > 0:
                if len(self.spy_positions) <= spy_max_positions:
                    if total_value_in_spy_stock + spy_new_position_max_value <= spy_buy_cap:
                        new_position = {}
                        stock_quantity = np.floor(spy_new_position_max_value/current_spy_price)
                        new_position['quantity'] = stock_quantity
                        new_position['bought_price'] = current_spy_price
                        self.spy_positions.append(new_position)
                        self.spy_cash -= new_position['quantity']*new_position['bought_price']
                        #compute commission
                        self.spy_cash -= trading_vix_and_spy_utils.compute_single_commission(stock_quantity)
                        self.spy_positions = sorted(self.spy_positions,key = lambda k : k['bought_price'])
                        bought_spy = True

        elif spy_buy_sell_action == -1:
            #sell spy
            if len(self.spy_positions) > 0:
                #make sure each position only generates positive cash
                if current_spy_price > self.spy_positions[0]['bought_price']:
                    sold_position = self.spy_positions.pop(0)
                    sold_value = sold_position['quantity'] * current_spy_price
                    self.spy_cash += sold_value
                    #compute commission
                    self.spy_cash -= trading_vix_and_spy_utils.compute_single_commission(sold_position['quantity'])
                    #record gain
                    self.total_realized_gain += sold_position['quantity']*(current_spy_price-sold_position['bought_price'])
                    sold_spy = True
        
        #record other variables
        info = {}
        info['bought_spy'] = bought_spy
        info['sold_spy'] = sold_spy
        info['total_portfolio_value'] = self.total_portfolio_value
        info['spy_price_trajectory'] = self.spy_price_trajectory
        if len(self.spy_positions) > 0:
            info['min_spy_purchase_price'] = self.spy_positions[0]['bought_price']
        else:
            info['min_spy_purchase_price'] = float('inf')
        info['spy_cash'] = self.spy_cash
        info['current_spy_positions'] = self.spy_positions


        reward = 0
        done = False

        if self.current_trajectory_length == self.max_trajectory_length:
            #the end of this trajectory
            done = True
            return None, reward, done, info

        #advance time
        self.current_time_index += 1
        self.current_trajectory_length += 1

        current_spy_price_list = self.index_feature_dataframe.iloc[self.current_time_index][-1]
        current_spy_price_list = ast.literal_eval(current_spy_price_list)
        current_spy_price = current_spy_price_list[-1]
        self.spy_price_trajectory.append(current_spy_price)
        total_value_in_spy = trading_vix_and_spy_utils.total_value_in_spy(self.spy_cash,self.spy_positions,current_spy_price)
        self.total_portfolio_value.append(total_value_in_spy)

        observation_based_on_dataframe = self.index_feature_dataframe.iloc[self.current_time_index][2:].to_numpy() 
        returned_spy_position = trading_vix_and_spy_utils.returned_spy_position(self.spy_cash,self.spy_positions,current_spy_price)
        returned_observation = np.concatenate((observation_based_on_dataframe,[returned_spy_position]),axis = 0) 

        returned_observation = np.reshape(returned_observation,(-1,))
        return returned_observation, reward, done, info


    def render(self):
        pass

    def close(self):
        pass