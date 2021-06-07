import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from random import randrange


class SimpleContinuousBuySellSpy():
    
    def __init__ (self, mv_feature_list = [5,10,15,30,50]):
        #first, load in the data
        index_data = pd.read_csv("SPY.csv")
        index_data = index_data.rename(columns = {
            "Date":"Date",
            "Open":"index_open",
            "High":'index_high',
            'Low':'index_low',
            'Close':'index_close',
            'Adj Close':'index_adj_close',
            'Volume':'index_volume'})

        #build feature matrix
        index_feature_dataframe = pd.DataFrame()
        index_feature_dataframe['index_raw_price'] = index_data['index_adj_close']
        period_list = [5,10,15]
        for period in period_list:
            ewm = index_feature_dataframe['index_raw_price'].ewm(span = period).mean()
            ratio = index_feature_dataframe['index_raw_price']/ewm
            index_feature_dataframe['ewm_'+str(period)] = ratio
        index_feature_dataframe = index_feature_dataframe.iloc[max(period_list):,:]
        
        index_feature_dataframe = index_feature_dataframe.reset_index(drop=True)
        self.index_feature_dataframe = index_feature_dataframe
        
        self.current_time_index = None
        self.current_portfolio_value = None
        self.positions = None
        self.cash = None
        self.min_buy_value = None
        self.buy_and_hold_stock_quantity = None
        self.expert_reward = None
        self.num_state = len(index_feature_dataframe.columns)
        self.action_space = 2

    """
    output: state, dim=[1,4]
    """
    def reset(self):
        #pick a random starting point on the self.index_feature_dataframe
        #self.current_time_index = self.begin_time[self.draw_begin_time_index]
        self.current_time_index = randrange(0, self.index_feature_dataframe.shape[0]-500)

        observation = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy()
        observation = observation.reshape((-1,1))
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        
        #initialize other variables
        self.cash = 1e5
        self.positions = []
        self.min_buy_value = 5e3
        self.buy_and_hold_stock_quantity = self.cash/current_stock_price
        self.expert_reward = 1
    
        value_in_stock = 0
        if len(self.positions)>0:
            for position in self.positions:
                value_in_stock += position['quantity']*current_stock_price
        else:
            value_in_stock = 0
        self.current_portfolio_value = self.cash + value_in_stock

        return np.concatenate((observation,[[0]]),axis = 0).reshape((-1, 4))
    
    """
    input: action, dim=[1]
    output: new state, dim=[1,4]; reward=[1]
    """
    def step(self, action):
        if self.current_portfolio_value == None:
            raise Exception("Please call reset first")

        execute_action = False
        execute_sell = False
        profit = 0
            
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        
        value_in_stock = 0
        if len(self.positions)>0:
            for position in self.positions:
                value_in_stock += position['quantity']*current_stock_price
        else:
            value_in_stock = 0
        
        
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value
        if current_percent_value_in_stock < action:
            need_to_buy = True
            need_to_sell = False
        else:
            need_to_buy = False
            need_to_sell = True
        
        
        while current_percent_value_in_stock < action and need_to_buy:
            #buy
            if self.cash > self.min_buy_value:
                new_position = {}
                new_position['quantity'] = np.floor(self.min_buy_value/current_stock_price)
                new_position['price'] = current_stock_price
                self.positions.append(new_position)
                execute_action = True
                self.cash -= new_position['quantity']*new_position['price']
            else:
                break #cannot buy anything
            
            value_in_stock = 0
            if len(self.positions)>0:
                for position in self.positions:
                    value_in_stock += position['quantity']*current_stock_price
            else:
                value_in_stock = 0
            
            self.current_portfolio_value = self.cash + value_in_stock
                
            current_percent_value_in_stock = value_in_stock/self.current_portfolio_value

            
            self.positions = sorted(self.positions,key = lambda k : k['price'])
        
        while current_percent_value_in_stock>action and need_to_sell:
            #sell
            if len(self.positions)>0:
                sold_position = self.positions[0]
                if current_stock_price>sold_position['price']:
                
                    self.cash += sold_position['quantity']*current_stock_price
                    profit += sold_position['quantity']*current_stock_price
                    execute_action = True
                    execute_sell = True
                    self.positions.pop(0)

                else:
                    break #not sell anything
            
            value_in_stock = 0
            if len(self.positions)>0:
                for position in self.positions:
                    value_in_stock += position['quantity']*current_stock_price
            else:
                value_in_stock = 0
                    
            self.current_portfolio_value = self.cash + value_in_stock
            
            current_percent_value_in_stock = value_in_stock/self.current_portfolio_value
        
    
        self.expert_reward = self.expert_reward*1.001
        self.current_time_index += 1
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        observation = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy()
        observation = observation.reshape((-1,1))
        
        value_in_stock = 0
        if len(self.positions)>0:
            for position in self.positions:
                value_in_stock += position['quantity']*current_stock_price
        else:
            value_in_stock = 0
        self.current_portfolio_value = self.cash + value_in_stock
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value
        
        reward = (self.current_portfolio_value/current_stock_price)/self.buy_and_hold_stock_quantity
        reward = reward-self.expert_reward

        observation = np.concatenate((observation,[[current_percent_value_in_stock]]),axis = 0).reshape((-1, 4))
        
        if execute_sell:
            return observation, reward
        
        return observation, 0