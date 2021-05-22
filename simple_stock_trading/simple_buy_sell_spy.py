import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from random import randrange


class simple_buy_sell_spy():
    
    def __init__ (self,mv_feature_list = [5,10,15,30,50,100]):
        #first, load in the data
        index_data = pd.read_csv("SPY.csv")
        index_data = index_data.rename(columns = {"Date":"Date",\
                                   "Open":"index_open",\
                                   "High":'index_high',\
                                   'Low':'index_low',\
                                   'Close':'index_close',\
                                   'Adj Close':'index_adj_close',\
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
        
        self.current_index = None
        self.current_portfolio_value = None
        self.have_position = None
        self.stock_quantity = None
        self.cash = None
        self.buy_and_hold_stock_quantity = None
        self.bought_price = float('inf')


        
        
        
    def reset(self,return_price = False):
        #pick a random starting point on the self.index_feature_dataframe
        self.current_index = randrange(0,self.index_feature_dataframe.shape[0]-500)
        observation = self.index_feature_dataframe.iloc[self.current_index][1:].to_numpy()
        observation = observation.reshape((-1,1))
        
        
        #initialize other variables
        self.cash = 1e5
        self.stock_quantity = 0
        self.current_portfolio_value = self.cash + self.stock_quantity*\
                                        self.index_feature_dataframe.iloc[self.current_index][0]
        self.have_position = False
        self.buy_and_hold_stock_quantity = self.cash/self.index_feature_dataframe.iloc[self.current_index][0]
        self.previous_reward = 1

        if return_price:
            current_stock_price = self.index_feature_dataframe.iloc[self.current_index][0]
            return current_stock_price, np.concatenate((observation,[[0]]),axis = 0)


        return np.concatenate((observation,[[0]]),axis = 0)
    
    
    
    def step(self,action,return_price = False,final_step = False):

        execute_action = False

        if self.current_portfolio_value == None:
            raise Exception("Please call reset first")
        
        if self.current_index == None:
            raise Exception("Please call reset first")
            
        if self.have_position == None:
            raise Exception("Please call reset first")
        
        current_stock_price = self.index_feature_dataframe.iloc[self.current_index][0]
        
        if action == 2:
            #buy
            if self.have_position == False:
                self.stock_quantity = np.floor(self.cash/current_stock_price)
                self.have_position = True
                self.cash -= self.stock_quantity*current_stock_price
                execute_action = True
                self.bought_price = current_stock_price
                if return_price:
                    print('the bought price is',self.bought_price)

        elif action == 1:
            #sell
            if self.have_position == True:
                if current_stock_price > self.bought_price:
                    reward = self.stock_quantity*(current_stock_price-self.bought_price)
                    self.cash += self.stock_quantity*current_stock_price
                    self.stock_quantity = 0
                    self.have_position = False
                    execute_action = True
                    if return_price:
                        print('the sold price is',current_stock_price)
                        print(' ')

        
        elif action == 0 :
            #hold
            pass
        
        
        #compute reward
        self.current_portfolio_value = self.cash + self.stock_quantity*\
                                        self.index_feature_dataframe.iloc[self.current_index][0]
        reward = (self.current_portfolio_value)/(self.index_feature_dataframe.iloc[self.current_index][0]*self.buy_and_hold_stock_quantity)
        # reward = (self.current_portfolio_value/self.index_feature_dataframe.iloc[self.current_index][0])/self.buy_and_hold_stock_quantity
        
        # return_reward = reward/self.previous_reward
        
        # if reward>self.previous_reward:
        #     self.previous_reward = reward


        #move one time step
        self.current_index += 1
        observation = self.index_feature_dataframe.iloc[self.current_index][1:].to_numpy()
        observation = observation.reshape((-1,1))
        
        if self.have_position == False:
            observation = np.concatenate((observation,[[0]]),axis = 0)
        else:
            observation = np.concatenate((observation,[[1]]),axis = 0)

        if return_price:
            return current_stock_price,observation,execute_action

        
        if execute_action and action == 1:
            return observation,reward

        return observation,0

        