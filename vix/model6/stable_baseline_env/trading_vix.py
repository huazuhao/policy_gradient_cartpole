import pandas as pd
import trading_vix_utils
from random import randrange
from sympy.solvers import solve
from sympy import Symbol
import numpy as np
import random

class trading_vix():

    def __init__ (self,threshold_list = [5,6,7]):

        seed_index = 0
        np.random.seed(seed_index)
        random.seed(seed_index)

        #load data
        index_data = pd.read_csv("^VIX.csv")
        index_data = index_data.rename(columns = {"Date":"Date",\
                                "Open":"index_open",\
                                "High":'index_high',\
                                'Low':'index_low',\
                                'Close':'index_close',\
                                'Adj Close':'index_adj_close',\
                                'Volume':'index_volume'})

        vix_price_data = pd.read_csv('VIXY.csv')
        vix_price_data = vix_price_data.rename(columns = {"Date":"Date",\
                                "Open":"vix_price_open",\
                                "High":'vix_price_high',\
                                'Low':'vix_price_low',\
                                'Close':'vix_price_close',\
                                'Adj Close':'vix_price_adj_close',\
                                'Volume':'vix_price_volume'})

        total_data = pd.merge(index_data, vix_price_data, on="Date",how = 'inner')

        #build features
        #compute the exponential moving average
        mv_10 = total_data['index_adj_close'].ewm(span = 10).mean()
        mv_20 = total_data['index_adj_close'].ewm(span = 20).mean()
        mv_30 = total_data['index_adj_close'].ewm(span = 30).mean()
        mv_50 = total_data['index_adj_close'].ewm(span = 50).mean()
        mv_100 = total_data['index_adj_close'].ewm(span = 100).mean()

        spot_to_mv_10 = total_data['index_adj_close']/mv_10
        spot_to_mv_20 = total_data['index_adj_close']/mv_20
        spot_to_mv_30 = total_data['index_adj_close']/mv_30
        spot_to_mv_50 = total_data['index_adj_close']/mv_50
        spot_to_mv_100 = total_data['index_adj_close']/mv_100

        vix_measure = spot_to_mv_10+spot_to_mv_20+spot_to_mv_30+spot_to_mv_50+spot_to_mv_100
        vix_measure_list = vix_measure.tolist()


        index_feature_dataframe = pd.DataFrame()
        index_feature_dataframe['vix_price_adj_close'] = total_data['vix_price_adj_close'][1:] #[1:] for matching counting_days
        index_feature_dataframe['vix_adj_close'] = total_data['index_adj_close'][1:]
        index_feature_dataframe['mv_ratio'] = vix_measure_list[1:]
        threshold_list = [5,6,7]
        for threshold in threshold_list:
            counting_days = trading_vix_utils.day_counter_helper(vix_measure_list,threshold)
            index_feature_dataframe['days_since_'+str(threshold)] = counting_days

        index_feature_dataframe = index_feature_dataframe.iloc[-1000:] #there may be a vix regime change in 2018/1??
        index_feature_dataframe = index_feature_dataframe.reset_index(drop=True)
        self.index_feature_dataframe = index_feature_dataframe
    
        #other variables
        self.current_time_index = None
        self.quantity = None
        self.cash = None
        self.min_transaction_value = None
        self.buy_and_hold_stock_quantity = None
        self.current_portfolio_value = None


    def reset(self,return_price = False):
        #pick a random starting point on the self.index_feature_dataframe
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-201) #201 because I assume 201 days of trading simulation length

        observation = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy() #[1:] because I ignore price
        observation = observation.reshape((-1,1))
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        current_vix = self.index_feature_dataframe.iloc[self.current_time_index][1]
        returned_observation = np.concatenate((observation,[[0]]),axis = 0) #[[0]] because I start off with 0 in vix

        #initialize other variables
        self.quantity = 0
        self.cash = 1e4
        self.min_transaction_value = 5e2
        self.buy_and_hold_stock_quantity = self.cash/current_stock_price
    
        value_in_stock = self.quantity*current_stock_price
        self.current_portfolio_value = self.cash + value_in_stock


        if return_price:
            return current_stock_price, returned_observation, self.current_portfolio_value, current_vix

        return returned_observation


    def step(self,action,return_price = False):

        # if action < 0:
        #     action = 0.001
        # if action > 1:
        #     action = 0.999
        
        if self.current_portfolio_value == None:
            raise Exception("Please call reset first")

        execute_action = False
        execute_sell = False
            
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        current_vix = self.index_feature_dataframe.iloc[self.current_time_index][1]
        
        value_in_stock = self.quantity*current_stock_price
        
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value
        if current_percent_value_in_stock<action:
            need_to_buy = True
            need_to_sell = False
        else:
            need_to_buy = False
            need_to_sell = True

        
        if need_to_buy:
            x = Symbol('x')
            r = solve((value_in_stock+x)/(value_in_stock+x+self.cash-x) - action,x)
            r = float(r[0])
            if r>self.min_transaction_value:
                if r > self.cash:
                    r = self.cash #cannot buy more than cash
                self.cash -= r
                bought_quantity = r/current_stock_price
                self.quantity += bought_quantity
                execute_action = True


        if need_to_sell:
            x = Symbol('x')
            r = solve((value_in_stock-x)/(value_in_stock-x+self.cash+x) - action,x)
            r = float(r[0])
            if r>self.min_transaction_value:
                sold_quantity = r/current_stock_price
                if sold_quantity > self.quantity:
                    sold_quantity = self.quantity
                self.quantity -= sold_quantity
                self.cash += sold_quantity*current_stock_price
                execute_action = True
                execute_sell = True



        self.current_time_index += 1
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        value_in_stock = self.quantity*current_stock_price
        self.current_portfolio_value = self.cash + value_in_stock
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value

        observation = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy()
        observation = observation.reshape((-1,1))

        observation = np.concatenate((observation,[[current_percent_value_in_stock]]),axis = 0)

        reward = 0 #I only care about the long term reward

        if return_price:
            return current_stock_price,observation,execute_action,need_to_buy,need_to_sell,self.current_portfolio_value,r,current_vix


        return observation,reward,execute_sell


    def final(self):
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        reward = (self.current_portfolio_value/current_stock_price)-self.buy_and_hold_stock_quantity
        return reward