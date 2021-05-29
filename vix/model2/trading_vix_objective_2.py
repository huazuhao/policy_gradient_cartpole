import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from random import randrange
import utils
from sympy.solvers import solve
from sympy import Symbol



class trading_vix():

    '''
    The objective for this trading strategy is to maximize the profit from each full cycle
    of transaction.
    '''

    def __init__ (self,threshold_list = [5,6,7]):

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
        index_feature_dataframe['vix_price_adj_close'] = total_data['vix_price_adj_close'][1:]
        index_feature_dataframe['vix_adj_close'] = total_data['index_adj_close'][1:]
        threshold_list = [5,6,7]
        for threshold in threshold_list:
            counting_days = utils.day_counter_helper(vix_measure_list,threshold)
            index_feature_dataframe['days_since_'+str(threshold)] = counting_days

        index_feature_dataframe = index_feature_dataframe.iloc[-1000:] #there may be a vix regime change in 2018/1??
        index_feature_dataframe = index_feature_dataframe.reset_index(drop=True)
        self.index_feature_dataframe = index_feature_dataframe
    
        #other variables
        self.current_time_index = None
        self.current_portfolio_value = None
        self.positions = None
        self.min_transaction_value = None
        self.buy_and_hold_stock_quantity = None



    def reset(self,return_price = False):
        #pick a random starting point on the self.index_feature_dataframe
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-201)

        observation = self.index_feature_dataframe.iloc[self.current_time_index][2:].to_numpy()
        observation = observation.reshape((-1,1))
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        
        #initialize other variables
        self.positions = []
        self.cash = 1e4
        self.min_transaction_value = 5e2
        self.buy_and_hold_stock_quantity = self.cash/current_stock_price
    
        value_in_stock = 0
        if len(self.positions)>0:
            for position in self.positions:
                value_in_stock += position['quantity']*current_stock_price
        else:
            value_in_stock = 0

        self.current_portfolio_value = self.cash + value_in_stock


        if return_price:
            return current_stock_price, np.concatenate((observation,[[0]]),axis = 0)

        return np.concatenate((observation,[[0]]),axis = 0)

    
    def step(self,action,return_price = False):
        
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
        if current_percent_value_in_stock<action:
            need_to_buy = True
            need_to_sell = False
        else:
            need_to_buy = False
            need_to_sell = True

        
        if need_to_buy:
            x = Symbol('x')
            r = solve((value_in_stock+x)/(value_in_stock+x+self.cash-x) - action,x)
            r = r[0]
            if r>self.min_transaction_value:
                already_bought_value = 0
                while already_bought_value < r:
                    #print('keep buying')
                    #print('r is',r)
                    #print('already bought value is',already_bought_value)
                    new_position = {}
                    new_position['quantity'] = self.min_transaction_value/current_stock_price
                    new_position['price'] = current_stock_price
                    self.positions.append(new_position)
                    execute_action = True
                    self.cash -= new_position['quantity']*new_position['price']
                    already_bought_value += new_position['quantity']*new_position['price']
                    if self.cash<self.min_transaction_value:
                        break #don't have enough cash to buy

        self.positions = sorted(self.positions,key = lambda k : k['price'])


        if need_to_sell:
            x = Symbol('x')
            r = solve((value_in_stock-x)/(value_in_stock-x+self.cash) - action,x)

            if len(r)==0 and self.cash < 1:
                #sell everything
                x = Symbol('x')
                temp_cash = 0.01
                r = solve((value_in_stock-x)/(value_in_stock-x+temp_cash) - action,x)

            if len(r)==0:
                print(value_in_stock)
                print(self.cash)
                print(action)

            r = r[0]
            if r>self.min_transaction_value:
                already_sold_value = 0
                while already_sold_value < r:
                    if len(self.positions) > 0:
                        #print('keep selling')
                        sold_position = self.positions.pop(0)
                        self.cash += sold_position['quantity']*current_stock_price
                        profit += sold_position['quantity']*(current_stock_price-sold_position['price'])
                        execute_action = True
                        execute_sell = True
                        already_sold_value += sold_position['quantity']*current_stock_price
                    else:
                        break #don't have any more positions to sell


        self.current_time_index += 1
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        value_in_stock = 0
        if len(self.positions)>0:
            for position in self.positions:
                value_in_stock += position['quantity']*current_stock_price
        else:
            value_in_stock = 0
        self.current_portfolio_value = self.cash + value_in_stock
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value

        observation = self.index_feature_dataframe.iloc[self.current_time_index][2:].to_numpy()
        observation = observation.reshape((-1,1))

        observation = np.concatenate((observation,[[current_percent_value_in_stock]]),axis = 0)

        if return_price:
            return current_stock_price,observation,execute_action,need_to_buy,need_to_sell

        if execute_sell:
            return observation,profit

        return observation,0


    def final(self):
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        reward = (self.current_portfolio_value/current_stock_price)-self.buy_and_hold_stock_quantity
        reward = 0
        return reward