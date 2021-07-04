import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import random
from random import randrange
import trading_vix_utils
from sympy.solvers import solve
from sympy import Symbol


class trading_vix_env(gym.Env):

    metadata = {'render.modes':['cannot_render']}


    def __init__(self):
        super(trading_vix_env, self).__init__()

        self.action_space = spaces.box.Box(
            low=0, #no position
            high=1, #all in stock
            shape=(1,),
            dtype=np.float32
        )

        high_observation = np.asarray([999,999,999,999,999,999])
        self.observation_space = spaces.box.Box(
            low=-1*high_observation,
            high=high_observation,
            dtype=np.float32
        )

        self.seed()

        self.max_trajectory_length = 101
        
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
        self.current_trajectory_length = None
        self.has_at_least_one_sell = None

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        # return [seed]
        pass

    def reset(self):
        #pick a random starting point on the self.index_feature_dataframe
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-self.max_trajectory_length) #201 because I assume 201 days of trading simulation length

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

        self.current_trajectory_length = 0
        self.has_at_least_one_sell = False

        # if return_price:
        #     return current_stock_price, returned_observation, self.current_portfolio_value, current_vix

        return np.reshape(returned_observation,(-1,))


    def step(self,action):

        # if action < 0:
        #     action = 0.001
        # if action > 1:
        #     action = 0.999

        action = np.clip(action, 0, 1)[0]
        
        if self.current_portfolio_value == None:
            raise Exception("Please call reset first")

        execute_action = False
        execute_sell = False
        execute_buy = False
            
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

        
        future_data_frame = self.index_feature_dataframe.iloc[self.current_time_index:self.current_time_index+5]
        last_vix_price = current_stock_price
        future_vix_price = np.mean(future_data_frame['vix_price_adj_close'])
        
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
                execute_buy = True


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
                if self.has_at_least_one_sell == False:
                    self.has_at_least_one_sell = True



        self.current_time_index += 1
        self.current_trajectory_length += 1
        current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        value_in_stock = self.quantity*current_stock_price
        self.current_portfolio_value = self.cash + value_in_stock
        current_percent_value_in_stock = value_in_stock/self.current_portfolio_value

        observation = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy()
        observation = observation.reshape((-1,1))

        observation = np.concatenate((observation,[[current_percent_value_in_stock]]),axis = 0)

        reward = 0
        #when I buy and the stock subsequently rises, then I get a positive reward. 
        if execute_buy:
            if last_vix_price<future_vix_price:
                reward = 1
            else:
                reward = -1
        #when I sell and the stock subsequently falls, then I get a positive reward
        if execute_sell:
            if last_vix_price>future_vix_price:
                reward = 1
            else:
                reward = -1
        reward = 0

        # if return_price:
        #     return current_stock_price,observation,execute_action,need_to_buy,need_to_sell,self.current_portfolio_value,r,current_vix
        info = {}
        info['current_portfolio'] = self.current_portfolio_value
        info['execute_buy'] = execute_buy
        info['execute_sell'] = execute_sell
        info['current_stock_price'] = current_stock_price

        if self.current_trajectory_length == self.max_trajectory_length:
            #the end of this trajectory
            done = True
            current_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
            reward = (self.current_portfolio_value/current_stock_price)-self.buy_and_hold_stock_quantity
            returned_observation = np.reshape(observation,(-1,))
            print('the reward is',reward)
            if self.has_at_least_one_sell == False:
                reward = 0
            return returned_observation, reward, done, info 
            
        done = False
        returned_observation = np.reshape(observation,(-1,))

        return returned_observation,reward,done, info

    def render(self):
        pass

    def close(self):
        pass
