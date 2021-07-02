import pandas as pd
import trading_vix_and_spy_utils
import numpy as np
import random
from random import randrange
from sympy.solvers import solve
from sympy import Symbol

class trading_vix_and_spy():


    def __init__(self):

        self.max_trajectory_length = 300
        self.total_init_cash = 50000
        self.min_vix_transaction_value = 500
        self.min_spy_transaction_value = 500
        self.regular_spy_transaction = 10000
        self.min_rebalance_transaction_value = self.total_init_cash * 0.01 
        self.min_rebalance_transaction_value = 1

        #load data
        index_data = pd.read_csv("^VIX.csv")
        index_data = index_data.rename(columns = {"Date":"Date",\
                                "Open":"vix_index_open",\
                                "High":'vix_index_high',\
                                'Low':'vix_index_low',\
                                'Close':'vix_index_close',\
                                'Adj Close':'vix_index_adj_close',\
                                'Volume':'vix_index_volume'})

        vix_price_data = pd.read_csv('VIXY.csv')
        vix_price_data = vix_price_data.rename(columns = {"Date":"Date",\
                                "Open":"vix_price_open",\
                                "High":'vix_price_high',\
                                'Low':'vix_price_low',\
                                'Close':'vix_price_close',\
                                'Adj Close':'vix_price_adj_close',\
                                'Volume':'vix_price_volume'})

        spy_price_data = pd.read_csv('SPY.csv')
        spy_price_data = spy_price_data.rename(columns = {"Date":"Date",\
                                "Open":"spy_price_open",\
                                "High":'spy_price_high',\
                                'Low':'spy_price_low',\
                                'Close':'spy_price_close',\
                                'Adj Close':'spy_price_adj_close',\
                                'Volume':'spy_price_volume'})

        total_data = pd.merge(index_data, vix_price_data, on="Date",how = 'inner')
        total_data = pd.merge(total_data, spy_price_data, on="Date",how = 'inner')

        #build features for vix based on vix6 trading environment
        #compute the exponential moving average
        mv_10 = total_data['vix_index_adj_close'].ewm(span = 10).mean()
        mv_20 = total_data['vix_index_adj_close'].ewm(span = 20).mean()
        mv_30 = total_data['vix_index_adj_close'].ewm(span = 30).mean()
        mv_50 = total_data['vix_index_adj_close'].ewm(span = 50).mean()
        mv_100 = total_data['vix_index_adj_close'].ewm(span = 100).mean()

        spot_to_mv_10 = total_data['vix_index_adj_close']/mv_10
        spot_to_mv_20 = total_data['vix_index_adj_close']/mv_20
        spot_to_mv_30 = total_data['vix_index_adj_close']/mv_30
        spot_to_mv_50 = total_data['vix_index_adj_close']/mv_50
        spot_to_mv_100 = total_data['vix_index_adj_close']/mv_100

        vix_measure = spot_to_mv_10+spot_to_mv_20+spot_to_mv_30+spot_to_mv_50+spot_to_mv_100
        vix_measure_list = vix_measure.tolist()

        index_feature_dataframe = pd.DataFrame()
        index_feature_dataframe['vix_price_adj_close'] = total_data['vix_price_adj_close'][1:] #[1:] for matching counting_days
        index_feature_dataframe['vix_adj_close'] = total_data['vix_index_adj_close'][1:]
        index_feature_dataframe['mv_ratio'] = vix_measure_list[1:]
        threshold_list = [5,6,7]
        for threshold in threshold_list:
            counting_days = trading_vix_and_spy_utils.day_counter_helper(vix_measure_list,threshold)
            index_feature_dataframe['days_since_'+str(threshold)] = counting_days

        index_feature_dataframe = index_feature_dataframe.iloc[-1000:] #there may be a vix regime change in 2018/1??
        index_feature_dataframe = index_feature_dataframe.reset_index(drop=True)

        #build spy observation
        spy_data_per_day = 1 #this is a parameter that we can tune
        spy_max_observation_history = 30 #this is a parameter that we can tune
        spy_temp_data_max_rows = total_data.shape[0]-spy_max_observation_history*spy_data_per_day+1
        spy_temp_data = np.zeros((spy_temp_data_max_rows,spy_max_observation_history))

        for end_interval_index in range(total_data.shape[0]-spy_temp_data_max_rows+1,total_data.shape[0]+1):
            price_history = total_data['spy_price_adj_close'][end_interval_index-spy_data_per_day*spy_max_observation_history\
                                                            :end_interval_index]
            spy_temp_data[end_interval_index-spy_data_per_day*spy_max_observation_history,:] = price_history
            
        spy_observation_data_list = []
        for row_index in range(0,spy_temp_data.shape[0]):
            spy_observation_data_list.append(spy_temp_data[row_index,:].tolist())

        index_feature_dataframe['spy_observation'] = spy_observation_data_list[-1000:]#because there is a cut for vix

        self.index_feature_dataframe = index_feature_dataframe

        #other variables
        self.current_time_index = None
        self.current_trajectory_length = None
        self.spy_positions = None #this should be a list
        self.spy_cash = None
        self.vix_position = None #this should be a number
        self.vix_cash = None
        self.current_portfolio_value = None
        self.profit_from_spy = None
        self.vix_price_trajectory = None
        self.spy_price_trajectory = None

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        
        self.vix_price_trajectory = []
        self.spy_price_trajectory = []

        #pick a random starting point
        self.current_time_index = randrange(0,self.index_feature_dataframe.shape[0]-self.max_trajectory_length)
        self.current_trajectory_length = 0

        observation_based_on_dataframe = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy() #[1:] because I ignore vix price
        current_vix_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        current_spy_price = self.index_feature_dataframe.iloc[self.current_time_index][-1][-1]
        self.vix_price_trajectory.append(current_vix_stock_price)
        self.spy_price_trajectory.append(current_spy_price)

        #initialize other variables
        self.spy_positions = []
        self.spy_cash = self.total_init_cash #i need to rebalance this (put some money into vix) during the first step call
        self.vix_position = 0
        self.vix_cash = 0
        self.current_portfolio_value = trading_vix_and_spy_utils.compute_total_portfolio_value(self.vix_cash,\
                                                                                                self.vix_position,\
                                                                                                current_vix_stock_price,\
                                                                                                self.spy_cash,\
                                                                                                self.spy_positions,\
                                                                                                current_spy_price)
        self.profit_from_spy = 0

        returned_observation = np.concatenate((observation_based_on_dataframe,[0.0,0.0]),axis = 0) #[[0,0]] because I start off with 0 stock in vix and 0 stock in spy

        return np.reshape(returned_observation,(-1,))

    def step(self,action):

        #the returned action has shape (3,)
        #the first action is the ratio of (value_in_vix_stock)/(value_in_vix_stock+cash_for_vix)
        #the second action is a classifier. 0 or anything less than 0.5 means sell a spy position
        #1 or anything greater than 0.5 means buy a spy position
        #the third action is the ratio of (value_in_vix_stock+cash_for_vix)/(value_in_vix_stock+cash_for_vix+value_in_spy_stock+cash_for_SPY)

        action0 = np.clip(action[0],0,1) #buy sell spy
        action1 = np.clip(action[1],0,1) #buy sell vix
        action2 = np.clip(action[2],0,1) #rebalance between vix and spy

        current_vix_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        current_spy_price = self.index_feature_dataframe.iloc[self.current_time_index][-1][-1]

        #first, we use action2 to rebalance the ratio between vix and spy
        total_value_in_vix = trading_vix_and_spy_utils.total_value_in_vix(self.vix_cash,self.vix_position,current_vix_stock_price)
        total_value_in_spy = trading_vix_and_spy_utils.total_value_in_spy(self.spy_cash,self.spy_positions,current_spy_price)

        x = Symbol('x')
        value_add_to_vix = solve((total_value_in_vix-x)/(total_value_in_vix-x+total_value_in_spy+x) - action2,x)
        value_add_to_vix = float(value_add_to_vix[0])
        if np.abs(value_add_to_vix) > self.min_rebalance_transaction_value:
            if value_add_to_vix > 0:
                #move from vix cash to spy cash
                if value_add_to_vix > self.vix_cash:
                    value_add_to_vix = self.vix_cash
                self.vix_cash -= value_add_to_vix
                self.spy_cash += value_add_to_vix
            elif value_add_to_vix < 0:
                #move from spy cash to vix cash
                if np.abs(value_add_to_vix) > self.spy_cash:
                    value_add_to_vix = self.spy_cash*-1.0
                self.vix_cash -= value_add_to_vix
                self.spy_cash += value_add_to_vix
        else:
            value_add_to_vix = 0
        
        #now, we buy/sell vix etf
        sold_vix = False
        bought_vix = False

        value_in_vix_stock = self.vix_position*current_vix_stock_price
        x = Symbol('x')
        sell_vix_value = solve((value_in_vix_stock-x)/(value_in_vix_stock-x+self.vix_cash+x) - action1,x)
        if len(sell_vix_value) > 0:
            sell_vix_value = float(sell_vix_value[0])
            if sell_vix_value > 0:
                #sell vix stock and add to vix cash
                if sell_vix_value > value_in_vix_stock:
                    sell_vix_value = value_in_vix_stock
                if sell_vix_value > self.min_vix_transaction_value:
                    sold_quantity = sell_vix_value/current_vix_stock_price
                    self.vix_position -= sold_quantity
                    self.vix_cash += sell_vix_value
                    sold_vix = True

            elif sell_vix_value < 0:
                #buy vix stock
                if np.abs(sell_vix_value) > self.vix_cash:
                    sell_vix_value = self.vix_cash * -1.0
                if np.abs(sell_vix_value) > self.min_vix_transaction_value:
                    buy_quantity = (-1.0 * sell_vix_value)/current_vix_stock_price
                    self.vix_position += buy_quantity
                    self.vix_cash += sell_vix_value
                    bought_vix = True

        #now, we buy/sell spy
        sold_spy = False
        bought_spy = False
        spy_new_position_value = 0

        if action0 >= 0.8:
            #buy spy stock
            if self.spy_cash > self.regular_spy_transaction:
                buy_spy_dollar_value = self.regular_spy_transaction
            else:
                buy_spy_dollar_value = self.spy_cash
            if buy_spy_dollar_value > self.min_spy_transaction_value:
                new_position = {}
                new_position['quantity'] = np.floor(buy_spy_dollar_value/current_spy_price)
                new_position['bought_price'] = current_spy_price
                self.spy_positions.append(new_position)
                self.spy_cash -= new_position['quantity'] * current_spy_price
                bought_spy = True
                spy_new_position_value = new_position['quantity']*new_position['bought_price']

            self.spy_positions = sorted(self.spy_positions,key = lambda k : k['bought_price'])

        elif action0 <= 0.2:
            #sell spy stock
            if len(self.spy_positions) > 0:

                if current_spy_price > self.spy_positions[0]['bought_price']:
                    #make sure each position only generates positive cash
                    sold_position = self.spy_positions.pop(0)
                    sold_value = sold_position['quantity']*current_spy_price
                    self.spy_cash += sold_value
                    sold_spy = True
                    self.profit_from_spy += sold_position['quantity']*(current_spy_price-sold_position['bought_price'])

        info = {}
        info['bought_spy'] = bought_spy
        info['sold_spy'] = sold_spy
        info['total_vix_value'] = trading_vix_and_spy_utils.total_value_in_vix(self.vix_cash,self.vix_position,current_vix_stock_price)
        info['total_spy_value'] = trading_vix_and_spy_utils.total_value_in_spy(self.spy_cash,self.spy_positions,current_spy_price)
        info['vix_price_trajectory'] = self.vix_price_trajectory
        info['spy_price_trajectory'] = self.spy_price_trajectory
        info['bought_vix'] = bought_vix
        info['sold_vix'] = sold_vix
        if len(self.spy_positions) > 0:
            info['min_spy_purchase_price'] = self.spy_positions[0]['bought_price']
        else:
            info['min_spy_purchase_price'] = float('inf')
        info['value_add_to_vix'] = value_add_to_vix
        info['spy_cash'] = self.spy_cash
        info['spy_new_position_value'] = spy_new_position_value

        reward = 0
        done = False

        if self.current_trajectory_length == self.max_trajectory_length:
            #the end of this trajectory
            done = True
            reward = self.profit_from_spy
            return None, reward, done, info

        #advance time
        self.current_time_index += 1
        self.current_trajectory_length += 1

        current_vix_stock_price = self.index_feature_dataframe.iloc[self.current_time_index][0]
        current_spy_price = self.index_feature_dataframe.iloc[self.current_time_index][-1][-1]
        self.vix_price_trajectory.append(current_vix_stock_price)
        self.spy_price_trajectory.append(current_spy_price)

        observation_based_on_dataframe = self.index_feature_dataframe.iloc[self.current_time_index][1:].to_numpy() #[1:] because I ignore vix price
        value_in_vix_stock = self.vix_position*current_vix_stock_price
        if value_in_vix_stock+self.vix_cash > 0:
            returned_vix_position = value_in_vix_stock/(value_in_vix_stock+self.vix_cash)
        else:
            returned_vix_position = 0
        returned_spy_position = trading_vix_and_spy_utils.returned_spy_position(self.spy_cash,self.spy_positions,current_spy_price)

        returned_observation = np.concatenate((observation_based_on_dataframe,[returned_vix_position,returned_spy_position]),axis = 0) #[[0,0]] because I start off with 0 stock in vix and 0 stock in spy
        returned_observation = np.reshape(returned_observation,(-1,))

        return returned_observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass