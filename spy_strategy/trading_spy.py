
import pandas as pd
import utils
import numpy as np
import random



class trading_spy():

    def __init__(self,seed_index,max_simulation_length,min_history_length):

        random.seed(seed_index)

        #load data
        ask_data = pd.read_csv("SPY_1000D_5mins_ASK.csv")
        ask_data = ask_data.rename(columns = {"date":"Date",\
                                "open":"ask_open",\
                                "high":'ask_high',\
                                'low':'ask_low',\
                                'close':'ask_close',\
                                'volume':'ask_volume'})

        bid_data = pd.read_csv("SPY_1000D_5mins_BID.csv")
        bid_data = bid_data.rename(columns = {"date":"Date",\
                                "open":"bid_open",\
                                "high":'bid_high',\
                                'low':'bid_low',\
                                'close':'bid_close',\
                                'volume':'bid_volume'})
        
        total_data = pd.merge(ask_data, bid_data, on="Date",how = 'inner')

        self.index_dataframe = total_data
        self.bid_price_list = total_data['bid_close'].tolist()
        self.ask_price_list = total_data['ask_close'].tolist()
        self.middle_price_list = ((total_data['ask_close']+total_data['bid_close'])/2).tolist()
        self.max_simulation_length = max_simulation_length #in unit of interval counts
        self.min_history_length = min_history_length #in unit of interval counts

        #other variables
        self.current_time_index = None
        self.positions = None
        self.cash = None
        self.buy_and_hold_stock_quantity = None
        self.current_portfolio_value = None
        self.last_buy_1_time = None
        self.last_buy_2_time = None
        self.last_sell_1_time = None
        self.last_sell_2_time = None

    
    def reset(self,return_price = False):

        #pick a random starting point
        self.current_time_index = random.randrange(self.min_history_length,len(self.ask_price_list)-self.max_simulation_length)

        ask_price_history = self.ask_price_list[0:self.current_time_index]
        bid_price_history = self.bid_price_list[0:self.current_time_index]
        ask_price = ask_price_history[-1]
        bid_price = bid_price_history[-1]
        middle_price_history = self.middle_price_list[0:self.current_time_index]

        self.positions = []
        self.cash = 1e5
        self.buy_and_hold_stock_quantity = np.floor(self.cash/ask_price)
        
        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*observed_bid_price
        
        self.current_portfolio_value = self.cash + value_in_stock

        self.last_buy_1_time = -1e9 #in unit of seconds
        self.last_buy_2_time = -1e9
        self.last_sell_1_time = -1e9
        self.last_sell_2_time = -1e9

        observation_dict = {}
        observation_dict['ask_price'] = ask_price
        observation_dict['bid_price'] = bid_price
        observation_dict['middle_price_history'] = middle_price_history
        observation_dict['last_buy_1_time'] = self.last_buy_1_time
        observation_dict['last_buy_2_time'] = self.last_buy_2_time
        observation_dict['last_sell_1_time'] = self.last_sell_1_time
        observation_dict['last_sell_2_time'] = self.last_sell_2_time
        observation_dict['cash'] = self.cash
        observation_dict['value_in_stock'] = value_in_stock
        observation_dict['first_position'] = None

        return observation_dict

    
    def step(self,action,buy_dollar_value):

        execute_sell = False

        ask_price_history = self.ask_price_list[0:self.current_time_index]
        bid_price_history = self.bid_price_list[0:self.current_time_index]
        ask_price = ask_price_history[-1]
        bid_price = bid_price_history[-1]


        if action == "buy_1":
            new_position = {}
            new_position['quantity'] = np.floor(buy_dollar_value/ask_price)
            new_position['price'] = ask_price
            self.positions.append(new_position)
            self.cash -= new_position['quantity'] * ask_price
            self.last_buy_1_time = self.current_time_index

        if action == "buy_2":
            new_position = {}
            new_position['quantity'] = np.floor(buy_dollar_value/ask_price)
            new_position['price'] = ask_price
            self.positions.append(new_position)
            self.cash -= new_position['quantity'] * ask_price
            self.last_buy_2_time = self.current_time_index

        if action == "sell_1":
            sold_position = self.positions.pop(0)
            sold_value = sold_position['quantity']*bid_price
            self.cash += sold_value
            self.last_sell_1_time = self.current_time_index
            execute_sell = True

        if action == "sell_2":
            sold_position = self.positions.pop(0)
            sold_value = sold_position['quantity']*bid_price
            self.cash += sold_value
            self.last_sell_2_time = self.current_time_index
            execute_sell = True


        #clear out buy time if I have 0 positions
        if len(self.positions)==0:
            self.last_buy_1_time = -1e9 #in unit of seconds
            self.last_buy_2_time = -1e9
            self.last_sell_1_time = -1e9
            self.last_sell_2_time = -1e9

        self.positions = sorted(self.positions,key = lambda k : k['price'])

        #observations
        self.current_time_index += 1
        ask_price_history = self.ask_price_list[0:self.current_time_index]
        bid_price_history = self.bid_price_list[0:self.current_time_index]
        ask_price = ask_price_history[-1]
        bid_price = bid_price_history[-1]
        middle_price_history = self.middle_price_list[0:self.current_time_index]

        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*bid_price

        self.current_portfolio_value = value_in_stock + self.cash

        observation_dict = {}
        observation_dict['ask_price'] = ask_price
        observation_dict['bid_price'] = bid_price
        observation_dict['middle_price_history'] = middle_price_history
        observation_dict['last_buy_1_time'] = self.last_buy_1_time
        observation_dict['last_buy_2_time'] = self.last_buy_2_time
        observation_dict['last_sell_1_time'] = self.last_sell_1_time
        observation_dict['last_sell_2_time'] = self.last_sell_2_time
        observation_dict['cash'] = self.cash
        observation_dict['value_in_stock'] = value_in_stock
        if len(self.positions)>0:
            observation_dict['first_position'] = self.positions[0]
        else:
            observation_dict['first_position'] = None

        reward = 0

        return observation_dict, reward, execute_sell

    def final(self):
        ask_price_history = self.ask_price_list[0:self.current_time_index]
        bid_price_history = self.bid_price_list[0:self.current_time_index]
        ask_price = ask_price_history[-1]
        bid_price = bid_price_history[-1]

        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*bid_price

        reward = (value_in_stock+self.cash)/ask_price-self.buy_and_hold_stock_quantity

        return reward

    def get_begin_index(self):
        return self.current_time_index