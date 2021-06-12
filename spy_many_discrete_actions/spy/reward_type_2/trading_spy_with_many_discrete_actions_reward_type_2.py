import random
import numpy as np
import pandas as pd
from sympy.solvers import solve
from sympy import Symbol




class trading_spy():

    '''
    The reward for this trading environment is measured by how well I can beat an expert.
    '''

    def __init__(self,max_simulation_length,min_history_length,max_position,init_cash_value):

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
        self.cash = init_cash_value
        self.buy_and_hold_stock_quantity = None
        self.initial_portfolio_value = init_cash_value
        self.current_portfolio_value = None
        self.max_position = max_position
        self.expert_policy = None
        self.last_action = None
        self.ask_price_history = None
        self.bid_price_history = None
        self.middle_price_history = None

    def reset(self):

        #pick a random starting point
        end_buffer = 1000
        self.current_time_index = random.randrange(self.min_history_length,len(self.ask_price_list)-self.max_simulation_length-end_buffer)

        self.ask_price_history = self.ask_price_list[0:self.current_time_index+1]
        self.bid_price_history = self.bid_price_list[0:self.current_time_index+1]
        ask_price = self.ask_price_history[-1]
        bid_price = self.bid_price_history[-1]
        self.middle_price_history = self.middle_price_list[0:self.current_time_index+1]

        self.positions = []
        self.buy_and_hold_stock_quantity = np.floor(self.cash/ask_price)
        self.expert_policy = 1
        self.last_action = None
        
        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*bid_price
        
        self.current_portfolio_value = self.cash + value_in_stock

        current_stock_portfolio_ratio = value_in_stock/self.current_portfolio_value

        observation_dict = {}
        price_for_percentage_computation = self.middle_price_history[-1*self.min_history_length:]
        observed_price_history = [100.0 * a1 / a2 - 100 for a1, a2 in zip(price_for_percentage_computation[1:], price_for_percentage_computation)]
        observation_dict['price_history'] = observed_price_history
        observation_dict['current_stock_ratio'] = current_stock_portfolio_ratio
        observation_dict['current_portfolio_value'] = self.current_portfolio_value


        return observation_dict

    def step(self,action):

        execute_action = False
        execute_sell = False
        execute_buy = False

        ask_price = self.ask_price_history[-1]
        bid_price = self.bid_price_history[-1]
        

        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*bid_price
        
        #the action is an integer anywhere between 0 and max_position
        if action != self.last_action:

            self.last_action = action
            x = Symbol('x')
            r = solve((value_in_stock+x)/(value_in_stock+x+self.cash-x) - (action/self.max_position),x)

            r = float(r[0])

            if r>0:
                #buy
                if r>self.cash:
                    r = self.cash
                
                num_new_position = action-len(self.positions)
                
                if num_new_position > 0 :

                    per_position_value = r/num_new_position

                    for _ in range(0,int(num_new_position)):
                        new_position = {}
                        new_position['quantity'] = np.floor(per_position_value/ask_price)
                        new_position['price'] = ask_price
                        self.cash -= new_position['quantity']*new_position['price']
                        self.positions.append(new_position)
                        execute_action = True
                        execute_buy = True

                    self.positions = sorted(self.positions,key = lambda k : k['price'])


            elif r<0:
                #sell

                num_sell_position = len(self.positions)-action
                
                for _ in range(0,int(num_sell_position)):
                    sold_position = self.positions.pop(0) #sell the cheapest first
                    self.cash += sold_position['quantity']*bid_price
                    execute_action = True
                    execute_sell = True


        self.current_time_index += 1

        self.ask_price_history.pop(0)
        self.ask_price_history.append(self.ask_price_list[self.current_time_index])
        self.bid_price_history.pop(0)
        self.bid_price_history.append(self.bid_price_list[self.current_time_index])
        self.middle_price_history.pop(0)
        self.middle_price_history.append(self.middle_price_list[self.current_time_index])

        ask_price = self.ask_price_history[-1]
        bid_price = self.bid_price_history[-1]


        value_in_stock = 0
        for position in self.positions:
            value_in_stock += position['quantity']*bid_price
        
        self.current_portfolio_value = self.cash + value_in_stock
        current_stock_portfolio_ratio = value_in_stock/self.current_portfolio_value

        observation_dict = {}
        price_for_percentage_computation = self.middle_price_history[-1*self.min_history_length:]
        observed_price_history = [100.0 * a1 / a2 - 100 for a1, a2 in zip(price_for_percentage_computation[1:], price_for_percentage_computation)]
        observation_dict['price_history'] = observed_price_history
        observation_dict['current_stock_ratio'] = current_stock_portfolio_ratio 
        observation_dict['current_portfolio_value'] = self.current_portfolio_value       
        observation_dict['action'] = action

        reward = 0
        
        if execute_buy:
            #reward for this action is given if the 
            #future price is high
            future_average_price = np.mean(self.bid_price_list[self.current_time_index:self.current_time_index+100])

            if future_average_price > ask_price:
                reward = 1*r
            else:
                reward = -1*r

        if execute_sell:
            #reward for this action is given if the 
            #future price is low
            future_average_price = np.mean(self.ask_price_list[self.current_time_index:self.current_time_index+100])

            if future_average_price < bid_price:
                reward = 1*(r*-1) #because r is negative in the case of sell
            else:
                reward = -1*(r*-1)
            

        return observation_dict,reward,execute_action
    