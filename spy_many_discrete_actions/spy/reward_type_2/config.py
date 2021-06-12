NUM_EPOCHS = 30
ALPHA = 5e-3        # learning rate
BATCH_SIZE = 10      # how many episodes we want to pack into an epoch
GAMMA = 0.99        # discount rate
HIDDEN_SIZE = 128    # number of hidden nodes we have in our dnn
BETA = 0.1          # the entropy bonus multiplier




max_simulation_day = 30
num_minutes_per_trading_day = (6*2+1)*30
data_interval_minute = 5
max_simulation_length = int(max_simulation_day*num_minutes_per_trading_day/data_interval_minute) #in unit of interval
max_history_day = 2
min_history_length = int(max_history_day*num_minutes_per_trading_day/data_interval_minute) #in unit of interval, the 30 means 30 days
max_position = 10
init_cash_value = 1e5
#we also observe what percent of the portfolio is in stock
#therefore we need to add 1
#however, since we use percentage change, we loose 1 because of the percentage change computation
observation_space_size = int(min_history_length-1+1) 
action_space_size = int(max_position+1) #we also can have 0 position