
total_number_of_simulation = 1200
BATCH_SIZE_PER_THREAD = 3
UPDATE_SIZE = 18                                        # batch size per model update
NUM_EPOCHS = int(total_number_of_simulation/(UPDATE_SIZE/BATCH_SIZE_PER_THREAD))
HIDDEN_SIZE = 128                                       # number of hidden nodes we have in our dnn
ALPHA = 1*1e-3                                            # learning rate
GAMMA = 0.99                                            # discount rate
BETA = 0.1                                              # the entropy bonus multiplier

trained_model_name = "nn_model_1_reward_type_4.pt"

use_cuda = True

max_simulation_day = 60
num_minutes_per_trading_day = (6*2+1)*30
data_interval_minute = 5
max_simulation_length = int(max_simulation_day*num_minutes_per_trading_day/data_interval_minute) #in unit of interval
max_history_day = 5
min_history_length = int(max_history_day*num_minutes_per_trading_day/data_interval_minute) #in unit of interval, the 30 means 30 days
max_position = 10
init_cash_value = 1e5
#we also observe what percent of the portfolio is in stock
#therefore we need to add 1
#however, since we use percentage change, we loose 1 because of the percentage change computation
observation_space_size = int(min_history_length-1+1) 
action_space_size = int(max_position+1) #we also can have 0 position

#observation_space_size = 4
#action_space_size = 2