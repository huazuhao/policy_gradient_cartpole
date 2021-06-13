
total_number_of_simulation = 400
BATCH_SIZE = 2      # batch size per model update
NUM_EPOCHS = int(total_number_of_simulation/BATCH_SIZE)
HIDDEN_SIZE = 128   # number of hidden nodes we have in our dnn
ALPHA = 1e-3        # learning rate
GAMMA = 0.99        # discount rate
BETA = 0.1          # the entropy bonus multiplier


use_cuda = True

max_simulation_length = 200

observation_space_size = 4
action_space_size = 2
