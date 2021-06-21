max_new_episode = 25
training_batch_size = 25
ppo_clip = 0.1

alpha = 5e-4 #learning rate, this should be useless for trpo
variance_for_exploration = 0.5

hidden_size = 128
device = 'cpu'
replay_buffer_size = 500

#some hyperparameters for trpo
damping = 1e-2
max_kl = 1e-3