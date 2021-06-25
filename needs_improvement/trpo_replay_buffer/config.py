max_new_episode = 100
training_batch_size = 100
ppo_clip = 0.1

alpha = 5e-4 #learning rate, this should be useless for trpo
variance_for_exploration = 0.5

hidden_size = 128
device = 'cpu'
replay_buffer_size = 100
max_offline_training = 1

#some hyperparameters for trpo
damping = 1e-3
max_kl = 1e-3