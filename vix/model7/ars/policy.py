import numpy as np
import config as C


class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
        self.max_sample_directions = C.max_sample_directions
        self.exploration_size = C.exploration_size
        self.max_best_sample_directions = C.max_best_sample_directions
        self.learning_rate = C.learning_rate

    def evaluate(self, input, delta=None, direction=None):

        #delta is generated from sample_deltas
        #delta is meant for the direction of exploration
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + self.exploration_size * delta).dot(input)
        else:
            return (self.theta - self.exploration_size * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.max_sample_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.learning_rate / (self.max_best_sample_directions * sigma_r) * step