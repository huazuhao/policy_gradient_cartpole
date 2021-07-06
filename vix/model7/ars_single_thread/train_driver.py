import utils
import numpy as np
import trading_vix_env
import policy
import train
import config as C


if __name__ == '__main__':

    np.random.seed(0)
    env = trading_vix_env.trading_vix_env()
    env.seed(0)
    policy_outputs_size = env.action_space.shape[0]
    policy = policy.Policy(C.extracted_feature_size, policy_outputs_size)
    normalizer = utils.Normalizer(C.extracted_feature_size)
    train.train(env, policy, normalizer)